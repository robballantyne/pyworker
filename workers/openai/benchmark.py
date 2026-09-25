"""Benchmark payloads for the OpenAI worker, one per route that can be benchmarked.

Which route is benchmarked is a property of the deployment, not of the engine: an omni
model may serve chat and speech, and only the template knows which the endpoint exists
for. BENCHMARK_ROUTE names it; the worker attaches a BenchmarkConfig to that route alone,
and the default is /v1/completions, so LLM workers benchmark exactly what they did before.

Every benchmark request is one reference-sized request in the worker's workload units, so
the score means the same thing whichever route is benchmarked.
"""

import array
import math
import os
import random
import re
import sys
import urllib.parse
import struct
import urllib.request
import wave
import zlib
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Iterable, Optional, Tuple

import aiohttp
import nltk

# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

WAV_RATE = 16000
# One reference image, per side. core.REF_IMAGE_PIXELS is this squared, and a test
# pins the two together: every image benchmark must weigh one reference request.
REF_IMAGE_SIDE = 1024
WAV_BYTES_PER_SECOND = WAV_RATE * 2     # 16-bit mono
# One uploaded clip, the reference the transcription workload is counted against, in
# SECONDS of audio rather than bytes: the same megabyte is 26 s of 320 kbps MP3 or 349 s
# of 24 kbps Opus, so bytes charged a 13x spread identically -- and under-charged the
# slow end, which is the direction that makes the queue estimate optimistic. 30 s is one
# Whisper window, and synthetic_wav() defaults to exactly this, so the benchmark request
# weighs one reference request like every other candidate's.
REF_AUDIO_SECONDS = 30.0

# One request's worth of text to embed. MEASURED 2026-09-21 against BAAI/bge-small-en-v1.5.
#
# At 2000 chars the benchmark 400'd every request and the worker never became ready:
# 1999 characters tokenised to 513 tokens, ONE over that model's 512 limit, and vLLM
# rejects an over-length pooling input rather than truncating it.
#
# Sized for the SMALLEST mainstream encoder rather than the largest, because the tokens
# per character VARY WITH THE DRAW and a reference near any limit passes or fails by
# luck. Measured over five draws each: 500 chars -> 133-160 tokens, 600 -> 177-186,
# 800 -> 232-248. all-MiniLM-L6 caps at 256, so 800 leaves 3% of margin and 600 leaves
# 27%; the bge/e5/gte family at 512 has room either way.
#
# `truncate_prompt_tokens: -1` also works on vLLM and was measured returning 200 on the
# over-length payload, but it is NOT sent: SGLang and llama.cpp serve this route too and
# an engine that rejects an unknown field would fail the benchmark outright, which is
# the failure being fixed. BENCHMARK_EMBED_CHARS is the escape hatch instead.
REF_EMBED_CHARS = int(os.environ.get("BENCHMARK_EMBED_CHARS", 600))


def resolve_model_name() -> Optional[str]:
    return next((v for var in _MODEL_NAME_VARS if (v := os.environ.get(var))), None)


def _model() -> dict:
    model = resolve_model_name()
    return {"model": model} if model else {}


def _words(chars: int) -> str:
    """Random dictionary words, about `chars` characters long."""
    out: List[str] = []
    size = 0
    while size < chars:
        word = random.choice(WORD_LIST)
        out.append(word)
        size += len(word) + 1
    return " ".join(out)


def _voice() -> dict:
    # Engines disagree on voice names, and the spec requires one; the probe and the
    # benchmark send none unless told which to use.
    voice = os.environ.get("BENCHMARK_SPEECH_VOICE")
    return {"voice": voice} if voice else {}


def synthetic_png(side: int = REF_IMAGE_SIDE, tile: int = 64) -> bytes:
    """A random RGB PNG, one reference image in size, built from the stdlib alone.

    The input to the edit benchmark. Synthetic rather than fetched, unlike the speech
    clip, and deliberately: a transcription's cost depends on the audio (real speech and
    noise differed about 2x), but a diffusion edit runs a fixed number of steps at a
    fixed resolution whatever the pixels are. Measured, not assumed: FLUX.2-klein-4B on
    an RTX PRO 6000, five alternating 1024x1024 edits each, engine-direct -- this noise
    median 11.526s, a real photograph median 11.564s, ratio 1.003.

    Built by tiling one random `tile`-pixel block, for the reason synthetic_wav tiles:
    for_test() runs inside the SDK's timed window, so its cost is charged to the engine.
    A tiled 1024x1024 builds in about 11 ms and compresses to about 0.2 MB.

    Re-rolled per call and that is load-bearing: engines cache processed multimodal
    input by content hash, so an image that was byte-identical every request would
    measure the cache after the first one.
    """
    rows = [os.urandom(tile * 3) * (side // tile) for _ in range(tile)]
    raw = b"".join(b"\x00" + rows[y % tile] for y in range(side))   # filter 0 per row

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)   # 8-bit RGB
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def synthetic_wav(seconds: float = REF_AUDIO_SECONDS,
                  rate: int = WAV_RATE) -> bytes:
    """Quiet noise as 16-bit mono WAV, one reference clip long by default.

    Not speech, so a transcription benchmark measures the encoder more faithfully than
    the decoder. It needs no bundled asset, and without some benchmark a
    transcription-only model never becomes ready.

    Built by tiling ONE second rather than generating every sample, because the SDK
    calls make_benchmark_payload() inside the timed window (backend.py: the timer starts
    before the request loop). Generating 480,000 samples through struct.pack cost 1.10s
    per four clips -- measured at ~41% of a real 2.18s benchmark window, charged to the
    engine and depressing max_throughput by roughly half. Tiling is 0.024s for the same
    four, and the encoder sees the same 30 seconds either way.

    The randomness is re-rolled per call and that is load-bearing: engines cache
    processed multimodal input by content hash, so a clip that was byte-identical every
    request would measure the cache after the first one.
    """
    frames = int(seconds * rate)
    block = array.array("h", (int(300 * math.sin(i / 7.0) + random.randint(-200, 200))
                              for i in range(min(frames, rate))))
    if not block:
        block = array.array("h", [0])
    samples = array.array("h")
    while len(samples) < frames:
        samples.extend(block)
    del samples[frames:]
    if sys.byteorder == "big":
        samples.byteswap()                     # WAV data is little-endian
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return buf.getvalue()


# Real speech for the transcription benchmark.
#
# MEASURED on a live whisper-large-v3 instance: quiet noise transcribes to 11 characters
# where 30s of speech gives 289, so the decoder is idle and the benchmark reports ~2x the
# throughput the engine actually has on real audio (158 vs 74 audio-seconds/second at
# concurrency 1; the same ratio at 2, 4 and 8). An ASR score measured on noise is an
# encoder score.
#
# DELIVERY IS AN OPEN DECISION. Downloading at boot matches how the word corpus already
# arrives, which is the cheapest thing that works today -- and the corpus is also the
# warning: a boot fetch that is REQUIRED takes the whole worker down when it fails. So
# this one is optional by construction, and falls back to synthetic noise with a warning.
# Before this is something customers depend on, pick one deliberately: vendor the clip in
# the repo (weight + licence review), bake it into the engine images (no network, but the
# worker then needs the image to carry it), or serve it from infrastructure we control
# (no third-party availability risk). See the notes for the trade-offs.
BENCHMARK_AUDIO_URL = os.environ.get(
    "BENCHMARK_AUDIO_URL",
    "https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav")
_MAX_BENCHMARK_AUDIO_BYTES = 8 * 1024 * 1024
_BENCHMARK_AUDIO_TIMEOUT = float(os.environ.get("BENCHMARK_AUDIO_TIMEOUT", 20))
_speech_clip: Optional[Tuple] = None       # None = not tried, () = unavailable


def _fetch_speech(url: str) -> Optional[bytes]:
    try:
        with urllib.request.urlopen(url, timeout=_BENCHMARK_AUDIO_TIMEOUT) as resp:
            return resp.read(_MAX_BENCHMARK_AUDIO_BYTES)
    except Exception as exc:
        print(f"benchmark speech sample could not be fetched: {type(exc).__name__}",
              flush=True)
        return None


def _tile_wav(raw: bytes, seconds: float) -> Optional[bytes]:
    """Repeat a clip up to `seconds` exactly, keeping its own format."""
    try:
        with wave.open(BytesIO(raw)) as r:
            channels, width, rate = r.getnchannels(), r.getsampwidth(), r.getframerate()
            frames = r.readframes(r.getnframes())
        if not frames:
            return None
        want = int(seconds * rate) * width * channels
        data = (frames * (want // len(frames) + 1))[:want]
        buf = BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(width)
            w.setframerate(rate)
            w.writeframes(data)
        return buf.getvalue()
    except Exception:
        return None


def _clip_filename(url: str) -> str:
    """The engine picks its decoder from the extension, so the name has to survive the
    URL: its path basename, without the query string a signed URL carries."""
    name = urllib.parse.urlparse(url).path.rpartition("/")[2]
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".")
    return name or "benchmark.wav"


def benchmark_audio(seconds: float = REF_AUDIO_SECONDS,
                    accepted: Iterable[str] = ("wav",),
                    probe: Optional[Callable[[bytes], Optional[float]]] = None
                    ) -> Tuple[bytes, str]:
    """One reference clip of REAL speech, with the filename the engine should see.

    Fetched once per process, on first use -- which is the warmup request, before the
    timed runs -- and never at import, so a worker whose network is blocked still starts.

    WAV is normalised: tiled or truncated to exactly `seconds`, so a benchmark request
    weighs exactly one reference request. Anything else is used AS IS, because re-encoding
    it would need a decoder this worker does not have: the score is still workload over
    time, and the workload is priced from the clip's real duration, so it stays a valid
    throughput -- it just is not exactly one reference request any more.

    Falls back to synthetic noise, loudly. `accepted` comes from the route's own format
    table, so this cannot offer the engine a container the worker would refuse.
    """
    global _speech_clip
    if _speech_clip is None:
        raw = _fetch_speech(BENCHMARK_AUDIO_URL)
        name = _clip_filename(BENCHMARK_AUDIO_URL)
        ext = name.rpartition(".")[2].lower()
        clip = None
        if raw:
            if raw.startswith(b"RIFF"):
                tiled = _tile_wav(raw, seconds)
                clip = (tiled, "benchmark.wav") if tiled else None
            elif ext in set(accepted) and probe and probe(raw):
                clip = (raw, name)
                print(f"benchmark speech sample: {name} used as supplied "
                      f"(only WAV can be resized to the {seconds:.0f}s reference, and "
                      f"only WAV varies per request -- an engine that caches processed "
                      f"audio by content hash may serve this one from its cache)",
                      flush=True)
        _speech_clip = clip or ()
        if not _speech_clip:
            # Warned here rather than at the fetch, so an unreadable download -- a 404
            # page is bytes too, and wave will not open it -- is as loud as no download,
            # and says WHICH it was: an operator who just set BENCHMARK_AUDIO_URL needs
            # to know their clip was rejected rather than their network.
            if raw is None:
                why = "could not be fetched"
            elif ext not in set(accepted):
                why = (f"has an extension this route does not accept ({ext or 'none'}; "
                       f"expected one of {', '.join(sorted(accepted))})")
            else:
                why = f"is not audio this can read ({len(raw)} bytes from {BENCHMARK_AUDIO_URL})"
            print(f"WARNING: the benchmark speech sample {why}; benchmarking "
                  f"transcription on synthetic noise instead, which leaves the decoder "
                  f"idle and overstates real-speech throughput by roughly 2x "
                  f"(measured on whisper-large-v3)", flush=True)

    if not _speech_clip:
        return synthetic_wav(seconds), "benchmark.wav"

    data, name = _speech_clip
    if not data.startswith(b"RIFF"):
        return data, name           # cannot vary a container we cannot rebuild
    # Engines cache processed multimodal input by content hash, so the bytes must differ
    # per request or the benchmark measures the cache. Re-rolling the tail is enough and
    # leaves >99% of the audio -- and all of its speech content -- identical.
    tail = 4096
    return data[:-tail] + random.randbytes(min(tail, len(data) // 2)), name


# Benchmark payloads. Each is one reference-sized request (see core.py's workload units).

def completions_benchmark_generator() -> dict:
    model = resolve_model_name()
    if not model:
        raise ValueError("No model set: MODEL_NAME / VLLM_MODEL / SGLANG_MODEL / LLAMA_MODEL all empty")
    prompt = " ".join(random.choices(WORD_LIST, k=250))
    return {"model": model, "prompt": prompt, "temperature": 0.7, "max_tokens": 500}


def chat_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=250))
    return {**_model(), "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7, "max_tokens": 500}


def embeddings_benchmark_generator() -> dict:
    return {**_model(), "input": _words(REF_EMBED_CHARS)}


def speech_benchmark_generator() -> dict:
    return {**_model(), "input": _words(500), **_voice()}


def images_benchmark_generator() -> dict:
    return {**_model(), "prompt": _words(60), "size": f"{REF_IMAGE_SIDE}x{REF_IMAGE_SIDE}", "n": 1}


def transcription_form(audio: bytes) -> aiohttp.FormData:
    """The multipart body the transcription benchmark sends."""
    form = aiohttp.FormData(default_to_multipart=True)
    form.add_field("file", audio, filename="benchmark.wav", content_type="audio/wav")
    for key, value in _model().items():
        form.add_field(key, value)
    return form


@dataclass(frozen=True)
class Benchmark:
    concurrency: int
    runs: int
    # None: the route's payload class builds benchmark payloads itself (for_test).
    generator: Optional[Callable[[], dict]] = None


BENCHMARKS = {
    "/v1/completions": Benchmark(10, 3, completions_benchmark_generator),
    "/v1/chat/completions": Benchmark(10, 3, chat_benchmark_generator),
    "/v1/embeddings": Benchmark(10, 3, embeddings_benchmark_generator),
    "/v1/audio/speech": Benchmark(4, 2, speech_benchmark_generator),
    "/v1/images/generations": Benchmark(2, 1, images_benchmark_generator),
    # the payload class builds these, because they are uploads. Edits needs its own
    # benchmark because some models are edit-only: without it such a deployment
    # could neither benchmark on edits (BENCHMARK_ROUTE refused it at startup) nor
    # on generations (which its model does not serve), so it never became ready.
    "/v1/images/edits": Benchmark(2, 1),
    "/v1/audio/transcriptions": Benchmark(4, 2),
    # Its own entry rather than "benchmark transcriptions instead": a deployment that
    # exposes only translations (OPENAI_ROUTES) would otherwise have no route it could
    # both serve and benchmark. Same clip, same weight -- the encoder work is identical
    # and the decoder emits text of similar length either way.
    "/v1/audio/translations": Benchmark(4, 2),
}
DEFAULT_BENCHMARK_ROUTE = "/v1/completions"
