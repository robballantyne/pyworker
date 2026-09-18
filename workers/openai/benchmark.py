"""Benchmark payloads for the OpenAI worker, one per route that can be benchmarked.

Which route is benchmarked is a property of the deployment, not of the engine: an omni
model may serve chat and speech, and only the template knows which the endpoint exists
for. BENCHMARK_ROUTE names it; the worker attaches a BenchmarkConfig to that route alone,
and the default is /v1/completions, so LLM workers benchmark exactly what they did before.

Every benchmark request is one reference-sized request in the worker's workload units, so
the score means the same thing whichever route is benchmarked.
"""

import math
import os
import random
import struct
import wave
from dataclasses import dataclass
from io import BytesIO
from typing import Callable, Optional

import aiohttp
import nltk

# One template serves both lanes; on-demand templates set only the engine var, so the
# benchmark recovers the model id from it. Only one is ever set.
_MODEL_NAME_VARS = ("MODEL_NAME", "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL")

# The corpus is a RUNTIME DOWNLOAD, not a pip dependency: requirements.txt installs the
# nltk package, and nltk then fetches `words` over the network the first time it is used.
# On a fresh worker that cannot reach the CDN -- offline, rate-limited, or simply behind
# a proxy, which nltk now refuses to fetch through by default (CWE-918 hardening) -- the
# corpus lookup raises LookupError AT IMPORT, and since core.py imports this module that
# takes the whole worker down, every route with it, not just the benchmark. Measured on
# a clean HOME with the network blocked.
#
# So the corpus stays the preferred source (identical benchmark payloads to every run
# before this, and the tgi worker uses it too) and a builtin list is the floor.
_FALLBACK_WORDS = (
    "absence action amount animal answer autumn balance barrier beacon border bottle "
    "branch bridge builder candle canvas carpet caution central chamber change circle "
    "clarity cluster coastal command compass concert content copper corner cotton "
    "council counter courage crystal culture current custom damage danger declare "
    "deliver density deposit desert detail develop device diamond digital distance "
    "district drawing driver eastern economy edition element energy engine evening "
    "example exhibit expert fabric factor family feather feature figure filter "
    "finance fixture flavour forest formal fortune forward founder fragment freedom "
    "friend function furnace gallery garden gateway gather general gesture glacier "
    "granite gravity ground habitat harbour harvest heading healthy hearing heavy "
    "helper history holiday horizon hunter husband illness imagine impact improve "
    "include initial inside instant invite island jacket journey justice keeper "
    "kitchen ladder landing language lantern leader leather lecture legend length "
    "lesson letter liberty library license lighting limited liquid listen litter "
    "machine magnet manner marble margin market master matter meadow measure medical "
    "meeting member memory mention message metal method middle mineral minute mirror "
    "mixture modern moment monitor morning mother motion mountain museum musical "
    "narrow nation native natural neither network neutral nothing notice number "
    "object observe ocean office opening operate opinion orange orbit order organic "
    "origin outdoor outline output palace parcel parent partner passage pattern "
    "payment pencil people perfect period person picture pioneer planet plastic "
    "player pocket poetry portion position possible powder prairie precise prepare "
    "present pressure primary printer private problem process produce program project "
    "promise protect provide public purpose quality quarter question quiet rabbit "
    "radius railway rainbow random reader reason recover reflect region regular "
    "related release remain remote repair report request reserve resolve respect "
    "result return reveal ribbon river roster routine sample sandy scatter science "
    "season second section segment select senior series service session settle "
    "shadow shelter signal silence silver similar simple singer sister skill slope "
    "social soldier solid source special speech spirit spring square stable station "
    "steady stone storage stream street strong studio subject success sudden summer "
    "sunset supply support surface survey symbol system table talent target teacher "
    "temple tender tension theory thread through thunder timber tissue title toward "
    "traffic transfer travel treasure treaty triangle tribute trouble tunnel turning "
    "uniform unique united unusual update upper urgent useful valley value vapour "
    "various vehicle velvet verdict version vessel victory village vintage violet "
    "virtue vision visitor voice volume voyage wander warning water weather weaving "
    "welcome western whisper willow window winter wisdom wonder wooden worker "
    "working worthy writing yellow"
).split()


def _load_words():
    try:
        nltk.download("words", quiet=True)
        words = nltk.corpus.words.words()
        if words:
            return words
    except Exception as exc:
        print(f"WARNING: nltk words corpus unavailable ({type(exc).__name__}); "
              f"benchmarking with the builtin word list", flush=True)
    return list(_FALLBACK_WORDS)


WORD_LIST = _load_words()

WAV_RATE = 16000
WAV_BYTES_PER_SECOND = WAV_RATE * 2     # 16-bit mono
# One uploaded clip, the reference the transcription workload is counted against, in
# SECONDS of audio rather than bytes: the same megabyte is 26 s of 320 kbps MP3 or 349 s
# of 24 kbps Opus, so bytes charged a 13x spread identically -- and under-charged the
# slow end, which is the direction that makes the queue estimate optimistic. 30 s is one
# Whisper window, and synthetic_wav() defaults to exactly this, so the benchmark request
# weighs one reference request like every other candidate's.
REF_AUDIO_SECONDS = 30.0


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


def synthetic_wav(seconds: float = REF_AUDIO_SECONDS,
                  rate: int = WAV_RATE) -> bytes:
    """Quiet noise as 16-bit mono WAV, one reference clip long by default.

    Not speech, so a transcription benchmark measures the encoder more faithfully than
    the decoder. It needs no bundled asset, and without some benchmark a
    transcription-only model never becomes ready.
    """
    frames = int(seconds * rate)
    samples = (int(300 * math.sin(i / 7.0) + random.randint(-200, 200)) for i in range(frames))
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return buf.getvalue()


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
    return {**_model(), "input": _words(2000)}


def speech_benchmark_generator() -> dict:
    return {**_model(), "input": _words(500), **_voice()}


def images_benchmark_generator() -> dict:
    return {**_model(), "prompt": _words(60), "size": "1024x1024", "n": 1}


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
    # the payload class builds this one, because it is an upload
    "/v1/audio/transcriptions": Benchmark(4, 2),
}
DEFAULT_BENCHMARK_ROUTE = "/v1/completions"
