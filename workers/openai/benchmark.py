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

nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

WAV_RATE = 16000
WAV_BYTES_PER_SECOND = WAV_RATE * 2     # 16-bit mono
# One uploaded clip, the reference the transcription workload is counted against
# (core.REF_AUDIO_BYTES). synthetic_wav() defaults to exactly this, so the benchmark
# request weighs one reference request like every other candidate's.
REF_AUDIO_BYTES = 1024 * 1024


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


def synthetic_wav(seconds: float = REF_AUDIO_BYTES / WAV_BYTES_PER_SECOND,
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
