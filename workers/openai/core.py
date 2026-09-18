"""Shared core for the OpenAI-compatible workers (vllm/sglang/llama/openai).

They all proxy the same /v1/completions + /v1/chat/completions API, so the logic lives
here and the per-engine adapters just pass an EngineDefaults. Every default is
env-overridable: the image is version-locked to the engine, so it owns the
engine/version-specific values (log path, health endpoint, log grammar)."""

import base64
import binascii
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig
from vastai.serverless.server.lib.data_types import ApiPayload, JsonDataException

from workers.openai.benchmark import (
    BENCHMARKS,
    DEFAULT_BENCHMARK_ROUTE,
    REF_AUDIO_BYTES,
    completions_benchmark_generator,  # noqa: F401  (re-exported)
    resolve_model_name as _resolve_model_name,
    synthetic_wav,
)


def _env_lines(name, default):
    """Newline-delimited env var -> list of stripped lines; default if unset/empty."""
    raw = os.environ.get(name)
    return [s for ln in raw.splitlines() if (s := ln.strip())] if raw else default


@dataclass(frozen=True)
class EngineDefaults:
    """Per-engine baked defaults; each is overridden by the matching env var if set."""

    name: str                 # engine id for the startup banner
    model_log_file: str       # MODEL_LOG
    load_log_msgs: List[str]  # MODEL_LOAD_LOG_MSG — model-loaded markers
    error_log_msgs: List[str]  # MODEL_ERROR_LOG_MSGS — failed-load markers
    info_log_msgs: List[str] = field(default_factory=lambda: ['"message":"Download'])  # MODEL_INFO_LOG_MSGS


MODEL_SERVER_URL = "http://127.0.0.1"
MODEL_SERVER_PORT = 18000

def request_parser(request):
    return request["input"] if request.get("input") is not None else request


DEFAULT_AUDIO_FILENAME = "audio.wav"
DEFAULT_IMAGE_FILENAME = "image.png"



def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        print(f"WARNING: {name} is not an integer; using {default}", flush=True)
        return default


# Uploads are buffered, base64-decoded and re-encoded before the SDK checks the request
# signature, so they are bounded here: per file (25 MiB is the OpenAI limit) and per
# request, across every file and inline reference in it.
MAX_UPLOAD_BYTES = _env_int("WORKER_MAX_UPLOAD_BYTES", 25 * 1024 * 1024)
MAX_REQUEST_UPLOAD_BYTES = _env_int("WORKER_MAX_REQUEST_UPLOAD_BYTES", 64 * 1024 * 1024)
MAX_UPLOAD_FILES = 16

# The content type is chosen from the extension, and engines pick a decoder from it, so
# only the formats each spec route accepts are allowed -- and the table is fixed rather
# than read from the host's mime database, which differs between images.
AUDIO_TYPES = {
    "flac": "audio/flac", "m4a": "audio/mp4", "mp3": "audio/mpeg", "mp4": "audio/mp4",
    "mpeg": "audio/mpeg", "mpga": "audio/mpeg", "ogg": "audio/ogg", "wav": "audio/wav",
    "webm": "audio/webm",
}
IMAGE_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp"}

# Reference fields an engine resolves for itself: only remote http(s) and inline data are
# forwarded. A local path (file://, or a bare /path an engine might open) would read the
# instance's own disk, so a value with no scheme must be real base64 of real size.
# Remote addresses are not resolved or filtered here; that is the engine's fetcher.
REFERENCE_SCHEMES = ("http", "https")
MIN_INLINE_BYTES = 64
MAX_DATA_URI_PREFIX = 128           # "data:audio/wav;base64" and the like

# Workload unit. The SDK's wait_time divides the in-flight workload of EVERY route by a
# max_throughput measured only on /v1/completions, in max_tokens, and 429s any request
# arriving while that exceeds max_queue_time. So every route must count in the same
# unit, or one upload (bytes, pixels) 429s the whole worker.
#
# Non-token routes count in benchmark requests: one reference-sized request weighs the
# same as one benchmark completion, and no request weighs more than
# MAX_REQUEST_MULTIPLE of them. The reference sizes are ESTIMATES, not measurements --
# the clamp is what keeps the gate safe, and it does not depend on them being right.
BENCHMARK_MAX_TOKENS = 500          # completions_benchmark_generator's max_tokens
MIN_REQUEST_MULTIPLE = 0.1
MAX_REQUEST_MULTIPLE = 8.0

REF_IMAGE_PIXELS = 1024 * 1024      # one image; also used when `size` is absent or "auto"
# REF_AUDIO_BYTES: an uploaded clip. Defined in benchmark.py, where synthetic_wav() has
# to produce exactly one of them.
REF_SPEECH_CHARS = 500              # text to synthesise; each clone reference adds one
REF_EMBED_CHARS = 2000              # text to embed
CHARS_PER_TOKEN = 4                 # sizes pre-tokenised embedding input
MAX_IMAGES = 10                     # the spec's ceiling on `n`
MAX_IMAGE_SIDE = 16384


def _in_request_units(size: float, reference: float) -> float:
    multiple = min(max(size / reference, MIN_REQUEST_MULTIPLE), MAX_REQUEST_MULTIPLE)
    return BENCHMARK_MAX_TOKENS * multiple


def _bounded_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError, OverflowError):
        return default


def _decode_b64(value: Any, field: str) -> bytes:
    """A base64 string -> bytes, or a JsonDataException naming the offending field.

    Accepts a data: URI and line-wrapped base64 (both common encoder outputs). The size
    limit is checked on the encoded length, before anything is decoded.
    """
    if not isinstance(value, str):
        raise JsonDataException({field: "must be a base64 string"})
    limit = (MAX_UPLOAD_BYTES + 2) // 3 * 4 + len(value) // 76 * 2 + 4
    if len(value) > limit + MAX_DATA_URI_PREFIX:
        # Checked before the prefix is stripped, or everything before the first comma
        # would be unbounded.
        raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})
    if value.startswith("data:"):
        prefix, _, rest = value.partition(",")
        if len(prefix) > MAX_DATA_URI_PREFIX:
            raise JsonDataException({field: "malformed data: URI"})
        value = rest
    if len(value) > limit:
        raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})
    value = re.sub(r"\s+", "", value)
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise JsonDataException({field: "not valid base64"})
    if not raw:
        raise JsonDataException({field: "decoded to zero bytes"})
    if len(raw) > MAX_UPLOAD_BYTES:
        raise JsonDataException({field: f"larger than {MAX_UPLOAD_BYTES} bytes"})
    return raw


def _file_part(raw: bytes, filename: Any, default: str, types: Dict[str, str],
               field: str) -> tuple:
    """(filename, bytes, content_type) -- the shape the SDK turns into a file part.

    The filename is reduced to a safe basename, and its extension must be one the route
    accepts; the content type comes from that extension.
    """
    name = os.path.basename(str(filename or "")).strip()
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name).lstrip(".") or default
    # Truncate the stem, not the name: cutting a long name mid-extension would reject a
    # legitimate upload for having no extension at all.
    stem, dot, ext = name.rpartition(".")
    name = (stem[:128] + dot + ext) if dot else name[:128]
    ext = ext.lower() if dot else ""
    if ext not in types:
        raise JsonDataException(
            {field: f"unsupported file type {ext or '(none)'!r}; "
                    f"expected one of {', '.join(sorted(types))}"})
    return (name, raw, types[ext])


def _check_budget(values: List[Any], field: str) -> None:
    """Refuse a request whose inline data, together, exceeds the per-request budget.

    Checked on the encoded length, before anything is decoded.
    """
    encoded = sum(len(v) for v in values if isinstance(v, str))
    if encoded * 3 // 4 > MAX_REQUEST_UPLOAD_BYTES:
        raise JsonDataException(
            {field: f"request uploads exceed {MAX_REQUEST_UPLOAD_BYTES} bytes in total"})


def _flatten(values: List[Any]) -> List[Any]:
    return [v for value in values
            for v in (value if isinstance(value, list) else [value])]


def _check_reference(value: Any, field: str) -> None:
    """Reject reference values the engine would resolve against the instance itself.

    Budget the request (_check_budget) first: inline values are decoded here.
    """
    values = value if isinstance(value, list) else [value]
    if len(values) > MAX_UPLOAD_FILES:
        raise JsonDataException({field: f"at most {MAX_UPLOAD_FILES} values"})
    for item in values:
        if item is None:
            continue
        if not isinstance(item, str):
            raise JsonDataException({field: "must be a string"})
        scheme = urlparse(item).scheme.lower()
        if scheme in REFERENCE_SCHEMES:
            continue
        if scheme in ("", "data"):
            # Inline audio or image: must decode, and be more than a short string that
            # merely happens to be valid base64 (a path like /tmp/abcd can be).
            if len(_decode_b64(item, field)) < MIN_INLINE_BYTES:
                raise JsonDataException({field: "inline data is too short to be media"})
            continue
        raise JsonDataException({field: "only http(s) URLs or inline base64 are accepted"})


def _parse_body(json_msg: Any) -> Dict[str, Any]:
    """The request as a dict, unwrapping the `input` wrapper the other routes accept."""
    if not isinstance(json_msg, dict):
        raise JsonDataException({"payload": "must be an object"})
    fields = request_parser(json_msg)
    if not isinstance(fields, dict):
        raise JsonDataException({"input": "must be an object"})
    return dict(fields)


def _fill_model(fields: Dict[str, Any]) -> None:
    # The engine needs a model id; on-demand templates set only the engine var.
    if not fields.get("model"):
        model = _resolve_model_name()
        if model:
            fields["model"] = model


class _UploadPayload(ApiPayload):
    """Shared by the routes whose spec body is multipart: never sent as JSON."""

    ROUTE = ""

    @classmethod
    def for_test(cls):
        # Benchmarking these routes needs a real sample (silence transcribes to nothing)
        # and the benchmark route is chosen elsewhere.
        raise NotImplementedError(f"{cls.ROUTE} has no benchmark payload")

    def generate_payload_json(self) -> Dict[str, Any]:
        # Never reached: generate_payload_multipart() returns a mapping, so the backend
        # takes the multipart branch. Loud rather than posting JSON the engine rejects.
        raise NotImplementedError(f"{self.ROUTE} is sent as multipart, not JSON")


class TranscriptionPayload(_UploadPayload):
    """base64 in, multipart out — the envelope is JSON, the OpenAI spec wants a file.

        {"file": "<base64>", "filename": "a.mp3", "model": "...", "language": "en"}

    `filename` sets the content type; engines infer the audio format from it. Other
    fields pass through as form fields.
    """

    ROUTE = "/v1/audio/transcriptions"

    def __init__(self, fields: Dict[str, Any], audio: bytes, filename: str):
        self.fields = fields
        self.audio = audio
        self.filename = filename

    @classmethod
    def for_test(cls) -> "TranscriptionPayload":
        fields: Dict[str, Any] = {}
        _fill_model(fields)
        return cls(fields=fields, audio=synthetic_wav(), filename="benchmark.wav")

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "TranscriptionPayload":
        fields = _parse_body(json_msg)
        raw = fields.pop("file", None)
        if raw is None:
            raise JsonDataException({"file": "field missing"})
        audio = _decode_b64(raw, "file")
        part = _file_part(audio, fields.pop("filename", None), DEFAULT_AUDIO_FILENAME,
                          AUDIO_TYPES, "filename")
        _fill_model(fields)
        return cls(fields=fields, audio=audio, filename=part[0])

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        part = _file_part(self.audio, self.filename, DEFAULT_AUDIO_FILENAME,
                          AUDIO_TYPES, "filename")
        return {"file": part, **self.fields}

    def count_workload(self) -> float:
        # Bytes proxy for duration, which would need a header parse per request.
        return _in_request_units(len(self.audio), REF_AUDIO_BYTES)


class ImageEditPayload(_UploadPayload):
    """base64 in, multipart out, for /v1/images/edits and /variations.

        {"image": "<b64>" | ["<b64>", ...], "filename": "a.png" | [...],
         "mask": "<b64>", "mask_filename": "m.png", "prompt": "..."}

    Engines also accept `url` (or `url[]`) instead of an upload; one of the two is
    required.
    """

    ROUTE = "image edits and variations"

    def __init__(self, fields: Dict[str, Any], files: Dict[str, list]):
        self.fields = fields
        self.files = files

    @classmethod
    def from_json_msg(cls, json_msg: Any) -> "ImageEditPayload":
        fields = _parse_body(json_msg)
        files: Dict[str, list] = {}

        images = fields.pop("image", None)
        names = fields.pop("filename", None)
        mask = fields.pop("mask", None)
        mask_name = fields.pop("mask_filename", None)
        images = images if isinstance(images, list) or images is None else [images]
        if images is not None and len(images) > MAX_UPLOAD_FILES:
            raise JsonDataException({"image": f"at most {MAX_UPLOAD_FILES} files"})
        urls = [fields.get("url"), fields.get("url[]")]
        _check_budget([*(images or []), mask, *_flatten(urls)], "image")
        for key in ("url", "url[]"):
            _check_reference(fields.get(key), key)
        if images is not None:
            names = names if isinstance(names, list) else [names]
            names = names + [None] * (len(images) - len(names))
            files["image"] = [
                _file_part(_decode_b64(img, "image"), name, DEFAULT_IMAGE_FILENAME,
                           IMAGE_TYPES, "filename")
                for img, name in zip(images, names)
            ]

        if mask is not None:
            files["mask"] = [_file_part(_decode_b64(mask, "mask"), mask_name,
                                        DEFAULT_IMAGE_FILENAME, IMAGE_TYPES,
                                        "mask_filename")]

        if not files.get("image") and not (fields.get("url") or fields.get("url[]")):
            raise JsonDataException({"image": "field missing (or pass `url`)"})
        _fill_model(fields)
        return cls(fields=fields, files=files)

    def generate_payload_multipart(self) -> Optional[Dict[str, Any]]:
        return {**self.files, **self.fields}

    def count_workload(self) -> float:
        return _image_workload(self.fields)


def speech_request_parser(request: Any) -> Dict[str, Any]:
    """Validation for /v1/audio/speech. Unlike the shared parser it does NOT unwrap
    `input`: there, `input` is the text to synthesise."""
    if not isinstance(request, dict):
        raise JsonDataException({"payload": "must be an object"})
    refs = [request.get("ref_audio"), request.get("ref_audio_2")]
    _check_budget(_flatten(refs), "ref_audio")
    for key in ("ref_audio", "ref_audio_2"):
        _check_reference(request.get(key), key)
    return request


def _image_workload(data: Dict[str, Any]) -> float:
    """n x pixels, in request units. `size` is "WxH", or "auto" when the engine decides.
    Both are caller-declared, so they are bounded before use."""
    n = _bounded_int(data.get("n") or 1, 1, MAX_IMAGES, 1)
    pixels = REF_IMAGE_PIXELS
    w, sep, h = str(data.get("size") or "").lower().partition("x")
    if sep:
        pixels = (_bounded_int(w, 1, MAX_IMAGE_SIDE, 1024)
                  * _bounded_int(h, 1, MAX_IMAGE_SIDE, 1024))
    return _in_request_units(n * pixels, REF_IMAGE_PIXELS)


def _speech_workload(data: Dict[str, Any]) -> float:
    """Input text, plus one reference-sized request per voice-clone reference.

    A reference costs a speaker-encoder pass the text does not account for. `ref_audio`
    is "URL, base64, or file URI" and may be a list, so references are counted rather
    than measured: a URL's length says nothing about the file it names.
    """
    text = data.get("input")
    chars = len(text) if isinstance(text, str) else 0
    refs = data.get("ref_audio")
    refs = refs if isinstance(refs, list) else [refs]
    refs = [*refs, data.get("ref_audio_2")]
    chars += REF_SPEECH_CHARS * sum(1 for r in refs if isinstance(r, str) and r)
    return _in_request_units(chars, REF_SPEECH_CHARS)


def _embeddings_workload(data: Dict[str, Any]) -> float:
    """Size of the embedding input, in request units.

    `input` is a string, an array of strings, an array of tokens, or an array of token
    arrays. A batch costs about its sum, so items are summed; tokens are converted to
    characters so every shape lands in one scale.
    """
    value = data.get("input")
    items = value if isinstance(value, (list, tuple)) else [value]
    chars = 0
    for item in items:
        if isinstance(item, str):
            chars += len(item)
        elif isinstance(item, (list, tuple)):
            chars += len(item) * CHARS_PER_TOKEN      # a token array
        elif isinstance(item, int):
            chars += CHARS_PER_TOKEN                  # a bare token
    return _in_request_units(chars, REF_EMBED_CHARS)


UPLOAD_ROUTES = ("/v1/images/edits", "/v1/images/variations",
                 "/v1/audio/transcriptions", "/v1/audio/translations")


def benchmark_route() -> str:
    """The route this deployment is benchmarked on.

    An engine can serve several of them, and only the template knows which the endpoint
    exists for: probing the engine would price speech traffic against a chat score.
    """
    route = os.environ.get("BENCHMARK_ROUTE", DEFAULT_BENCHMARK_ROUTE).strip()
    if route not in BENCHMARKS:
        raise RuntimeError(
            f"BENCHMARK_ROUTE={route!r} cannot be benchmarked; expected one of "
            + ", ".join(BENCHMARKS))
    return route


def _served_routes(handlers: List[HandlerConfig]) -> List[HandlerConfig]:
    """The handlers to serve.

    OPENAI_ROUTES (comma-separated) narrows them; unset serves all. The worker is pulled
    from main by every instance at boot, so on an SDK that cannot send multipart it
    degrades instead of failing: the four upload routes are not served, and answer 404.
    """
    known = {h.route for h in handlers}
    wanted = {r.strip() for r in os.environ.get("OPENAI_ROUTES", "").split(",") if r.strip()}
    if wanted - known:
        print("WARNING: OPENAI_ROUTES names unknown routes: "
              + ", ".join(sorted(wanted - known)), flush=True)
    multipart = "generate_payload_multipart" in vars(ApiPayload)
    if not multipart:
        print("ERROR: installed vastai SDK cannot send multipart; not serving "
              + ", ".join(UPLOAD_ROUTES), flush=True)

    served = [h for h in handlers
              if (not wanted or h.route in wanted)
              and (multipart or h.route not in UPLOAD_ROUTES)]

    benchmarked = [h.route for h in served if h.benchmark_config]
    if not benchmarked:
        raise RuntimeError(
            f"the benchmark route {benchmark_route()} is not served; set BENCHMARK_ROUTE "
            "to a route this deployment serves, or widen OPENAI_ROUTES")
    print("serving: " + ", ".join(h.route for h in served), flush=True)
    print(f"benchmarking: {benchmarked[0]}. If this model does not serve it, set "
          "BENCHMARK_ROUTE to the route it does.", flush=True)
    return served


def build_config(defaults: EngineDefaults, model_server_url: str = MODEL_SERVER_URL,
                 model_server_port: int = MODEL_SERVER_PORT) -> dict:
    """The WorkerConfig kwargs for an engine. Separate from run() so the handler table
    can be inspected without starting a server."""

    # Relative path resolves against the server url+port; a full URL is used as-is.
    healthcheck_url = os.environ.get("MODEL_HEALTH_ENDPOINT", "/health")
    # Exactly one route carries a BenchmarkConfig, which is what the SDK requires.
    benchmarked = benchmark_route()

    def route(path, **kw):
        if path == benchmarked:
            b = BENCHMARKS[path]
            kw["benchmark_config"] = BenchmarkConfig(
                generator=b.generator, concurrency=b.concurrency, runs=b.runs)
        return HandlerConfig(route=path, allow_parallel_requests=True,
                             max_queue_time=600.0, **kw)

    tokens = lambda data: data.get("max_tokens", 0)   # noqa: E731
    handlers = [
        route("/v1/completions", workload_calculator=tokens, request_parser=request_parser),
        route("/v1/chat/completions", workload_calculator=tokens,
              request_parser=request_parser),
        # `input` is a real field on speech and embeddings, so they must not use the
        # shared parser, which unwraps it.
        route("/v1/audio/speech", workload_calculator=_speech_workload,
              request_parser=speech_request_parser),
        route("/v1/embeddings", workload_calculator=_embeddings_workload),
        route("/v1/images/generations", workload_calculator=_image_workload,
              request_parser=request_parser),
        # Payload classes apply their own parsing; request_parser is ignored with them.
        route("/v1/images/edits", payload_class=ImageEditPayload),
        route("/v1/images/variations", payload_class=ImageEditPayload),
        route("/v1/audio/transcriptions", payload_class=TranscriptionPayload),
        route("/v1/audio/translations", payload_class=TranscriptionPayload),
    ]
    # Routes an engine does not implement answer 404 from behind the worker.

    config = dict(
        model_server_url=model_server_url,
        model_server_port=model_server_port,
        model_log_file=os.environ.get("MODEL_LOG", defaults.model_log_file),
        model_healthcheck_url=healthcheck_url,
        handlers=_served_routes(handlers),
        log_action_config=LogActionConfig(
            on_load=_env_lines("MODEL_LOAD_LOG_MSG", defaults.load_log_msgs),
            on_error=_env_lines("MODEL_ERROR_LOG_MSGS", defaults.error_log_msgs),
            on_info=_env_lines("MODEL_INFO_LOG_MSGS", defaults.info_log_msgs),
        ),
    )
    return config


def run(defaults: EngineDefaults) -> None:
    """Build the WorkerConfig from defaults (env-overridable) and run the worker."""
    # BACKEND=openai aliases vllm, so report the real engine and note the alias.
    backend = os.environ.get("BACKEND")
    alias = f" (BACKEND={backend})" if backend and backend != defaults.name else ""
    print(f"Using worker backend: {defaults.name}{alias}", flush=True)

    Worker(WorkerConfig(**build_config(defaults))).run()
