#!/usr/bin/env python3
"""Run a real PyWorker against a stub engine, locally, with no GPU and no platform.

Starts three things in one process:

  stub engine   an OpenAI-compatible server that answers the routes the worker proxies
  report sink   swallows the autoscaler reports the worker emits, and shows the last one
  the worker    the real Worker/Backend from the SDK, UNSECURED so no signing is needed

then sends envelope-shaped requests at it with the SDK client's own request function
(`vastai.serverless.client.connection._make_request`), so responses are consumed the way
a real caller consumes them:

    {"auth_data": {...}, "session_id": null, "payload": {...}}

Run:  python tests/harness.py            (add -v to see worker logs)

UNSECURED skips signature checking, so everything binds 127.0.0.1 and the harness
refuses to run on a Vast instance. Its working directory is a temp dir, which is where
the SDK writes .has_benchmark.

Not covered here: the admission gate (wait_time vs max_queue_time). The stub answers
instantly, so its benchmark score is far above a GPU's and the gate cannot trip; the
unit tests check it against a realistic score instead.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
import shutil
import tempfile
from pathlib import Path

WORKER_PORT = 13000
ENGINE_PORT = 13010
REPORT_PORT = 13020
LOAD_LINE = "harness: model loaded"



def configure_env(workdir: Path, worker_port: int, bench_route=None) -> Path:
    """The environment a worker expects on an instance, pointed at local stubs."""
    global WORKER_PORT
    WORKER_PORT = worker_port
    model_log = workdir / "model.log"
    os.environ.update(
        WORKER_PORT=str(worker_port),
        WORKER_HTTP_PORT=str(worker_port + 1),
        **{f"VAST_TCP_PORT_{worker_port}": str(worker_port)},
        PUBLIC_IPADDR="127.0.0.1",
        CONTAINER_ID="1",
        REPORT_ADDR=f"http://127.0.0.1:{REPORT_PORT}",
        UNSECURED="true",
        MODEL_NAME="stub/model",
        MODEL_LOG=str(model_log),
        MODEL_LOAD_LOG_MSG=LOAD_LINE,
        MODEL_HEALTH_ENDPOINT="/health",
    )
    # What the template sets for a non-LLM deployment.
    os.environ.pop("BENCHMARK_ROUTE", None)
    if bench_route:
        os.environ["BENCHMARK_ROUTE"] = bench_route
    return model_log

from aiohttp import ClientSession, web  # noqa: E402

reports = []
engine_hits = {}


# ── stub engine ────────────────────────────────────────────────────────────
async def engine_health(_req):
    return web.json_response({"status": "ok"})


async def engine_completions(req):
    body = await req.json()
    n = body.get("max_tokens", 16)
    return web.json_response({
        "id": "cmpl-stub", "object": "text_completion", "model": body.get("model"),
        "choices": [{"text": " stub" * min(n, 8), "index": 0, "finish_reason": "length"}],
        "usage": {"prompt_tokens": 250, "completion_tokens": n, "total_tokens": 250 + n},
    })


async def engine_chat(req):
    body = await req.json()
    return web.json_response({
        "id": "chatcmpl-stub", "object": "chat.completion", "model": body.get("model"),
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "stub reply"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    })


async def engine_transcriptions(req):
    """Multipart in, JSON out -- the OpenAI transcription shape."""
    if req.content_type != "multipart/form-data":
        return web.json_response({"error": f"expected multipart, got {req.content_type}"},
                                 status=415)
    seen, form = {}, {}
    reader = await req.multipart()
    while (part := await reader.next()) is not None:
        raw = await part.read(decode=False)
        seen[part.name] = (part.filename, len(raw), part.headers.get("Content-Type"))
        if part.filename is None:
            form[part.name] = raw.decode(errors="replace")
    fn, size, ctype = seen.get("file", (None, 0, None))
    text = f"transcribed {size} bytes of {ctype} from {fn!r}"
    if form.get("response_format") in ("text", "srt", "vtt"):
        return web.Response(text=text, content_type="text/plain")
    return web.json_response({"text": text})


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20


async def engine_images_generations(req):
    body = await req.json()
    n = int(body.get("n", 1))
    return web.json_response({
        "created": 0, "size": body.get("size"),
        "data": [{"b64_json": base64.b64encode(PNG).decode()} for _ in range(n)],
    })


async def engine_images_edits(req):
    """Multipart with possibly repeated `image` parts, mirroring vllm-omni's signature."""
    if req.content_type != "multipart/form-data":
        return web.json_response({"error": f"expected multipart, got {req.content_type}"},
                                 status=415)
    images, others = [], {}
    reader = await req.multipart()
    while (part := await reader.next()) is not None:
        raw = await part.read(decode=False)
        if part.name == "image":
            images.append((part.filename, len(raw), part.headers.get("Content-Type")))
        elif part.name != "mask":
            others[part.name] = raw.decode(errors="replace")
    return web.json_response({
        "created": 0, "images_received": len(images), "encoding": req.content_type,
        "url": others.get("url"),
        "filenames": [i[0] for i in images], "types": [i[2] for i in images],
        "prompt": others.get("prompt"),
        "data": [{"b64_json": base64.b64encode(PNG).decode()}],
    })


async def engine_embeddings(req):
    """JSON in, JSON out. Echoes the batch size so the harness can check it arrived."""
    body = await req.json()
    value = body.get("input")
    items = value if isinstance(value, list) else [value]
    return web.json_response({
        "object": "list", "model": body.get("model"),
        "data": [{"object": "embedding", "index": i, "embedding": [0.0, 1.0, 0.0]}
                 for i, _ in enumerate(items)],
        "usage": {"prompt_tokens": len(items), "total_tokens": len(items)},
    })


async def engine_translations(req):
    """Same multipart shape as transcriptions; output is always English."""
    if req.content_type != "multipart/form-data":
        return web.json_response({"error": f"expected multipart, got {req.content_type}"},
                                 status=415)
    reader = await req.multipart()
    size = 0
    while (part := await reader.next()) is not None:
        if part.name == "file":
            size = len(await part.read(decode=False))
        else:
            await part.read()
    return web.json_response({"text": f"translated {size} bytes to English"})


async def engine_speech(req):
    """JSON in, BINARY out -- the OpenAI speech shape."""
    body = await req.json()
    text = body.get("input") or ""
    fmt = body.get("response_format", "mp3")
    ctype = {"mp3": "audio/mpeg", "wav": "audio/wav", "opus": "audio/opus",
             "flac": "audio/flac", "pcm": "audio/L16"}.get(fmt, "application/octet-stream")
    # A byte per character, so the harness can prove the whole body survived.
    ref = body.get("ref_audio") or ""
    return web.Response(body=b"\xff" * len(text), content_type=ctype,
                        headers={"X-Stub-Voice": body.get("voice", ""),
                                 "X-Stub-Ref-Len": str(len(ref)),
                                 "X-Stub-Task": body.get("task_type", "")})


ENGINE_ROUTES = {
    "/v1/completions": engine_completions,
    "/v1/chat/completions": engine_chat,
    "/v1/audio/transcriptions": engine_transcriptions,
    "/v1/audio/speech": engine_speech,
    "/v1/audio/translations": engine_translations,
    "/v1/embeddings": engine_embeddings,
    "/v1/images/generations": engine_images_generations,
    "/v1/images/edits": engine_images_edits,
}


def make_engine_app(served=None):
    """An engine serving `served` routes (all when None); the rest answer 404, as a
    real engine does for a task its model cannot do."""
    def route(path, handler):
        async def dispatch(req):
            engine_hits[path] = engine_hits.get(path, 0) + 1
            if served is not None and path not in served:
                await req.read()
                return web.json_response({"error": "not found"}, status=404)
            return await handler(req)
        return dispatch

    # 0 is unlimited: uploads and the reference-sized benchmark probe exceed aiohttp's
    # 1 MiB default
    app = web.Application(client_max_size=0)
    app.router.add_get("/health", engine_health)
    for path, handler in ENGINE_ROUTES.items():
        app.router.add_post(path, route(path, handler))
    return app


# ── report sink ────────────────────────────────────────────────────────────
async def sink(req):
    raw = await req.text()
    try:
        body = json.loads(raw)
    except ValueError:
        body = {"unparsed": raw[:300]}
    reports.append({"path": req.path, "body": body})
    return web.json_response({"ok": True})


def make_sink_app():
    app = web.Application()
    app.router.add_route("*", "/{tail:.*}", sink)
    return app


# ── client: the envelope the SDK client sends ──────────────────────────────
def envelope(payload, cost=100):
    return {
        "auth_data": {"cost": str(cost), "endpoint": "stub", "reqnum": 1,
                      "request_idx": 1, "signature": "unsigned", "url": "local"},
        "session_id": None,
        "payload": payload,
    }


class _SdkClient:
    """The two things _make_request needs from a client."""

    def __init__(self, session):
        self._session = session

    async def _get_session(self):
        return self._session

    async def get_ssl_context(self):
        return None


async def call(session, route, payload, want_headers=False):
    from vastai.serverless.client.connection import _make_request

    try:
        result = await _make_request(
            client=_SdkClient(session), route=route, api_key="harness",
            url=f"http://127.0.0.1:{WORKER_PORT}", body=envelope(payload),
            method="POST", retries=1, timeout=60,
        )
    except Exception as exc:
        # The client failing to read a response is a failed check, not a crash.
        err = f"client raised {type(exc).__name__}: {exc}"[:120]
        return (None, "", err, {}) if want_headers else (None, "", err)
    headers = result.get("headers") or {}
    ctype = headers.get("Content-Type", "")
    body = next((result[k] for k in ("content", "json") if result.get(k) is not None),
                result.get("text"))
    if want_headers:
        return result["status"], ctype, body, headers
    return result["status"], ctype, body


async def serve(app, port):
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def wait_for_worker(session, attempts=120):
    for _ in range(attempts):
        try:
            async with session.get(f"http://127.0.0.1:{WORKER_PORT}/") as _r:
                return True     # any HTTP answer means the port is bound
        except Exception:
            await asyncio.sleep(0.5)
    return False


WAV = b"RIFF\x24\x00\x00\x00WAVEfmt " + b"\x00" * 40


SCENARIOS = [
    # (title, routes the engine serves (None = all), worker port, checks, benchmark route)
    ("engine serving every route", None, 13000, "full", None),
    # Whisper on vLLM: no generate routes at all, so /v1/completions 404s. The template
    # says so, and the worker benchmarks transcriptions instead.
    ("engine serving only transcription", {"/v1/audio/transcriptions",
                                           "/v1/audio/translations"}, 13100, "asr",
     "/v1/audio/transcriptions"),
    # A deployment exposing only translations, and an edit-only image model. Each had no
    # route it could both serve and benchmark, so neither could ever become ready.
    ("engine serving only translation", {"/v1/audio/translations"}, 13200, "only",
     "/v1/audio/translations"),
    ("engine serving only image edits", {"/v1/images/edits"}, 13300, "only",
     "/v1/images/edits"),
]


async def main(verbose, force):
    # UNSECURED turns off signature checks, and CONTAINER_ID is set on every Vast
    # instance (this repo is cloned onto them), so refuse to run there by default.
    if os.environ.get("CONTAINER_ID") and not force:
        print("refusing to run: CONTAINER_ID is set, so this looks like a Vast instance "
              "(pass --force to override)")
        return 2
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        # The SDK names its loggers getLogger(__file__) and imports after this runs,
        # so per-name levels cannot reach them. Disable globally instead.
        logging.disable(logging.CRITICAL)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    cwd = os.getcwd()
    results = []
    for title, served, port, checks, bench_route in SCENARIOS:
        print(f"\n══ {title} " + "═" * max(0, 60 - len(title)))
        workdir = Path(tempfile.mkdtemp(prefix="pyworker-harness-"))
        model_log = configure_env(workdir, port, bench_route)
        model_log.write_text("")   # the load line is appended after the worker starts
        # The SDK writes .has_benchmark relative to cwd and skips benchmarking when it
        # exists, so a fresh directory keeps it out of the repo and forces a real run.
        os.chdir(workdir)
        try:
            results.append(await run_harness(model_log, workdir, served, checks))
        finally:
            os.chdir(cwd)
            shutil.rmtree(workdir, ignore_errors=True)
    ok = all(r == 0 for r in results)
    print("\nOVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


async def run_harness(model_log, workdir, served, checks):
    reports.clear()
    engine_hits.clear()
    engine = await serve(make_engine_app(served), ENGINE_PORT)
    reporter = await serve(make_sink_app(), REPORT_PORT)
    try:
        return await run_worker(model_log, workdir, checks)
    finally:
        await engine.cleanup()
        await reporter.cleanup()


async def run_worker(model_log, workdir, checks):
    from vastai import Worker, WorkerConfig, LogActionConfig
    from workers.openai.core import EngineDefaults, build_config

    # The SHIPPED handler table, not a copy of it. A harness that redeclares the
    # handlers tests a parallel definition and silently stops covering the real one.
    cfg_kwargs = build_config(
        EngineDefaults(name="stub", model_log_file=str(model_log),
                       load_log_msgs=[LOAD_LINE], error_log_msgs=["harness: fatal"]),
        model_server_url="http://127.0.0.1", model_server_port=ENGINE_PORT,
    )
    cfg_kwargs.update(
        model_log_file=str(model_log),
        model_healthcheck_url="/health",
        log_action_config=LogActionConfig(on_load=[LOAD_LINE],
                                          on_error=["harness: fatal"]),
    )

    worker = Worker(WorkerConfig(**cfg_kwargs))
    task = asyncio.create_task(worker.run_async(host="127.0.0.1"))
    try:
        await asyncio.sleep(1.0)
        # Appended, not pre-written: the backend tails this file, so the marker has to
        # arrive while it is following. This is what triggers the benchmark.
        with model_log.open("a") as fh:
            fh.write(LOAD_LINE + "\n")
        print(f"worker       : http://127.0.0.1:{WORKER_PORT}\n")
        if checks == "asr":
            return await asr_only_checks(workdir)
        if checks == "only":
            return await single_route_checks(workdir, os.environ["BENCHMARK_ROUTE"])
        return await requests(workdir)
    finally:
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass
        # Cancelling run_async does not close the Backend's engine session.
        sess = getattr(getattr(worker, "backend", None), "session", None)
        if sess is not None and not sess.closed:
            await sess.close()


async def wait_for_benchmark(timeout=60):
    """The benchmark runs off the model-loaded marker and is reported asynchronously.
    Poll for a scored report; the first reports carry max_perf 0.0 legitimately."""
    for _ in range(int(timeout / 0.5)):
        if any(r["body"].get("max_perf") for r in reports if isinstance(r.get("body"), dict)):
            return
        await asyncio.sleep(0.5)


def benchmark_summary(workdir, checks):
    bodies = [r["body"] for r in reports if isinstance(r.get("body"), dict)]
    perfs = [b["max_perf"] for b in bodies if "max_perf" in b]
    errors = [b["error_msg"] for b in bodies if b.get("error_msg")]
    print(f"\nautoscaler reports received: {len(reports)}")
    # max_perf is the benchmark score -- the number the autoscaler divides in-flight
    # workload by to estimate wait_time.
    print("  engine calls per route: "
          + ", ".join(f"{r} {n}" for r, n in sorted(engine_hits.items())))
    print(f"  max_perf across reports: {perfs if perfs else 'none carried it'}")
    if errors:
        print(f"  worker errors reported: {errors[-1]}")
    score_file = workdir / ".has_benchmark"
    written = score_file.read_text().strip() if score_file.exists() else "not written"
    print(f"  score written to .has_benchmark: {written}")
    scored = 0.0
    try:
        scored = float(written)
    except ValueError:
        pass
    # A zero score is worse than none: wait_time divides by max(max_throughput, 1e-5),
    # so every request would 429.
    checks.append(("the benchmark ran and wrote a non-zero score", scored > 0))
    checks.append(("the autoscaler was told a non-zero max_perf",
                   any(p > 0 for p in perfs)))


def report(checks):
    print("\n── results ──────────────────────────────────────────────")
    ok = True
    for name, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
        ok &= passed
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


async def asr_only_checks(workdir):
    checks = []
    async with ClientSession() as s:
        if not await wait_for_worker(s):
            print("worker never became reachable")
            return 1
        await wait_for_benchmark()
        st, ct, body = await call(s, "/v1/audio/transcriptions",
                                  {"file": base64.b64encode(WAV).decode(),
                                   "filename": "clip.wav"})
        print(f"/v1/audio/transcriptions {st}  {str(body)[:60]}")
        checks.append(("transcription works on a transcription-only engine", st == 200))
        st, ct, body = await call(s, "/v1/completions", {"prompt": "hi", "max_tokens": 1})
        print(f"/v1/completions          {st}  (the engine does not serve it)")
        checks.append(("completions answers the engine's 404", st == 404))

    benchmark_summary(workdir, checks)
    # Only the request the harness itself made: the benchmark never touches completions.
    checks.append(("completions was never benchmarked",
                   engine_hits.get("/v1/completions", 0) <= 1))
    checks.append(("the benchmark ran on transcriptions",
                   engine_hits.get("/v1/audio/transcriptions", 0) >= 5))
    return report(checks)


async def single_route_checks(workdir, route):
    """An engine serving exactly one route: the worker must benchmark on it, reach a
    score, and never probe a route the engine does not serve."""
    checks = []
    async with ClientSession() as s:
        if not await wait_for_worker(s):
            print("worker never became reachable")
            return 1
        await wait_for_benchmark()
    benchmark_summary(workdir, checks)
    others = {p: n for p, n in engine_hits.items() if p != route and n}
    checks.append((f"the benchmark ran on {route}", engine_hits.get(route, 0) >= 1))
    checks.append(("no other route was ever called", not others))
    if others:
        print("  unexpected engine calls:", others)
    return report(checks)


async def requests(workdir):
    from workers.openai.core import _speech_workload


    async with ClientSession() as s:
        if not await wait_for_worker(s):
            print("worker never became reachable")
            return 1

        print("── requests ─────────────────────────────────────────────")
        checks = []

        st, ct, body = await call(s, "/v1/completions",
                                  {"model": "stub/model", "prompt": "hi", "max_tokens": 4})
        print(f"/v1/completions          {st}  {str(body)[:70]}")
        checks.append(("completions", st == 200))

        st, ct, body = await call(s, "/v1/chat/completions",
                                  {"model": "stub/model",
                                   "messages": [{"role": "user", "content": "hi"}],
                                   "max_tokens": 4})
        print(f"/v1/chat/completions     {st}  {str(body)[:70]}")
        checks.append(("chat", st == 200))

        st, ct, body = await call(s, "/v1/audio/transcriptions",
                                  {"file": base64.b64encode(WAV).decode(),
                                   "filename": "clip.wav", "model": "stub/model"})
        print(f"/v1/audio/transcriptions {st}  {str(body)[:70]}")
        checks.append(("transcriptions", st == 200 and "transcribed" in str(body)))
        checks.append(("audio reached the engine as multipart", "audio/wav" in str(body)))
        checks.append(("byte count survived the round trip", str(len(WAV)) in str(body)))

        SPEECH_TEXT = "Hello from the harness."
        st, ct, body = await call(s, "/v1/audio/speech",
                                  {"model": "stub/model", "input": SPEECH_TEXT,
                                   "voice": "alloy", "response_format": "mp3"})
        print(f"/v1/audio/speech         {st}  {ct}  {len(body) if isinstance(body, bytes) else body} bytes")
        checks.append(("speech returns audio bytes, not JSON",
                       st == 200 and isinstance(body, bytes)))
        checks.append(("audio content type preserved", "audio/mpeg" in ct))
        checks.append(("full body survived the worker",
                       isinstance(body, bytes) and len(body) == len(SPEECH_TEXT)))

        # Inline references must be media-sized (64 bytes at least) to be accepted.
        REF = base64.b64encode(WAV + b"\x00" * 512).decode()
        st, ct, body, hdrs = await call(s, "/v1/audio/speech",
                                        {"model": "stub/model", "input": "Cloned.",
                                         "task_type": "Base", "ref_audio": REF,
                                         "ref_text": "reference transcript"},
                                        want_headers=True)
        print(f"/v1/audio/speech clone   {st}  ref_len={hdrs.get('X-Stub-Ref-Len')} "
              f"task={hdrs.get('X-Stub-Task')}")
        checks.append(("voice-clone reference reaches the engine intact",
                       hdrs.get("X-Stub-Ref-Len") == str(len(REF))))
        checks.append(("clone fields pass through untouched",
                       hdrs.get("X-Stub-Task") == "Base"))
        checks.append(("a clone costs more workload than plain speech",
                       _speech_workload({"input": "Cloned.", "ref_audio": REF})
                       > _speech_workload({"input": "Cloned."})))

        st, ct, body = await call(s, "/v1/audio/translations",
                                  {"file": base64.b64encode(WAV).decode(),
                                   "filename": "clip.wav", "model": "stub/model"})
        print(f"/v1/audio/translations   {st}  {str(body)[:60]}")
        checks.append(("translations round-trip as multipart",
                       st == 200 and "translated" in str(body)))

        st, ct, body = await call(s, "/v1/embeddings",
                                  {"model": "stub/model",
                                   "input": ["first text", "second text"]})
        n = len(body.get("data", [])) if isinstance(body, dict) else 0
        print(f"/v1/embeddings           {st}  {n} vectors")
        checks.append(("embeddings batch survives as a list", st == 200 and n == 2))
        # The trap this route shares with speech: the shared parser would unwrap
        # `input` and the engine would receive a bare list instead of the request.
        checks.append(("embeddings `input` was not unwrapped",
                       isinstance(body, dict) and body.get("model") == "stub/model"))

        st, ct, body = await call(s, "/v1/images/generations",
                                  {"model": "stub/model", "prompt": "a cat",
                                   "size": "512x512", "n": 2})
        n = len(body.get("data", [])) if isinstance(body, dict) else 0
        print(f"/v1/images/generations   {st}  {n} images")
        checks.append(("image generation returns n images", st == 200 and n == 2))

        st, ct, body = await call(s, "/v1/images/edits",
                                  {"image": [base64.b64encode(PNG).decode()] * 2,
                                   "filename": ["a.png", "b.jpg"],
                                   "prompt": "make it blue", "model": "stub/model"})
        got = body if isinstance(body, dict) else {}
        print(f"/v1/images/edits         {st}  {got.get('images_received')} parts "
              f"{got.get('filenames')} {got.get('types')}")
        checks.append(("both images arrive as repeated multipart parts",
                       st == 200 and got.get("images_received") == 2))
        checks.append(("per-file names and types preserved",
                       got.get("filenames") == ["a.png", "b.jpg"]
                       and got.get("types") == ["image/png", "image/jpeg"]))
        checks.append(("prompt survives alongside the files",
                       got.get("prompt") == "make it blue"))

        st, ct, body = await call(s, "/v1/audio/transcriptions",
                                  {"file": base64.b64encode(WAV).decode(),
                                   "response_format": "text"})
        print(f"  text transcript        {st}  {ct}  {str(body)[:40]}")
        checks.append(("a text/plain transcript is returned, not retried",
                       st == 200 and isinstance(body, str) and body.startswith("transcribed")))

        st, ct, body = await call(s, "/v1/images/edits",
                                  {"url": "https://example.com/a.png", "prompt": "p"})
        got = body if isinstance(body, dict) else {}
        print(f"  url-only image edit    {st}  {got.get('encoding')}")
        checks.append(("a url-only image edit is still sent multipart",
                       st == 200 and got.get("encoding") == "multipart/form-data"
                       and got.get("url") == "https://example.com/a.png"))

        st, ct, body = await call(s, "/v1/audio/transcriptions", {"file": "!!!"})
        print(f"  bad base64 rejected    {st}")
        checks.append(("bad base64 rejected", st == 422))

        st, ct, body = await call(s, "/v1/audio/speech",
                                  {"input": "hi", "ref_audio": "file:///etc/passwd"})
        print(f"  file:// ref rejected   {st}")
        checks.append(("a file:// reference is refused before the engine", st == 422))

        st, ct, body = await call(s, "/v1/images/edits",
                                  {"image": base64.b64encode(PNG).decode(),
                                   "filename": "x.svg", "prompt": "p"})
        print(f"  svg upload rejected    {st}")
        checks.append(("an unsupported upload type is refused", st == 422))

        await wait_for_benchmark()

    benchmark_summary(workdir, checks)
    checks.append(("an LLM engine is still benchmarked on completions",
                   engine_hits.get("/v1/completions", 0) >= 10))
    return report(checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="show worker logs")
    ap.add_argument("--force", action="store_true",
                    help="run even where CONTAINER_ID is set")
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.verbose, args.force)))
