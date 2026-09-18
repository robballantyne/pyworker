"""workers/openai/core: the handler table, and TranscriptionPayload.

Stdlib unittest so it needs no test dependency the worker does not already have:
    python -m unittest discover -s tests
"""

import base64
import json
import os
import unittest
from unittest import mock

# This repo is cloned onto instances, where these are set for real; the suite must not
# read them.
os.environ["MODEL_NAME"] = "openai/whisper-large-v3"
for _leak in ("OPENAI_ROUTES", "BENCHMARK_ROUTE", "BENCHMARK_SPEECH_VOICE",
              "VLLM_MODEL", "SGLANG_MODEL", "LLAMA_MODEL"):
    os.environ.pop(_leak, None)

from vastai.serverless.server.lib.data_types import JsonDataException  # noqa: E402

import workers.openai.core as core  # noqa: E402
from workers.openai.benchmark import BENCHMARKS  # noqa: E402
from workers.openai.core import (  # noqa: E402
    BENCHMARK_MAX_TOKENS,
    MAX_REQUEST_MULTIPLE,
    MIN_REQUEST_MULTIPLE,
    EngineDefaults,
    ImageEditPayload,
    TranscriptionPayload,
    build_config,
    request_parser,
)

WAV = b"RIFF\x24\x00\x00\x00WAVEfmt "


def b64(data=WAV):
    return base64.b64encode(data).decode()


class TestTranscriptionPayloadValidation(unittest.TestCase):
    def test_rejects_bad_file_field(self):
        for label, payload in [
            ("missing", {"model": "m"}),
            ("not a string", {"file": 123}),
            ("not base64", {"file": "!!!"}),
            ("empty", {"file": ""}),
        ]:
            with self.subTest(label), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg(payload)

    def test_rejects_non_dict_payload(self):
        with self.assertRaises(JsonDataException):
            TranscriptionPayload.from_json_msg(["not", "a", "dict"])


class TestTranscriptionPayloadParsing(unittest.TestCase):
    def test_a_tiny_clip_counts_as_the_floor(self):
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        self.assertEqual(p.count_workload(), BENCHMARK_MAX_TOKENS * MIN_REQUEST_MULTIPLE)

    def test_model_falls_back_to_the_engine_env_var(self):
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        self.assertEqual(p.fields["model"], "openai/whisper-large-v3")
        p = TranscriptionPayload.from_json_msg({"file": b64(), "model": "given"})
        self.assertEqual(p.fields["model"], "given")

    def test_input_wrapped_request_is_unwrapped(self):
        p = TranscriptionPayload.from_json_msg({"input": {"file": b64(), "model": "m"}})
        self.assertEqual(p.fields["model"], "m")


class TestTranscriptionPayloadMultipart(unittest.TestCase):
    def test_file_part_carries_filename_and_type_from_the_extension(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "filename": "clip.mp3"}
        ).generate_payload_multipart()
        self.assertEqual(fields["file"], ("clip.mp3", WAV, "audio/mpeg"))

    def test_unsupported_extension_is_rejected(self):
        for name in ("clip.unknown", "clip.html", "clip"):
            with self.subTest(name), self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": b64(), "filename": name})

    def test_null_filename_uses_the_default(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "filename": None}).generate_payload_multipart()
        self.assertEqual(fields["file"][:1], ("audio.wav",))
        self.assertEqual(fields["file"][2], "audio/wav")

    def test_filename_is_reduced_to_a_safe_basename(self):
        for given, expected in [("../../etc/cron.d/x.wav", "x.wav"),
                                ('a"\r\nX: 1.wav', "a___X__1.wav"),
                                (".hidden.wav", "hidden.wav")]:
            with self.subTest(given):
                fields = TranscriptionPayload.from_json_msg(
                    {"file": b64(), "filename": given}).generate_payload_multipart()
                self.assertEqual(fields["file"][0], expected)

    def test_other_fields_pass_through(self):
        fields = TranscriptionPayload.from_json_msg(
            {"file": b64(), "language": "en", "temperature": 0}
        ).generate_payload_multipart()
        self.assertEqual(fields["language"], "en")
        self.assertEqual(fields["temperature"], 0)

    def test_json_body_is_refused(self):
        # The backend takes the multipart branch; posting JSON would be silently wrong.
        p = TranscriptionPayload.from_json_msg({"file": b64()})
        with self.assertRaises(NotImplementedError):
            p.generate_payload_json()


DEFAULTS = EngineDefaults(
    name="stub", model_log_file="/tmp/stub.log",
    load_log_msgs=["loaded"], error_log_msgs=["failed"],
)


def handlers():
    return {h.route: h for h in build_config(DEFAULTS)["handlers"]}


class TestHandlerTable(unittest.TestCase):
    def test_input_bearing_routes_do_not_use_the_shared_request_parser(self):
        """`input` is a real field in the speech spec (and in embeddings), but the
        shared parser treats a top-level `input` as a wrapper and unwraps it -- which
        would reduce {"model","input","voice"} to the bare text and drop the rest."""
        body = {"model": "t", "input": "Hi", "voice": "a"}
        self.assertEqual(request_parser(body), "Hi")
        for route in ("/v1/audio/speech", "/v1/embeddings"):
            with self.subTest(route):
                parser = handlers()[route].request_parser
                self.assertIsNot(parser, request_parser)
                if parser is not None:
                    self.assertEqual(parser(dict(body)), body)

    def test_wrapper_routes_keep_the_shared_request_parser(self):
        for route in ("/v1/completions", "/v1/chat/completions"):
            with self.subTest(route):
                self.assertIsNotNone(handlers()[route].request_parser)

    def test_translations_shares_the_transcription_payload(self):
        h = handlers()
        self.assertIs(h["/v1/audio/translations"].payload_class,
                      h["/v1/audio/transcriptions"].payload_class)

    def test_exactly_one_route_carries_a_benchmark(self):
        """The SDK requires exactly one, which is why this worker needs no SDK change
        to pick a route."""
        carrying = [r for r, h in handlers().items() if h.benchmark_config]
        self.assertEqual(carrying, ["/v1/completions"])   # the default: LLM unchanged

    def test_benchmark_route_moves_the_benchmark(self):
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/audio/transcriptions"}):
            carrying = [r for r, h in handlers().items() if h.benchmark_config]
        self.assertEqual(carrying, ["/v1/audio/transcriptions"])

    def test_a_route_that_cannot_be_benchmarked_is_refused(self):
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/images/edits"}):
            with self.assertRaises(RuntimeError):
                handlers()

    def test_a_benchmark_route_that_is_not_served_is_refused(self):
        """Otherwise the worker boots, fails its benchmark on a route it does not
        serve, and reports an engine problem."""
        with mock.patch.dict(os.environ, {"BENCHMARK_ROUTE": "/v1/audio/speech",
                                          "OPENAI_ROUTES": "/v1/completions"}):
            with self.assertRaises(RuntimeError):
                handlers()

    def test_the_benchmark_route_is_logged_at_startup(self):
        with mock.patch("builtins.print") as printed:
            handlers()
        lines = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("benchmarking: /v1/completions", lines)
        self.assertIn("BENCHMARK_ROUTE", lines)

    def test_no_benchmark_hook_is_passed_to_the_sdk(self):
        """The released SDK has neither field; needing them would mean an SDK release
        before this worker can ship."""
        config = build_config(DEFAULTS)
        self.assertNotIn("benchmark_selector", config)
        self.assertNotIn("benchmark_route", config)


class TestImageEdit(unittest.TestCase):
    PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20

    def b64(self, data=None):
        return base64.b64encode(data or self.PNG).decode()

    def test_single_image_becomes_one_file_part(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": self.b64(), "prompt": "make it blue"}
        ).generate_payload_multipart()
        self.assertEqual(fields["image"], [("image.png", self.PNG, "image/png")])
        self.assertEqual(fields["prompt"], "make it blue")

    def test_multiple_images_become_repeated_parts(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": [self.b64(), self.b64()], "filename": ["a.png", "b.jpg"],
             "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual([f[0] for f in fields["image"]], ["a.png", "b.jpg"])
        self.assertEqual([f[2] for f in fields["image"]], ["image/png", "image/jpeg"])

    def test_mask_is_carried_when_present(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": self.b64(), "mask": self.b64(), "prompt": "p"}
        ).generate_payload_multipart()
        self.assertIn("mask", fields)

    def test_mask_has_its_own_filename(self):
        fields = ImageEditPayload.from_json_msg(
            {"image": [self.b64(), self.b64()], "filename": ["a.jpg", "b.jpg"],
             "mask": self.b64(), "mask_filename": "m.png", "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual(fields["mask"], [("m.png", self.PNG, "image/png")])

    def test_non_image_types_are_rejected(self):
        for name in ("x.svg", "x.html", "x.wav"):
            with self.subTest(name), self.assertRaises(JsonDataException):
                ImageEditPayload.from_json_msg(
                    {"image": self.b64(), "filename": name, "prompt": "p"})

    def test_model_is_filled_from_the_engine_var(self):
        payload = ImageEditPayload.from_json_msg({"image": self.b64(), "prompt": "p"})
        self.assertEqual(payload.fields["model"], "openai/whisper-large-v3")

    def test_url_is_accepted_instead_of_an_upload(self):
        fields = ImageEditPayload.from_json_msg(
            {"url": "https://x/a.png", "prompt": "p"}
        ).generate_payload_multipart()
        self.assertEqual(fields["url"], "https://x/a.png")
        self.assertNotIn("image", fields)

    def test_neither_image_nor_url_is_rejected(self):
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"prompt": "p"})

    def test_bad_base64_is_rejected(self):
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": "!!!", "prompt": "p"})

    def test_json_body_is_refused(self):
        p = ImageEditPayload.from_json_msg({"image": self.b64(), "prompt": "p"})
        with self.assertRaises(NotImplementedError):
            p.generate_payload_json()


ONE_REQUEST = float(BENCHMARK_MAX_TOKENS)
FLOOR = BENCHMARK_MAX_TOKENS * MIN_REQUEST_MULTIPLE
CEILING = BENCHMARK_MAX_TOKENS * MAX_REQUEST_MULTIPLE


class TestImageWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/images/generations"].workload_calculator

    def test_one_reference_image_is_one_request(self):
        self.assertEqual(self.calc({"size": "1024x1024", "n": 1}), ONE_REQUEST)

    def test_scales_with_n_and_pixels(self):
        self.assertEqual(self.calc({"size": "1024x1024", "n": 2}), 2 * ONE_REQUEST)
        self.assertEqual(self.calc({"size": "512x512", "n": 4}), ONE_REQUEST)

    def test_auto_missing_or_unparseable_size_is_one_reference_image(self):
        for size in (None, "auto", "nonsense"):
            with self.subTest(size):
                self.assertEqual(self.calc({"size": size}), ONE_REQUEST)

    def test_is_clamped(self):
        self.assertEqual(self.calc({"size": "64x64"}), FLOOR)
        self.assertEqual(self.calc({"size": "8192x8192", "n": 10}), CEILING)


class TestEmbeddingsWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/embeddings"].workload_calculator

    def test_all_four_input_shapes_land_on_one_scale(self):
        # 2000 characters, or 500 tokens at four characters each, is one request.
        for label, value in [
            ("string",          "x" * 2000),
            ("list of strings", ["x" * 1000, "x" * 1000]),
            ("token array",     list(range(500))),
            ("array of arrays", [list(range(250)), list(range(250))]),
        ]:
            with self.subTest(label):
                self.assertEqual(self.calc({"input": value}), ONE_REQUEST)

    def test_a_batch_costs_more_than_one_item(self):
        one = self.calc({"input": ["x" * 2000]})
        three = self.calc({"input": ["x" * 2000] * 3})
        self.assertEqual(three, 3 * one)

    def test_missing_or_null_input_is_the_floor(self):
        self.assertEqual(self.calc({}), FLOOR)
        self.assertEqual(self.calc({"input": None}), FLOOR)


class TestSpeechWorkload(unittest.TestCase):
    def setUp(self):
        self.calc = handlers()["/v1/audio/speech"].workload_calculator

    def test_reference_length_text_is_one_request(self):
        self.assertEqual(self.calc({"input": "x" * 500}), ONE_REQUEST)

    def test_short_or_missing_text_is_the_floor(self):
        self.assertEqual(self.calc({"input": "hi"}), FLOOR)
        self.assertEqual(self.calc({}), FLOOR)

    def test_each_clone_reference_adds_one_request(self):
        """Counted, not measured: a URL's length says nothing about the file it names,
        so a URL and an inline blob cost the same."""
        base = {"input": "x" * 500}
        for ref in ("https://x/a.wav", "file:///a.wav", "A" * 100_000):
            with self.subTest(ref[:20]):
                self.assertEqual(self.calc({**base, "ref_audio": ref}), 2 * ONE_REQUEST)
        self.assertEqual(self.calc({**base, "ref_audio": ["u1", "u2"]}), 3 * ONE_REQUEST)
        self.assertEqual(self.calc({**base, "ref_audio": "u1", "ref_audio_2": "u2"}),
                         3 * ONE_REQUEST)

    def test_a_clone_request_is_never_free(self):
        plain = self.calc({"input": "hello"})
        clone = self.calc({"input": "hello", "ref_audio": "https://x/a.wav"})
        self.assertGreater(clone, plain)


class TestUploadWorkload(unittest.TestCase):
    def test_a_reference_sized_clip_is_one_request(self):
        audio = base64.b64encode(b"\x00" * (1024 * 1024)).decode()
        self.assertEqual(TranscriptionPayload.from_json_msg({"file": audio}).count_workload(),
                         ONE_REQUEST)


class TestUploadLimits(unittest.TestCase):
    def test_oversized_upload_is_rejected_before_decoding(self):
        with mock.patch.object(core, "MAX_UPLOAD_BYTES", 1024), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                TranscriptionPayload.from_json_msg({"file": "A" * 10_000})

    def test_data_uri_and_line_wrapped_base64_are_accepted(self):
        wrapped = base64.encodebytes(WAV * 20).decode()      # 76-column lines
        for value in (f"data:audio/wav;base64,{b64()}", wrapped):
            with self.subTest(value[:20]):
                TranscriptionPayload.from_json_msg({"file": value})

    def test_an_oversized_data_uri_prefix_is_rejected(self):
        """The size limit is checked before the prefix is stripped: everything before
        the first comma would otherwise be unbounded."""
        huge = "data:" + "A" * (2 * core.MAX_UPLOAD_BYTES) + "," + b64()
        with self.assertRaises(JsonDataException):
            TranscriptionPayload.from_json_msg({"file": huge})

    def test_a_long_filename_keeps_its_extension(self):
        """Truncating mid-extension would reject a legitimate upload for having none."""
        name = "a" * 140 + ".wav"
        payload = TranscriptionPayload.from_json_msg({"file": b64(), "filename": name})
        part = payload.generate_payload_multipart()["file"]
        self.assertTrue(part[0].endswith(".wav"))
        self.assertEqual(part[2], "audio/wav")

    def test_too_many_images_is_rejected(self):
        img = base64.b64encode(b"\x89PNG").decode()
        with self.assertRaises(JsonDataException):
            ImageEditPayload.from_json_msg({"image": [img] * 17, "prompt": "p"})

    def test_non_object_input_is_a_422_not_a_500(self):
        for payload in ({"input": "text"}, ["a"], "x"):
            with self.subTest(str(payload)):
                with self.assertRaises(JsonDataException):
                    TranscriptionPayload.from_json_msg(payload)
                with self.assertRaises(JsonDataException):
                    ImageEditPayload.from_json_msg(payload)


class TestTranscriptionBenchmarkPayload(unittest.TestCase):
    def test_for_test_is_a_real_wav_upload(self):
        fields = TranscriptionPayload.for_test().generate_payload_multipart()
        name, audio, ctype = fields["file"]
        self.assertEqual((name, ctype), ("benchmark.wav", "audio/wav"))
        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertEqual(fields["model"], "openai/whisper-large-v3")

    def test_the_clip_is_audio_of_the_reference_size(self):
        """A shorter clip makes the benchmark weigh less than one request, so the score
        is in a different unit from every other candidate's."""
        import wave
        from io import BytesIO
        from workers.openai.benchmark import (REF_AUDIO_BYTES, WAV_RATE,
                                              WAV_BYTES_PER_SECOND, synthetic_wav)

        clip = synthetic_wav()
        self.assertAlmostEqual(len(clip), REF_AUDIO_BYTES, delta=1024)
        with wave.open(BytesIO(clip)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()),
                             (1, 2, WAV_RATE))
            self.assertEqual(
                w.getnframes(),
                int(REF_AUDIO_BYTES / WAV_BYTES_PER_SECOND * WAV_RATE))
            frames = w.readframes(w.getnframes())
        self.assertNotEqual(frames, b"\x00" * len(frames), "clip is digital silence")

    def test_every_candidate_benchmarks_one_reference_request(self):
        """The score means the same thing whichever route wins only if every candidate's
        benchmark request weighs the same."""
        weigh = {"/v1/completions": lambda b: b.get("max_tokens", 0),
                 "/v1/chat/completions": lambda b: b.get("max_tokens", 0),
                 "/v1/embeddings": core._embeddings_workload,
                 "/v1/audio/speech": core._speech_workload,
                 "/v1/images/generations": core._image_workload,
                 "/v1/audio/transcriptions":
                     lambda _b: TranscriptionPayload.for_test().count_workload()}
        for route, b in BENCHMARKS.items():
            with self.subTest(route):
                body = b.generator() if b.generator else None
                weight = weigh[route](body)
                self.assertGreaterEqual(weight, 0.5 * core.BENCHMARK_MAX_TOKENS)
                self.assertLessEqual(weight, 2 * core.BENCHMARK_MAX_TOKENS)


class TestRequestBudget(unittest.TestCase):
    def test_image_uploads_over_the_request_budget_are_refused_before_decoding(self):
        img = "A" * 4096
        with mock.patch.object(core, "MAX_REQUEST_UPLOAD_BYTES", 8000), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                ImageEditPayload.from_json_msg({"image": [img] * 3, "prompt": "p"})

    def test_inline_references_count_against_the_budget(self):
        ref = "A" * 4096
        parser = handlers()["/v1/audio/speech"].request_parser
        with mock.patch.object(core, "MAX_REQUEST_UPLOAD_BYTES", 5000), \
                mock.patch.object(core.base64, "b64decode",
                                  side_effect=AssertionError("decoded")):
            with self.assertRaises(JsonDataException):
                parser({"input": "hi", "ref_audio": [ref], "ref_audio_2": ref})

    def test_bad_upload_limit_env_falls_back(self):
        with mock.patch.dict(os.environ, {"X_LIMIT": "lots"}), \
                mock.patch("builtins.print"):
            self.assertEqual(core._env_int("X_LIMIT", 7), 7)


class TestReferences(unittest.TestCase):
    INLINE = base64.b64encode(b"RIFF" + b"\x00" * 200).decode()
    ALLOWED = ("https://x/a.png", "http://x/a.png", f"data:audio/wav;base64,{INLINE}", INLINE)
    REFUSED = ("file:///root/.ssh/id_rsa", "file:/etc/passwd", "ftp://x/a", "gopher://x",
               # bare paths an engine might open; the second is valid base64 but tiny
               "/etc/passwd", "/tmp/abcd/efgh", "../../workspace/x.wav",
               "\\\\host\\share\\a.wav", "AAAA")

    def test_image_edit_urls(self):
        for key in ("url", "url[]"):
            for value in self.ALLOWED[:2]:
                with self.subTest((key, value)):
                    ImageEditPayload.from_json_msg({key: [value], "prompt": "p"})
            for value in self.REFUSED:
                with self.subTest((key, value)), self.assertRaises(JsonDataException):
                    ImageEditPayload.from_json_msg({key: value, "prompt": "p"})

    def test_speech_references(self):
        parser = handlers()["/v1/audio/speech"].request_parser
        for key in ("ref_audio", "ref_audio_2"):
            for value in self.ALLOWED:
                with self.subTest((key, value)):
                    parser({"input": "hi", key: value})
            for value in self.REFUSED:
                with self.subTest((key, value)), self.assertRaises(JsonDataException):
                    parser({"input": "hi", key: value})


class TestServedRoutes(unittest.TestCase):
    ALL = {"/v1/completions", "/v1/chat/completions", "/v1/audio/speech",
           "/v1/embeddings", "/v1/images/generations", "/v1/images/edits",
           "/v1/images/variations", "/v1/audio/transcriptions", "/v1/audio/translations"}

    def test_all_routes_by_default(self):
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": ""}):
            self.assertEqual(set(handlers()), self.ALL)

    def test_openai_routes_narrows(self):
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": "/v1/audio/speech",
                                          "BENCHMARK_ROUTE": "/v1/audio/speech"}):
            self.assertEqual(set(handlers()), {"/v1/audio/speech"})

    def test_a_route_list_without_the_benchmark_route_is_refused(self):
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": "/v1/audio/translations"}):
            with self.assertRaises(RuntimeError):
                handlers()

    def test_unknown_route_names_are_reported(self):
        with mock.patch.dict(os.environ,
                             {"OPENAI_ROUTES": "/v1/audio/speech,/v1/embedding",
                              "BENCHMARK_ROUTE": "/v1/audio/speech"}), \
                mock.patch("builtins.print") as printed:
            handlers()
        lines = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("unknown routes: /v1/embedding", lines)
        self.assertIn("serving: /v1/audio/speech", lines)

    def test_upload_routes_are_dropped_on_an_sdk_without_multipart(self):
        class OldApiPayload:
            pass

        with mock.patch.object(core, "ApiPayload", OldApiPayload):
            served = handlers()
        self.assertEqual(set(served), self.ALL - set(core.UPLOAD_ROUTES))
        # derived from the table, so dropping a route from UPLOAD_ROUTES fails here
        # instead of leaving it served and 500ing on every request
        multipart_only = {r for r, h in handlers().items()
                          if h.payload_class is not None
                          and issubclass(h.payload_class, core._UploadPayload)}
        self.assertEqual(multipart_only, set(core.UPLOAD_ROUTES))
        self.assertFalse(any(h.payload_class is not None
                             and issubclass(h.payload_class, core._UploadPayload)
                             for h in served.values()))


REALISTIC_MAX_THROUGHPUT = 200.0   # tokens/s: 10 x 500-token completions in ~25s


def wait_time_with_one_in_flight(workload):
    """The SDK's own wait_time for a worker holding exactly one request."""
    from vastai.serverless.server.lib.data_types import ModelMetrics, RequestMetrics

    metrics = ModelMetrics.empty()
    metrics.max_throughput = REALISTIC_MAX_THROUGHPUT
    metrics.requests_working[1] = RequestMetrics(
        request_idx=1, reqnum=1, workload=workload, status="Started")
    return metrics.wait_time


class TestAdmissionGate(unittest.TestCase):
    """wait_time sums in-flight workload across EVERY route and divides by a
    max_throughput measured on /v1/completions only; the SDK 429s any request that
    arrives while it exceeds max_queue_time. So a single legal request on any route
    must not be able to push it over on its own, or one upload 429s the whole worker."""

    WORST_CASES = {
        "/v1/audio/speech": {"input": "x" * 100_000,
                             "ref_audio": ["A" * 10_000] * 10,
                             "ref_audio_2": "https://x/a.wav"},
        "/v1/embeddings": {"input": ["x" * 10_000] * 1_000},
        "/v1/images/generations": {"n": 100_000, "size": "8192x8192"},
    }

    def test_no_single_request_trips_the_gate(self):
        h = handlers()
        for route, data in self.WORST_CASES.items():
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(h[route].workload_calculator(data))
                self.assertLess(wt, h[route].max_queue_time)

    def test_a_large_upload_does_not_trip_the_gate(self):
        h = handlers()
        big = base64.b64encode(b"\x00" * (25 * 1024 * 1024)).decode()   # OpenAI's file limit
        for route, payload in [
            ("/v1/audio/transcriptions", TranscriptionPayload.from_json_msg({"file": big})),
            ("/v1/images/edits", ImageEditPayload.from_json_msg(
                {"image": [big] * 2, "prompt": "p", "n": 100_000, "size": "8192x8192"})),
        ]:
            with self.subTest(route):
                wt = wait_time_with_one_in_flight(payload.count_workload())
                self.assertLess(wt, h[route].max_queue_time)

    def test_absurd_declared_sizes_do_not_raise(self):
        calc = handlers()["/v1/images/generations"].workload_calculator
        for data in ({"size": "9" * 4000 + "x" + "9" * 4000},
                     {"n": "9" * 5000, "size": "1024x1024"},
                     {"n": -5, "size": "-10x-10"}):
            with self.subTest(str(data)[:40]):
                self.assertGreaterEqual(calc(data), FLOOR)

    def test_malformed_speech_input_does_not_raise(self):
        calc = handlers()["/v1/audio/speech"].workload_calculator
        for data in ({"input": 123}, {"input": "hi", "ref_audio": 5},
                     {"input": ["a", "b"]}, {"ref_audio": [None, 7, ""]}):
            with self.subTest(str(data)):
                self.assertGreater(calc(data), 0)

    def test_completions_units_are_unchanged(self):
        # Existing scores must not move: the benchmarked route keeps max_tokens.
        calc = handlers()["/v1/completions"].workload_calculator
        self.assertEqual(calc({"max_tokens": 500}), 500)


if __name__ == "__main__":
    unittest.main()
