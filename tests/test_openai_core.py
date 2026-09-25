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


def _flac_like(seconds, rate=16000):
    """A FLAC whose STREAMINFO declares `seconds` of audio, with almost no payload:
    the point of these fixtures is that duration comes from the header, not the size."""
    samples = int(seconds * rate)
    info = bytearray(34)
    packed = (rate << 44) | (0 << 41) | ((16 - 1) << 36) | samples
    info[10:18] = packed.to_bytes(8, "big")
    return b"fLaC" + bytes([0x00, 0, 0, 34]) + bytes(info) + b"\x00" * 256


def _mp4_like(seconds, timescale=600):
    """An mvhd atom declaring duration/timescale, as m4a and mp4 carry it."""
    body = (bytes([0]) + b"\x00" * 3 + b"\x00" * 8
            + timescale.to_bytes(4, "big") + int(seconds * timescale).to_bytes(4, "big"))
    return b"\x00" * 8 + b"moov" + b"\x00" * 4 + b"mvhd" + body + b"\x00" * 64


def _ogg_like(seconds, rate=48000):
    """Opus counts its granule positions in 48 kHz units whatever the input rate."""
    head = b"OggS" + bytes([0, 2]) + (0).to_bytes(8, "little") + b"\x00" * 200
    head += b"OpusHead" + b"\x00" * 16
    last = b"OggS" + bytes([0, 4]) + int(seconds * rate).to_bytes(8, "little")
    return head + last + b"\x00" * 16


def _mp3_like(seconds, rate=16000, frames=None):
    """MPEG-2 Layer III at 16 kHz, 576 samples per frame, with the Xing/Info frame
    count a VBR encoder writes. ffmpeg produces exactly this shape by default."""
    frames = frames if frames is not None else int(seconds * rate / 576)
    header = bytes([0xFF, 0xF3, 0x58, 0xC0])
    info = b"Info" + (1).to_bytes(4, "big") + frames.to_bytes(4, "big")
    return header + b"\x00" * 20 + info + b"\x00" * 4096


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
        """One reference request of text, however the caller shapes it: a string, a
        batch, pre-tokenised ids, or a batch of those."""
        from workers.openai.benchmark import REF_EMBED_CHARS
        chars, tokens = REF_EMBED_CHARS, REF_EMBED_CHARS // core.CHARS_PER_TOKEN
        for label, value in [
            ("string",          "x" * chars),
            ("list of strings", ["x" * (chars // 2), "x" * (chars // 2)]),
            ("token array",     list(range(tokens))),
            ("array of arrays", [list(range(tokens // 2)), list(range(tokens // 2))]),
        ]:
            with self.subTest(label):
                self.assertEqual(self.calc({"input": value}), ONE_REQUEST)

    def test_a_batch_costs_more_than_one_item(self):
        from workers.openai.benchmark import REF_EMBED_CHARS
        one = self.calc({"input": ["x" * REF_EMBED_CHARS]})
        three = self.calc({"input": ["x" * REF_EMBED_CHARS] * 3})
        self.assertEqual(three, 3 * one)

    def test_the_benchmark_payload_fits_a_short_context_encoder(self):
        """THE defect this sizing exists for, measured on BAAI/bge-small-en-v1.5: at
        2000 chars the payload tokenised to 513 against a 512 limit, vLLM refused it,
        and the worker never became ready. Tokens per character vary with the draw
        (500 chars measured 133-160 tokens over five draws), so the reference is sized
        for all-MiniLM-L6's 256 rather than the 512 of the bge/e5/gte family."""
        from workers.openai.benchmark import REF_EMBED_CHARS, embeddings_benchmark_generator
        worst_case_tokens = len(embeddings_benchmark_generator()["input"]) / 3.1
        self.assertLess(worst_case_tokens, 256 * 0.8,
                        f"{REF_EMBED_CHARS} chars can tokenise past a 256-token encoder")

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
    """ASR is counted in SECONDS of audio, not bytes: the same megabyte is 26s of
    320 kbps MP3 or 349s of 24 kbps Opus, and charging both the same under-counted the
    slow end -- the direction that makes the queue estimate optimistic. Duration is also
    knowable before the request runs, which output tokens are not, and it is what the
    engine itself bills (vLLM answers with usage.type == "duration")."""

    def test_a_reference_length_clip_is_one_request(self):
        from workers.openai.benchmark import REF_AUDIO_SECONDS, synthetic_wav

        audio = base64.b64encode(synthetic_wav(REF_AUDIO_SECONDS)).decode()
        payload = TranscriptionPayload.from_json_msg({"file": audio,
                                                      "filename": "a.wav"})
        self.assertAlmostEqual(payload.count_workload(), ONE_REQUEST, delta=1)

    def test_the_same_audio_costs_the_same_whatever_the_container(self):
        """The defect this replaced: an encoding choice changed the price by 13x."""
        from workers.openai.benchmark import synthetic_wav

        wav = synthetic_wav(10.0)
        loose = base64.b64encode(wav).decode()                     # 320 KB of PCM
        # the same ten seconds as a compact container, an eighth of the bytes
        tight = base64.b64encode(_flac_like(10.0)).decode()
        a = TranscriptionPayload.from_json_msg({"file": loose, "filename": "a.wav"})
        b = TranscriptionPayload.from_json_msg({"file": tight, "filename": "a.flac"})
        self.assertAlmostEqual(a.count_workload(), b.count_workload(), delta=1)
        self.assertLess(len(_flac_like(10.0)), len(wav) // 4)

    def test_a_clip_with_no_parsable_header_falls_back_to_bytes(self):
        """The fallback must still produce a number, since refusing to price a request
        is worse than pricing it approximately."""
        audio = base64.b64encode(b"\x00" * (320 * 1000)).decode()
        payload = TranscriptionPayload.from_json_msg({"file": audio,
                                                      "filename": "a.webm"})
        self.assertGreater(payload.count_workload(), 0)


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
        from workers.openai.benchmark import REF_AUDIO_SECONDS, WAV_RATE, synthetic_wav

        clip = synthetic_wav()
        with wave.open(BytesIO(clip)) as w:
            self.assertEqual((w.getnchannels(), w.getsampwidth(), w.getframerate()),
                             (1, 2, WAV_RATE))
            self.assertEqual(w.getnframes(), int(REF_AUDIO_SECONDS * WAV_RATE))
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
           "/v1/audio/transcriptions", "/v1/audio/translations"}

    def test_variations_is_not_served(self):
        """No engine serves /v1/images/variations: vLLM-Omni v0.28.0 registers generations
        and edits but not variations, and vLLM, SGLang and llama.cpp implement no image
        routes. It could not be live-tested against anything, so it is not advertised."""
        with mock.patch.dict(os.environ, {"OPENAI_ROUTES": ""}):
            self.assertNotIn("/v1/images/variations", set(handlers()))

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


class TestAudioDuration(unittest.TestCase):
    """Duration is read from the container, because it is what an ASR request costs,
    what the engine bills (vLLM: usage.type == "duration"), and -- unlike the output
    tokens -- knowable before the request runs.

    Every expectation below was cross-checked against ffprobe on real ffmpeg-encoded
    fixtures during development: wav, flac, m4a, ogg and mp3 all landed at 1.00x, and
    webm at 1.17x through the byte-table fallback.
    """

    def test_wav_is_exact(self):
        from workers.openai.benchmark import synthetic_wav
        self.assertAlmostEqual(core._audio_seconds(synthetic_wav(7.5), "a.wav"),
                               7.5, places=2)

    def test_flac_reads_streaminfo(self):
        self.assertAlmostEqual(core._audio_seconds(_flac_like(7.5), "a.flac"),
                               7.5, places=2)

    def test_mp4_reads_mvhd(self):
        self.assertAlmostEqual(core._audio_seconds(_mp4_like(7.5), "a.m4a"),
                               7.5, places=2)

    def test_ogg_reads_the_last_granule_position(self):
        self.assertAlmostEqual(core._audio_seconds(_ogg_like(7.5), "a.ogg"),
                               7.5, places=2)

    def test_mp3_prefers_the_xing_frame_count_over_the_first_frame_bitrate(self):
        """A VBR file's first frame says nothing useful about the whole: reading the
        bitrate there measured 0.60x against ffprobe, the frame count 1.00x."""
        raw = _mp3_like(7.5)
        self.assertAlmostEqual(core._audio_seconds(raw, "a.mp3"), 7.5, places=1)

    def test_mp3_falls_back_to_the_bitrate_without_a_xing_header(self):
        """CBR, which is what the first frame's bitrate actually describes. The header
        below is MPEG-2 (16 kHz), where index 5 is 40 kbps -- reading it off the MPEG-1
        table instead calls it 64 kbps and under-counts the clip by a third, which is
        what a 16 kHz upload measured at 0.38x against ffprobe before the split."""
        payload = 40 * 1000 // 8 * 5                      # five seconds at 40 kbps
        raw = bytes([0xFF, 0xF3, 0x58, 0xC0]) + b"\x00" * payload
        self.assertAlmostEqual(core._audio_seconds(raw, "a.mp3"), 5.0, delta=0.1)

    def test_an_unparsable_container_uses_its_own_byte_rate(self):
        """webm has no cheap header parse, so it is priced from bytes -- per container,
        which is the part that was wrong before: one global constant charged 26s of MP3
        and 349s of Opus identically."""
        raw = b"\x1aE\xdf\xa3" + b"\x00" * 80000
        ogg_guess = core._audio_seconds(raw, "a.webm")
        self.assertAlmostEqual(ogg_guess, 80004 / core.AUDIO_BYTES_PER_SECOND["webm"],
                               places=2)

    def test_a_byte_rate_exists_for_every_accepted_format(self):
        """A format the worker accepts but cannot price would fall to the default."""
        for ext in core.AUDIO_TYPES:
            with self.subTest(ext):
                self.assertIn(ext, core.AUDIO_BYTES_PER_SECOND)


class TestBenchmarkClipCost(unittest.TestCase):
    """The SDK builds each benchmark payload INSIDE the timed window (backend.py starts
    its timer before the request loop), so whatever the generator costs is charged to
    the engine and comes straight off max_throughput.

    Measured before this was fixed: 1.10s to generate four 30s clips through
    struct.pack, against a real 2.18s benchmark window on a live instance -- about 41%
    of the measurement, halving the reported throughput of the engine it was supposed
    to be measuring. Under-reporting makes the worker shed load early rather than
    accept work it cannot do, so it is the safe direction and still wrong.
    """

    def test_a_reference_clip_is_cheap_to_build(self):
        import time
        from workers.openai.benchmark import REF_AUDIO_SECONDS, synthetic_wav

        started = time.time()
        for _ in range(4):
            synthetic_wav(REF_AUDIO_SECONDS)
        elapsed = time.time() - started
        self.assertLess(elapsed, 0.5,
                        f"four clips took {elapsed:.2f}s; at this cost the benchmark "
                        f"measures the worker, not the engine")

    def test_every_clip_differs(self):
        """Engines cache processed multimodal input by content hash. Identical audio
        every request would measure that cache from the second request onward."""
        import hashlib
        from workers.openai.benchmark import synthetic_wav

        digests = {hashlib.sha256(synthetic_wav(2.0)).hexdigest() for _ in range(4)}
        self.assertEqual(len(digests), 4)

    def test_a_tiled_clip_is_still_the_full_length(self):
        """Tiling must not shorten the audio: the encoder's work is the duration."""
        from workers.openai.benchmark import synthetic_wav
        for seconds in (1.0, 7.5, 30.0):
            with self.subTest(seconds):
                self.assertAlmostEqual(
                    core._audio_seconds(synthetic_wav(seconds), "a.wav"),
                    seconds, places=2)


class TestBenchmarkSpeechClip(unittest.TestCase):
    """The transcription benchmark uses REAL speech, because noise leaves the decoder
    idle: measured on whisper-large-v3, 30s of noise transcribes to 11 characters and
    30s of speech to 289, and the reported throughput differs by ~2x at every
    concurrency level tested (158 vs 74 audio-seconds/second at 1).

    The clip is downloaded, which is the same delivery the word corpus already uses AND
    the same hazard -- a required boot fetch is what takes a worker down when it fails.
    So it is optional by construction: no network, no speech, a warning, and a benchmark
    that still runs on synthetic noise. Delivery in production is an open decision.
    """

    def setUp(self):
        from workers.openai import benchmark
        self.bm = benchmark
        self._saved = benchmark._speech_clip
        benchmark._speech_clip = None

    def tearDown(self):
        self.bm._speech_clip = self._saved

    def test_a_failed_download_still_produces_a_reference_clip(self):
        with mock.patch.object(self.bm, "_fetch_speech", return_value=None), \
                mock.patch("builtins.print"):
            clip, _name = self.bm.benchmark_audio()
        self.assertAlmostEqual(core._audio_seconds(clip, "a.wav"),
                               self.bm.REF_AUDIO_SECONDS, places=2)

    def test_a_failed_download_is_not_silent(self):
        with mock.patch.object(self.bm, "_fetch_speech", return_value=None), \
                mock.patch("builtins.print") as printed:
            self.bm.benchmark_audio()
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("synthetic noise", said)

    def test_a_downloaded_clip_is_tiled_to_the_reference_length(self):
        """The sample is 11s; the workload unit is 30s, and a benchmark request has to
        weigh exactly one reference request like every other route's."""
        short = self.bm.synthetic_wav(11.0)
        with mock.patch.object(self.bm, "_fetch_speech", return_value=short):
            clip, _name = self.bm.benchmark_audio()
        self.assertAlmostEqual(core._audio_seconds(clip, "a.wav"),
                               self.bm.REF_AUDIO_SECONDS, places=2)

    def test_each_request_gets_distinct_bytes(self):
        short = self.bm.synthetic_wav(11.0)
        with mock.patch.object(self.bm, "_fetch_speech", return_value=short):
            import hashlib
            digests = {hashlib.sha256(self.bm.benchmark_audio()[0]).hexdigest()
                       for _ in range(4)}
        self.assertEqual(len(digests), 4, "an identical clip measures the engine's cache")

    def test_the_sample_is_fetched_once_not_per_request(self):
        """for_test() runs inside the SDK's timed window; a fetch per request would be
        measured as engine time."""
        short = self.bm.synthetic_wav(11.0)
        with mock.patch.object(self.bm, "_fetch_speech", return_value=short) as fetch:
            for _ in range(5):
                self.bm.benchmark_audio()
        self.assertEqual(fetch.call_count, 1)

    def test_an_unreadable_download_falls_back(self):
        """A 404 page or an HTML redirect is bytes too, and wave will not open it."""
        with mock.patch.object(self.bm, "_fetch_speech", return_value=b"<html>404</html>"), \
                mock.patch("builtins.print"):
            clip, _name = self.bm.benchmark_audio()
        self.assertAlmostEqual(core._audio_seconds(clip, "a.wav"),
                               self.bm.REF_AUDIO_SECONDS, places=2)


class TestOperatorSuppliedClip(unittest.TestCase):
    """BENCHMARK_AUDIO_URL exists so a deployment can benchmark on audio that matches
    its own traffic -- its language, noise floor and speech density, which is what
    drives decode work. That only helps if the formats operators actually have are
    accepted: before this, an mp3 fetched fine and then silently reverted to noise."""

    def setUp(self):
        from workers.openai import benchmark
        self.bm = benchmark
        self._saved = benchmark._speech_clip
        benchmark._speech_clip = None

    def tearDown(self):
        self.bm._speech_clip = self._saved

    def _fetch(self, url, raw):
        with mock.patch.object(self.bm, "BENCHMARK_AUDIO_URL", url), \
                mock.patch.object(self.bm, "_fetch_speech", return_value=raw), \
                mock.patch("builtins.print"):
            return core.TranscriptionPayload.for_test()

    def test_a_compressed_clip_is_used_as_supplied(self):
        """It cannot be resized without a decoder, so it is sent whole and priced by
        its real duration -- still workload over time, still a valid throughput."""
        payload = self._fetch("https://host/calls.mp3", _mp3_like(7.5))
        name, _data, ctype = payload.generate_payload_multipart()["file"]
        self.assertEqual((name, ctype), ("calls.mp3", "audio/mpeg"))
        self.assertAlmostEqual(payload.count_workload(),
                               7.5 / self.bm.REF_AUDIO_SECONDS * 500, delta=25)

    def test_a_signed_url_does_not_leak_its_query_into_the_filename(self):
        """The engine picks its decoder from the extension; `a.mp3?sig=...` is not one."""
        payload = self._fetch("https://host/a.mp3?sig=deadbeef&x=1", _mp3_like(5.0))
        self.assertEqual(payload.filename, "a.mp3")

    def test_wav_is_still_normalised_to_the_reference(self):
        payload = self._fetch("https://host/clip.wav", self.bm.synthetic_wav(11.0))
        self.assertEqual(payload.filename, "benchmark.wav")
        self.assertAlmostEqual(payload.count_workload(), 500.0, delta=1)

    def test_a_container_the_route_would_refuse_falls_back(self):
        """A clip the worker would reject from a caller must not be sent by the
        benchmark either, or the benchmark tests something no request can do."""
        with mock.patch.object(self.bm, "BENCHMARK_AUDIO_URL", "https://host/a.aiff"), \
                mock.patch.object(self.bm, "_fetch_speech", return_value=b"FORM....AIFF"), \
                mock.patch("builtins.print") as printed:
            payload = core.TranscriptionPayload.for_test()
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("does not accept", said)
        self.assertAlmostEqual(payload.count_workload(), 500.0, delta=1)   # noise

    def test_a_compressed_clip_is_not_varied_and_says_so(self):
        """Only WAV can be rebuilt, so a compressed clip goes out byte-identical every
        request. That is a real caveat -- an engine caching by content hash would serve
        it from cache -- so it is stated rather than hidden."""
        with mock.patch.object(self.bm, "BENCHMARK_AUDIO_URL", "https://host/a.mp3"), \
                mock.patch.object(self.bm, "_fetch_speech", return_value=_mp3_like(5.0)), \
                mock.patch("builtins.print") as printed:
            first, _ = self.bm.benchmark_audio(accepted=core.AUDIO_TYPES,
                                               probe=core.parsed_audio_seconds)
            second, _ = self.bm.benchmark_audio(accepted=core.AUDIO_TYPES,
                                                probe=core.parsed_audio_seconds)
        self.assertEqual(first, second)
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("cache", said)

    def test_an_extension_does_not_vouch_for_the_bytes(self):
        """A 404 page served from a .wav URL has an accepted extension and is not
        audio. Before the bytes were probed, it was forwarded to the engine as one."""
        with mock.patch.object(self.bm, "BENCHMARK_AUDIO_URL", "https://host/clip.mp3"), \
                mock.patch.object(self.bm, "_fetch_speech",
                                  return_value=b"<!DOCTYPE html><title>404</title>"), \
                mock.patch("builtins.print") as printed:
            payload = core.TranscriptionPayload.for_test()
        self.assertEqual(payload.filename, "benchmark.wav")          # fell back to noise
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("not audio this can read", said)

    def test_real_audio_in_an_unaccepted_container_is_still_refused(self):
        """Valid mp3 bytes served from a .opus URL: the engine picks its decoder from
        the extension we send, and _file_part would refuse that one from a caller, so
        the benchmark must not send it either. Both halves have to hold -- readable
        bytes AND an extension this route accepts."""
        self.assertNotIn("opus", core.AUDIO_TYPES)
        with mock.patch.object(self.bm, "BENCHMARK_AUDIO_URL", "https://host/a.opus"), \
                mock.patch.object(self.bm, "_fetch_speech", return_value=_mp3_like(5.0)), \
                mock.patch("builtins.print") as printed:
            payload = core.TranscriptionPayload.for_test()
        self.assertEqual(payload.filename, "benchmark.wav")          # fell back to noise
        said = " ".join(str(c.args[0]) for c in printed.call_args_list)
        self.assertIn("does not accept", said)
