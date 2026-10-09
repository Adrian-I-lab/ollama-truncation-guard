"""End-to-end tests: client -> proxy -> fake Ollama, on 127.0.0.1 only.

Worked numbers. The proxy estimates len // 6 tokens. The fake counts len // 4.
A 6,000-character prompt is expected as 1,000 tokens and the honest fake reports 1,500.
"""
import contextlib
import io
import json
import threading
import unittest
import urllib.error
import urllib.request

import proxy
from tests.fake_ollama import start_fake


def start_proxy(upstream, *flags):
    server = proxy.make_server(proxy.parse_args(["--port", "0", "--upstream", upstream, *flags]))
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def call(base, path, body=None, headers=None, method=None):
    """Return (status, text). HTTP errors are returned, not raised."""
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data, {"Content-Type": "application/json",
                                                      **(headers or {})}, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode()


def generate(chars, num_ctx=2048, stream=False):
    return {"model": "fake", "prompt": "x" * chars, "stream": stream, "options": {"num_ctx": num_ctx}}


class ProxyCase(unittest.TestCase):
    def setUp(self):
        self.log = io.StringIO()
        redirect = contextlib.redirect_stderr(self.log)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def fake(self, **kw):
        server, base = start_fake(**kw)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, base

    def proxy(self, upstream, *flags):
        server, base = start_proxy(upstream, *flags)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return base

    def guarded(self, *flags, **fake_kw):
        fake, base = self.fake(**fake_kw)
        return fake, self.proxy(base, *flags)


class FailThenPass(ProxyCase):
    def test_1_non_streamed_502_then_200(self):
        _, bad = self.guarded(truncate_to=600)
        status, text = call(bad, "/api/generate", generate(6000))
        self.assertEqual(status, 502)
        out = json.loads(text)
        self.assertEqual((out["expected_tokens"], out["prompt_eval_count"]), (1000, 600))
        self.assertEqual(out["num_ctx"], 2048)

        _, good = self.guarded()
        status, text = call(good, "/api/generate", generate(6000))
        self.assertEqual(status, 200)
        out = json.loads(text)
        self.assertEqual((out["response"], out["prompt_eval_count"]), ("fake answer", 1500))

    def test_1b_half_window_signature(self):
        _, base = self.guarded(truncate_to="half")
        status, text = call(base, "/api/generate", generate(9000))
        self.assertEqual(status, 502)
        out = json.loads(text)
        self.assertEqual((out["expected_tokens"], out["prompt_eval_count"]), (1500, 1026))
        self.assertTrue(any("half-window" in r for r in out["reasons"]))

    def test_2_streamed_buffer_502_then_200(self):
        _, bad = self.guarded(truncate_to=600)
        status, text = call(bad, "/api/generate", generate(6000, stream=True))
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(text)["prompt_eval_count"], 600)

        _, good = self.guarded()
        status, text = call(good, "/api/generate", generate(6000, stream=True))
        self.assertEqual(status, 200)
        lines = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[-1]["prompt_eval_count"], 1500)

    def test_chat_endpoint_is_guarded(self):
        msgs = [{"role": "user", "content": "x" * 3000}, {"role": "user", "content": "y" * 3000}]
        body = {"model": "fake", "messages": msgs, "stream": False, "options": {"num_ctx": 2048}}
        _, bad = self.guarded(truncate_to=600)
        self.assertEqual(call(bad, "/api/chat", body)[0], 502)
        _, good = self.guarded()
        status, text = call(good, "/api/chat", body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["message"]["content"], "fake answer")

    def test_missing_count_is_refused(self):
        _, base = self.guarded(drop_count=True)
        status, text = call(base, "/api/generate", generate(6000))
        self.assertEqual(status, 502)
        self.assertIn("missing prompt_eval_count", text)


class StreamModes(ProxyCase):
    def test_3_passthrough_replaces_final_line_with_error(self):
        _, base = self.guarded("--stream-mode", "passthrough", truncate_to=600)
        status, text = call(base, "/api/generate", generate(6000, stream=True))
        self.assertEqual(status, 200)  # headers were already sent when the stream began
        lines = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(len(lines), 4)
        self.assertEqual("".join(o.get("response", "") for o in lines[:3]), "fake answer")
        self.assertIn("error", lines[-1])
        self.assertTrue(lines[-1]["done"])
        self.assertEqual(lines[-1]["prompt_eval_count"], 600)

    def test_passthrough_honest_stream_is_unchanged(self):
        _, base = self.guarded("--stream-mode", "passthrough")
        status, text = call(base, "/api/generate", generate(6000, stream=True))
        self.assertEqual(status, 200)
        lines = [json.loads(line) for line in text.splitlines()]
        self.assertEqual([o["done"] for o in lines], [False, False, False, True])
        self.assertNotIn("error", lines[-1])

    def test_passthrough_non_streamed_request_still_gets_502(self):
        _, base = self.guarded("--stream-mode", "passthrough", truncate_to=600)
        self.assertEqual(call(base, "/api/generate", generate(6000))[0], 502)


class Preflight(ProxyCase):
    def test_4_preflight_413_without_calling_the_model(self):
        fake, base = self.guarded()
        status, text = call(base, "/api/generate", generate(15000))
        self.assertEqual(status, 413)
        out = json.loads(text)
        self.assertEqual((out["expected_tokens"], out["num_ctx"]), (2500, 2048))
        self.assertEqual(fake.calls, [])

    def test_num_ctx_flag_used_when_request_sets_none(self):
        fake, base = self.guarded("--num-ctx", "2048")
        status, _ = call(base, "/api/generate", {"model": "fake", "prompt": "x" * 15000, "stream": False})
        self.assertEqual(status, 413)
        self.assertEqual(fake.calls, [])


class Inputs(ProxyCase):
    def test_5_header_overrides_estimate(self):
        _, base = self.guarded()  # honest fake reports 1,500
        hdr = {"X-Expected-Prompt-Tokens": "3000"}  # the estimate alone would be 1,000
        status, text = call(base, "/api/generate", generate(6000, num_ctx=4096), hdr)
        self.assertEqual(status, 502)
        self.assertEqual(json.loads(text)["expected_tokens"], 3000)
        hdr = {"X-Expected-Prompt-Tokens": "1500"}
        status, _ = call(base, "/api/generate", generate(6000, num_ctx=4096), hdr)
        self.assertEqual(status, 200)

    def test_6_control_guard_disabled_lets_truncation_through(self):
        _, base = self.guarded("--tolerance", "0", truncate_to=600)
        body = {"model": "fake", "prompt": "x" * 6000, "stream": False}  # no num_ctx anywhere
        status, text = call(base, "/api/generate", body)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["prompt_eval_count"], 600)

    def test_7_tags_pass_through(self):
        _, base = self.guarded()
        status, text = call(base, "/api/tags")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text), {"models": [{"name": "fake:latest"}]})

    def test_upstream_http_error_is_forwarded(self):
        _, base = self.guarded()
        status, text = call(base, "/api/unknown", {"x": 1})
        self.assertEqual(status, 404)
        self.assertIn("not found", text)

    def test_upstream_unreachable_gives_502(self):
        fake, upstream = self.fake()
        fake.shutdown()
        fake.server_close()
        base = self.proxy(upstream)
        status, text = call(base, "/api/generate", generate(6000))
        self.assertEqual(status, 502)
        self.assertIn("upstream unreachable", text)


class Logging(ProxyCase):
    def test_8_refusal_log_line_has_both_numbers(self):
        _, base = self.guarded(truncate_to=600)
        call(base, "/api/generate", generate(6000))
        lines = [ln for ln in self.log.getvalue().splitlines() if ln.startswith("REFUSED")]
        self.assertEqual(len(lines), 1)
        for part in ("path=/api/generate", "expected_tokens=1000", "prompt_eval_count=600",
                     "num_ctx=2048", "reasons="):
            self.assertIn(part, lines[0])

    def test_honest_request_logs_nothing(self):
        _, base = self.guarded()
        call(base, "/api/generate", generate(6000))
        self.assertNotIn("REFUSED", self.log.getvalue())


if __name__ == "__main__":
    unittest.main()
