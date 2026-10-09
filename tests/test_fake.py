import json
import unittest
import urllib.request

from tests.fake_ollama import start_fake


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return resp.headers.get("Content-Type"), resp.read().decode()


class FakeOllamaTests(unittest.TestCase):
    def run_fake(self, **kw):
        server, base = start_fake(**kw)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server, base

    def test_honest_count(self):
        _, base = self.run_fake()
        _, out = post(base, "/api/generate", {"prompt": "x" * 4000, "stream": False})
        self.assertEqual(json.loads(out)["prompt_eval_count"], 1000)

    def test_half_truncation_when_overflowing(self):
        _, base = self.run_fake(truncate_to="half")
        body = {"prompt": "x" * 9000, "stream": False, "options": {"num_ctx": 2048}}
        _, out = post(base, "/api/generate", body)
        self.assertEqual(json.loads(out)["prompt_eval_count"], 1026)

    def test_half_mode_honest_when_it_fits(self):
        _, base = self.run_fake(truncate_to="half")
        body = {"prompt": "x" * 4000, "stream": False, "options": {"num_ctx": 2048}}
        _, out = post(base, "/api/generate", body)
        self.assertEqual(json.loads(out)["prompt_eval_count"], 1000)

    def test_fixed_count(self):
        _, base = self.run_fake(truncate_to=600)
        _, out = post(base, "/api/generate", {"prompt": "x" * 6000, "stream": False})
        self.assertEqual(json.loads(out)["prompt_eval_count"], 600)

    def test_drop_count(self):
        _, base = self.run_fake(drop_count=True)
        _, out = post(base, "/api/generate", {"prompt": "hi", "stream": False})
        self.assertNotIn("prompt_eval_count", json.loads(out))

    def test_streamed_is_four_ndjson_lines(self):
        _, base = self.run_fake()
        ctype, out = post(base, "/api/generate", {"prompt": "x" * 400})
        self.assertEqual(ctype, "application/x-ndjson")
        lines = [json.loads(line) for line in out.splitlines()]
        self.assertEqual(len(lines), 4)
        self.assertEqual([o["done"] for o in lines], [False, False, False, True])
        self.assertEqual(lines[-1]["prompt_eval_count"], 100)

    def test_chat_counts_every_message(self):
        server, base = self.run_fake()
        body = {"messages": [{"role": "user", "content": "x" * 400},
                             {"role": "user", "content": "y" * 400}], "stream": False}
        _, out = post(base, "/api/chat", body)
        obj = json.loads(out)
        self.assertEqual(obj["prompt_eval_count"], 200)
        self.assertEqual(obj["message"]["content"], "fake answer")
        self.assertEqual(server.calls, ["/api/chat"])

    def test_tags(self):
        _, base = self.run_fake()
        with urllib.request.urlopen(base + "/api/tags") as resp:
            self.assertEqual(json.load(resp)["models"][0]["name"], "fake:latest")


if __name__ == "__main__":
    unittest.main()
