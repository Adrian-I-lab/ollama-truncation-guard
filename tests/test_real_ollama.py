"""Optional check against a real Ollama server. Skipped unless both variables are set:

    OLLAMA_URL=http://127.0.0.1:11434 OTG_MODEL=<a small local model> \\
        python3 -m unittest tests.test_real_ollama -v
"""
import os
import unittest

from tests.test_proxy import call, start_proxy

URL, MODEL = os.environ.get("OLLAMA_URL"), os.environ.get("OTG_MODEL")


@unittest.skipUnless(URL and MODEL, "set OLLAMA_URL and OTG_MODEL to run against a real Ollama")
class RealOllama(unittest.TestCase):
    def setUp(self):
        self.server, self.base = start_proxy(URL)
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)

    def test_long_prompt_is_refused(self):
        body = {"model": MODEL, "prompt": "word " * 2000, "stream": False,
                "options": {"num_ctx": 512}}
        status, text = call(self.base, "/api/generate", body, {"X-Expected-Prompt-Tokens": "2000"})
        self.assertIn(status, (413, 502), text)

    def test_short_prompt_passes(self):
        body = {"model": MODEL, "prompt": "Say hello in one word.", "stream": False,
                "options": {"num_ctx": 2048}}
        status, text = call(self.base, "/api/generate", body)
        self.assertEqual(status, 200, text)


if __name__ == "__main__":
    unittest.main()
