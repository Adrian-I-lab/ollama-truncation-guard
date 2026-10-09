"""A stdlib fake of a local Ollama server that can truncate prompts on purpose.

Tokens are counted as len(text) // 4. Knobs:
  truncate_to=None    report the true count (honest)
  truncate_to="half"  if the prompt overflows num_ctx, report num_ctx // 2 + 2
  truncate_to=<int>   always report that number
  drop_count=True     omit prompt_eval_count entirely
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ANSWER = "fake answer"


def _handler(truncate_to, drop_count, calls):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, status, data, ctype="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/tags":
                self._send(200, json.dumps({"models": [{"name": "fake:latest"}]}).encode())
            else:
                self._send(404, b'{"error": "not found"}')

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if self.path not in ("/api/generate", "/api/chat"):
                return self._send(404, b'{"error": "not found"}')
            calls.append(self.path)
            chat = self.path == "/api/chat"
            if chat:
                text = "".join(m.get("content", "") for m in body.get("messages", []))
            else:
                text = body.get("system", "") + body.get("prompt", "")
            num_ctx = (body.get("options") or {}).get("num_ctx", 2048)
            count = len(text) // 4
            if truncate_to == "half":
                count = num_ctx // 2 + 2 if count > num_ctx else count
            elif truncate_to is not None:
                count = truncate_to

            def chunk(text, done):
                obj = {"model": body.get("model", "fake:latest"), "done": done}
                if chat:
                    obj["message"] = {"role": "assistant", "content": text}
                else:
                    obj["response"] = text
                if done:
                    obj["eval_count"] = 3
                    if not drop_count:
                        obj["prompt_eval_count"] = count
                return obj

            if body.get("stream", True) is False:
                return self._send(200, json.dumps(chunk(ANSWER, True)).encode())
            parts = ["fake ", "ans", "wer"]
            lines = [chunk(p, False) for p in parts] + [chunk("", True)]
            data = "".join(json.dumps(o) + "\n" for o in lines).encode()
            self._send(200, data, "application/x-ndjson")

    return Handler


def start_fake(truncate_to=None, drop_count=False, port=0):
    """Start the fake on 127.0.0.1 (port 0 picks a free port). Returns (server, base_url).

    server.calls lists the generate and chat paths received. Stop with server.shutdown().
    """
    calls = []
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(truncate_to, drop_count, calls))
    server.calls = calls
    threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run the fake Ollama server for demos.")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--truncate-to", default=None, help='"half" or an integer')
    p.add_argument("--drop-count", action="store_true")
    a = p.parse_args()
    cut = a.truncate_to if a.truncate_to in (None, "half") else int(a.truncate_to)
    srv, url = start_fake(cut, a.drop_count, a.port)
    print(f"fake Ollama on {url} truncate_to={cut}", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
