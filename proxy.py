"""Guarded inference proxy for a local Ollama server.

Every request is forwarded to Ollama. On /api/generate and /api/chat the proxy compares the
prompt_eval_count Ollama reports with the tokens the client sent, and refuses any answer built
on a silently truncated prompt. This is a localhost tool. It has no auth and no TLS.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import guard

GUARDED = ("/api/generate", "/api/chat")


class Handler(BaseHTTPRequestHandler):
    args = None  # set by make_server

    def log_message(self, *args):
        pass

    def start(self, status, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype or "application/json")
        self.end_headers()

    def reply(self, status, data, ctype="application/json"):
        self.start(status, ctype)
        self.wfile.write(data if isinstance(data, bytes) else json.dumps(data).encode())

    def refusal(self, info, reasons, error="prompt truncation detected; answer refused"):
        line = " ".join(f"{k}={v}" for k, v in info.items())
        print(f'REFUSED {line} reasons="{"; ".join(reasons)}"', file=sys.stderr, flush=True)
        body = {k: v for k, v in info.items() if k != "path"}
        return {"error": error, **body, "reasons": reasons}

    def upstream(self, raw):
        """Open the upstream response, or answer the client and return None on error."""
        ctype = self.headers.get("Content-Type", "application/json")
        req = urllib.request.Request(self.args.upstream + self.path, data=raw or None,
                                     method=self.command, headers={"Content-Type": ctype})
        try:
            return urllib.request.urlopen(req)
        except urllib.error.HTTPError as err:
            self.reply(err.code, err.read(), err.headers.get("Content-Type"))
        except (urllib.error.URLError, OSError) as err:
            self.reply(502, {"error": f"upstream unreachable: {err}"})
        return None

    def forward(self, raw):
        if (resp := self.upstream(raw)) is None:
            return
        self.start(resp.status, resp.headers.get("Content-Type"))
        while chunk := resp.read1(65536):
            self.wfile.write(chunk)
            self.wfile.flush()

    def read_body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def do_GET(self):
        self.forward(self.read_body())

    do_DELETE = do_GET

    def do_POST(self):
        raw = self.read_body()
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = None
        if self.path not in GUARDED or not isinstance(body, dict):
            return self.forward(raw)
        a, header = self.args, self.headers.get("X-Expected-Prompt-Tokens", "").strip()
        expected = int(header) if header.isdigit() else guard.estimate_tokens(
            guard.prompt_text(self.path, body), a.chars_per_token)
        num_ctx = (body.get("options") or {}).get("num_ctx") or a.num_ctx
        info = {"path": self.path, "expected_tokens": expected, "num_ctx": num_ctx}
        reason = guard.preflight(expected, num_ctx)
        if reason:
            return self.reply(413, self.refusal(info, [reason], "prompt cannot fit; model not called"))
        resp = self.upstream(raw)
        if resp is not None:
            self.stream(resp, info, a.stream_mode == "passthrough" and body.get("stream", True))

    def stream(self, resp, info, passthrough):
        """Relay NDJSON lines. Hold the done line (buffer mode: every line) until it is judged."""
        held, final = [], None
        if passthrough:
            self.start(resp.status, resp.headers.get("Content-Type"))
        for line in resp:
            try:
                obj = json.loads(line)
            except ValueError:
                obj = {}
            if isinstance(obj, dict) and obj.get("done"):
                final = obj
            if passthrough and final is None:
                self.wfile.write(line)
                self.wfile.flush()
            else:
                held.append(line)
        count = final.get("prompt_eval_count") if final else None
        reasons = guard.check(count, info["expected_tokens"], info["num_ctx"], self.args.tolerance)
        if reasons:
            refusal = self.refusal({**info, "prompt_eval_count": count}, reasons)
            if not passthrough:
                return self.reply(502, refusal)
            held = [json.dumps({**refusal, "done": True}).encode() + b"\n"]
        if not passthrough:
            self.start(resp.status, resp.headers.get("Content-Type"))
        self.wfile.write(b"".join(held))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Refuse Ollama answers built on a truncated prompt.")
    p.add_argument("--upstream", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--host", default="127.0.0.1", help="address to bind")
    p.add_argument("--port", type=int, default=11435, help="port to bind")
    p.add_argument("--num-ctx", type=int, default=None, help="window when a request sets none")
    p.add_argument("--tolerance", type=float, default=0.85, help="0 disables the ratio check")
    p.add_argument("--chars-per-token", type=float, default=6.0, help="estimate divisor")
    p.add_argument("--stream-mode", choices=["buffer", "passthrough"], default="buffer")
    return p.parse_args(argv)


def make_server(args):
    return ThreadingHTTPServer((args.host, args.port), type("Bound", (Handler,), {"args": args}))


def main(argv=None):
    args = parse_args(argv)
    server = make_server(args)
    print(f"guarding {args.upstream} on http://{args.host}:{server.server_address[1]}", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
