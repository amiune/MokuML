from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import urlparse

from worker.api import handle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MokuML local worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()

    try:
        import uvicorn

        uvicorn.run(
            "worker.fastapi_app:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
    except Exception as exc:
        print(f"FastAPI/uvicorn unavailable, starting stdlib worker server: {exc}", flush=True)
        server = ThreadingHTTPServer((args.host, args.port), _Handler)
        print(f"MokuML stdlib worker listening on http://{args.host}:{args.port}", flush=True)
        server.serve_forever()


class _Handler(BaseHTTPRequestHandler):
    server_version = "MokuMLWorker/0.1"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        self._write_json(handle("GET", path))

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(length) if length else b"{}"
        try:
            payload: Dict[str, Any] = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as exc:
            self._write_json({"ok": False, "message": f"Invalid JSON body: {exc}"}, status=400)
            return
        self._write_json(handle("POST", path, payload))

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    main()

