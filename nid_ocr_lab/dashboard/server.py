from __future__ import annotations

import argparse
import json
import mimetypes
from contextlib import suppress
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from nid_ocr_lab.dashboard.filters import FILTER_MODES, render_image, render_mask_overlay, render_sdk_crop
from nid_ocr_lab.dashboard.indexer import build_index, dataset_health, load_annotation

STATIC_DIR = Path(__file__).with_name("static")


class DashboardHandler(BaseHTTPRequestHandler):
    index_cache: list[dict] | None = None

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_static("index.html")
        elif parsed.path == "/api/samples":
            self.send_json(self.samples())
        elif parsed.path == "/api/health":
            self.send_json(dataset_health())
        elif parsed.path == "/api/annotation":
            query = parse_qs(parsed.query)
            self.send_json(load_annotation(first(query, "path")) or {})
        elif parsed.path == "/image":
            self.send_image(parsed.query)
        elif parsed.path == "/mask-overlay":
            self.send_mask_overlay(parsed.query)
        elif parsed.path == "/sdk-crop":
            self.send_sdk_crop(parsed.query)
        elif parsed.path.startswith("/static/"):
            self.send_static(parsed.path.removeprefix("/static/"))
        else:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def samples(self) -> list[dict]:
        if DashboardHandler.index_cache is None:
            DashboardHandler.index_cache = build_index()
        return DashboardHandler.index_cache

    def send_image(self, query_string: str) -> None:
        query = parse_qs(query_string)
        path = first(query, "path")
        mode = first(query, "mode") or "original"
        if not path or mode not in FILTER_MODES:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid image request")
            return
        try:
            payload, _info = render_image(unquote(path), mode=mode)
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_mask_overlay(self, query_string: str) -> None:
        query = parse_qs(query_string)
        image_path = first(query, "image")
        mask_path = first(query, "mask")
        if not image_path or not mask_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid mask overlay request")
            return
        try:
            payload, _info = render_mask_overlay(unquote(image_path), unquote(mask_path))
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "Image or mask not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_sdk_crop(self, query_string: str) -> None:
        query = parse_qs(query_string)
        image_path = first(query, "image")
        annotation_path = first(query, "annotation")
        mode = first(query, "mode") or "original"
        if not image_path or not annotation_path:
            self.send_error(HTTPStatus.BAD_REQUEST, "Invalid SDK crop request")
            return
        annotation = load_annotation(unquote(annotation_path)) or {}
        points = first_quad_points(annotation)
        if not points:
            self.send_error(HTTPStatus.BAD_REQUEST, "No 4-point annotation quad found")
            return
        try:
            payload, _info = render_sdk_crop(unquote(image_path), points, mode=mode)
        except (OSError, ValueError):
            self.send_error(HTTPStatus.NOT_FOUND, "SDK crop source not found")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(payload)

    def send_json(self, payload: object) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def send_static(self, name: str) -> None:
        path = (STATIC_DIR / name).resolve()
        if not str(path).startswith(str(STATIC_DIR.resolve())) or not path.exists():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        data = path.read_bytes()
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        with suppress(BrokenPipeError, ConnectionResetError):
            self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        return


def first(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    return values[0] if values else None


def first_quad_points(annotation: dict) -> list[list[float]] | None:
    for shape in annotation.get("shapes", []):
        points = shape.get("points") or []
        if len(points) == 4:
            return points
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SmartScan OCR R&D dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard: http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
