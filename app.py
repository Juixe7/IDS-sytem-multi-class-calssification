from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from wsgiref.simple_server import make_server

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

from src.config import Config
from src.inference import explain_records, load_production_artifacts, predict_records


CONFIG = Config(project_root=PROJECT_ROOT)
ARTIFACTS = load_production_artifacts(PROJECT_ROOT)


def _json_response(start_response, status: str, payload: dict) -> list[bytes]:
    body = json.dumps(payload, indent=2).encode("utf-8")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def _read_json(environ) -> dict | list[dict]:
    try:
        length = int(environ.get("CONTENT_LENGTH", "0") or "0")
    except ValueError:
        length = 0
    raw_body = environ["wsgi.input"].read(length) if length else b""
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


def _metadata_payload() -> dict:
    manifest = {}
    if CONFIG.final_model_manifest_path.exists():
        manifest = json.loads(CONFIG.final_model_manifest_path.read_text(encoding="utf-8"))
    repo_audit = {}
    if CONFIG.repo_audit_json_path.exists():
        repo_audit = json.loads(CONFIG.repo_audit_json_path.read_text(encoding="utf-8"))
    return {
        "service": "unsw-ids-saas-api",
        "final_model": manifest.get("final_model", ARTIFACTS.model_name),
        "class_names": ARTIFACTS.feature_info.get("class_names", []),
        "selected_feature_count": len(ARTIFACTS.feature_info.get("selected_feature_names", [])),
        "repo_benchmark_audit_available": CONFIG.repo_audit_json_path.exists(),
        "repo_benchmark_summary": {
            "with_leakage_accuracy": repo_audit.get("with_leakage", {}).get("accuracy"),
            "without_leakage_accuracy": repo_audit.get("without_attack_cat_dummies", {}).get("accuracy"),
        }
        if repo_audit
        else None,
    }


def _records_from_payload(payload: dict | list[dict]) -> dict | list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    return payload


def application(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    try:
        if method == "GET" and path in {"/", "/health"}:
            return _json_response(start_response, "200 OK", {"status": "ok", **_metadata_payload()})
        if method == "GET" and path == "/metadata":
            return _json_response(start_response, "200 OK", _metadata_payload())
        if method == "POST" and path == "/predict":
            payload = _read_json(environ)
            records = _records_from_payload(payload)
            predictions = predict_records(records, ARTIFACTS)
            return _json_response(start_response, "200 OK", {"predictions": predictions})
        if method == "POST" and path == "/explain":
            payload = _read_json(environ)
            records = _records_from_payload(payload)
            explanations = explain_records(records, ARTIFACTS, top_n=5)
            return _json_response(start_response, "200 OK", {"explanations": explanations})
        return _json_response(
            start_response,
            "404 Not Found",
            {"error": "Route not found", "available_routes": ["GET /health", "GET /metadata", "POST /predict", "POST /explain"]},
        )
    except Exception as exc:
        return _json_response(start_response, "500 Internal Server Error", {"error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the UNSW IDS SaaS API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with make_server(args.host, args.port, application) as server:
        print(f"[INFO] Serving UNSW IDS SaaS API on http://{args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
