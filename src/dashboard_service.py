from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd

from src.inference import explain_records, load_production_artifacts, predict_records


def _post_json(url: str, payload: dict[str, Any], timeout: float = 60.0) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


@dataclass
class DashboardService:
    project_root: Path
    api_base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 60.0
    _artifacts: Any | None = None

    def __post_init__(self) -> None:
        self.project_root = Path(self.project_root)

    def _normalize_records(self, records: pd.DataFrame | dict | list[dict]) -> list[dict]:
        if isinstance(records, pd.DataFrame):
            normalized = records.where(pd.notna(records), None)
            return normalized.to_dict(orient="records")
        if isinstance(records, dict):
            return [records]
        return records

    def api_is_available(self) -> bool:
        try:
            payload = _get_json(f"{self.api_base_url}/health", timeout=self.timeout_seconds)
            return payload.get("status") == "ok"
        except (URLError, TimeoutError, ValueError, OSError):
            return False

    def connection_mode(self) -> str:
        return "api" if self.api_is_available() else "fallback"

    def mode_label(self) -> str:
        return "Connected to local API" if self.connection_mode() == "api" else "Running in direct local inference mode"

    def _get_artifacts(self):
        if self._artifacts is None:
            self._artifacts = load_production_artifacts(self.project_root)
        return self._artifacts

    def get_metadata(self) -> dict[str, Any]:
        if self.api_is_available():
            return _get_json(f"{self.api_base_url}/metadata", timeout=self.timeout_seconds)
        artifacts = self._get_artifacts()
        return {
            "service": "unsw-ids-streamlit-fallback",
            "final_model": artifacts.model_name,
            "class_names": artifacts.feature_info.get("class_names", []),
            "selected_feature_count": len(artifacts.feature_info.get("selected_feature_names", [])),
        }

    def predict(self, records: pd.DataFrame | dict | list[dict]) -> list[dict[str, Any]]:
        normalized = self._normalize_records(records)
        if self.api_is_available():
            response = _post_json(
                f"{self.api_base_url}/predict",
                {"records": normalized},
                timeout=self.timeout_seconds,
            )
            return response.get("predictions", [])
        return predict_records(normalized, self._get_artifacts())

    def explain(self, records: pd.DataFrame | dict | list[dict], top_n: int = 5) -> list[dict[str, Any]]:
        normalized = self._normalize_records(records)
        if self.api_is_available():
            response = _post_json(
                f"{self.api_base_url}/explain",
                {"records": normalized},
                timeout=self.timeout_seconds,
            )
            return response.get("explanations", [])
        return explain_records(normalized, self._get_artifacts(), top_n=top_n)
