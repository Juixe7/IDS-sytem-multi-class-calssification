from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


PROJECT_ROOT = Path(__file__).resolve().parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
SCREENSHOTS_DIR = ARTIFACTS_DIR / "screenshots"
LOGS_DIR = ARTIFACTS_DIR / "logs"
API_URL = "http://127.0.0.1:8000/health"
DASHBOARD_URL = "http://127.0.0.1:8501"


def wait_for_url(url: str, timeout_seconds: float = 120.0) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, OSError):
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {url}")


def chrome_path() -> Path:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Google Chrome was not found on this machine.")


def create_sample_csv() -> Path:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = SCREENSHOTS_DIR / "sample_input_with_id.csv"
    row = pd.read_csv(PROJECT_ROOT / "unsw" / "UNSW_NB15_testing-set.csv", nrows=1)
    row = row.drop(columns=["attack_cat", "label"], errors="ignore")
    row.to_csv(sample_path, index=False)
    return sample_path


def start_process(command: list[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
    return subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=env,
    )


def stop_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def capture_screenshots() -> dict:
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    sample_csv = create_sample_csv()
    api_proc = None
    dashboard_proc = None
    results = {
        "api_started": False,
        "dashboard_started": False,
        "overview_loaded": False,
        "manual_prediction_rendered": False,
        "csv_with_id_prediction_rendered": False,
        "explain_view_loaded": False,
        "screenshots": {},
        "sample_csv": str(sample_csv),
    }
    try:
        api_proc = start_process(
            [sys.executable, "app.py", "--host", "127.0.0.1", "--port", "8000"],
            LOGS_DIR / "final_delivery_api.log",
        )
        wait_for_url(API_URL, timeout_seconds=60.0)
        results["api_started"] = True

        dashboard_proc = start_process(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard.py",
                "--server.headless",
                "true",
                "--server.port",
                "8501",
            ],
            LOGS_DIR / "final_delivery_dashboard.log",
        )
        wait_for_url(DASHBOARD_URL, timeout_seconds=90.0)
        results["dashboard_started"] = True

        chrome = str(chrome_path())
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                executable_path=chrome,
                headless=True,
                args=["--disable-gpu", "--no-sandbox"],
            )
            page = browser.new_page(viewport={"width": 1600, "height": 1000})
            page.goto(DASHBOARD_URL, wait_until="networkidle")
            page.get_by_text("UNSW IDS Dashboard").wait_for(timeout=20000)
            overview_path = SCREENSHOTS_DIR / "dashboard_overview.png"
            page.screenshot(path=str(overview_path), full_page=True)
            results["overview_loaded"] = True
            results["screenshots"]["overview"] = str(overview_path)

            page.get_by_text("Predict", exact=True).click()
            page.get_by_text("Manual Single-Record Input").wait_for(timeout=20000)
            input_path = SCREENSHOTS_DIR / "dashboard_predict_form.png"
            page.screenshot(path=str(input_path), full_page=True)
            results["screenshots"]["predict_form"] = str(input_path)

            page.get_by_role("button", name="Run Prediction").click()
            page.get_by_text("Predicted Class").wait_for(timeout=30000)
            page.get_by_text("Per-Class Probabilities").wait_for(timeout=30000)
            result_path = SCREENSHOTS_DIR / "dashboard_prediction_result.png"
            page.screenshot(path=str(result_path), full_page=True)
            results["manual_prediction_rendered"] = True
            results["screenshots"]["prediction_result"] = str(result_path)

            page.get_by_text("CSV Batch Upload", exact=True).click()
            page.locator("input[type=file]").set_input_files(str(sample_csv))
            page.get_by_text("Preview").wait_for(timeout=20000)
            page.get_by_role("button", name="Run Batch Predictions").click()
            page.get_by_role("button", name="Download Batch Predictions").wait_for(timeout=30000)
            results["csv_with_id_prediction_rendered"] = True

            page.get_by_text("Explain", exact=True).click()
            page.get_by_text("Saved SHAP Visuals", exact=True).click()
            page.get_by_text("Saved SHAP Visuals", exact=True).wait_for(timeout=20000)
            page.get_by_text("Global SHAP summary", exact=False).wait_for(timeout=20000)
            results["explain_view_loaded"] = True
            browser.close()
    except PlaywrightTimeoutError as exc:
        raise RuntimeError(f"Browser automation timed out: {exc}") from exc
    finally:
        stop_process(dashboard_proc)
        stop_process(api_proc)

    validation_path = ARTIFACTS_DIR / "reports" / "gui_validation.json"
    validation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    results = capture_screenshots()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
