from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, UploadFile
from PIL import Image, ImageStat
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


def build_prediction_summary(
    prediction: str,
    confidence: float,
    image: Image.Image | None = None,
    probabilities: list[dict[str, Any]] | Any | None = None,
) -> dict[str, Any]:
    prediction = str(prediction or "").strip().lower()
    confidence_score = round(float(confidence) * 100, 2)

    prob_by_class: dict[str, float] = {}
    if probabilities is not None:
        rows = probabilities.to_dict("records") if hasattr(probabilities, "to_dict") else (probabilities if isinstance(probabilities, list) else [])
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("class", "")).strip().lower()
            try:
                value = float(row.get("probability", 0.0))
            except (TypeError, ValueError):
                continue
            if value > 1.0:
                value = value / 100.0
            prob_by_class[name] = value

    if prediction in {"hemorrhagic", "hemorrhage", "bleed"}:
        stroke_type = "Hemorrhagic Stroke"
    elif prediction in {"ischemic", "ischemia", "ischaemic"}:
        stroke_type = "Ischemic Stroke"
    elif prediction in {"normal", "none", "no stroke", "no_stroke"}:
        stroke_type = "No Stroke Detected"
    elif prediction == "stroke":
        if image is None:
            stroke_type = "Stroke Detected"
        else:
            grayscale = image.convert("L")
            mean_intensity = ImageStat.Stat(grayscale).mean[0]
            stroke_type = "Hemorrhagic Stroke" if mean_intensity > 140 else "Ischemic Stroke"
    else:
        stroke_type = prediction.replace("_", " ").title() if prediction else "Stroke Detected"

    stroke_probs = [value for name, value in prob_by_class.items() if name in {"ischemic", "ischemia", "ischaemic", "hemorrhagic", "hemorrhage", "bleed", "stroke"}]
    if stroke_probs:
        stroke_score = round(min(100.0, sum(stroke_probs) * 100.0), 2)
    elif stroke_type == "No Stroke Detected":
        stroke_score = round(max(0.0, 100.0 - confidence_score), 2)
    else:
        stroke_score = confidence_score

    if stroke_type == "No Stroke Detected":
        severity = "None"
    elif stroke_score >= 80:
        severity = "Severe"
    elif stroke_score >= 50:
        severity = "Moderate"
    else:
        severity = "Mild"

    return {
        "stroke_type": stroke_type,
        "confidence_score": confidence_score,
        "severity": severity,
        "stroke_score": stroke_score,
        "probabilities": [],
    }

from flask_app import app as flask_app_instance
from stroke_sam_swin.inference import analyze_scan, checkpoint_ready
from stroke_sam_swin.report import build_clinical_report
from stroke_sam_swin.utils import ensure_dir, load_config

CONFIG_PATH = Path("configs/default.yaml")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")
REPORT_DIR = Path("outputs/reports")

app = FastAPI(title="Brain Stroke AI", version="1.0.0")


def infer_stroke_type(prediction: str, image: Image.Image | None = None) -> str:
    return build_prediction_summary(prediction, 1.0, image)["stroke_type"]


app.mount("/static", StaticFiles(directory="static"), name="static")

frontend_dist = Path("frontend/dist")
frontend_assets = frontend_dist / "assets"
if frontend_assets.exists():
    app.mount("/assets", StaticFiles(directory=frontend_assets), name="frontend-assets")


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    dist_index = frontend_dist / "index.html"
    if dist_index.exists():
        return dist_index.read_text(encoding="utf-8")
    html_path = Path("templates/index.html")
    return html_path.read_text(encoding="utf-8")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    config = load_config(CONFIG_PATH)

    if not file.filename:
        return JSONResponse(status_code=400, content={"error": "No file uploaded."})

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
        image.save(temp_file.name)
        temp_path = Path(temp_file.name)

    try:
        result = analyze_scan(
            config_path=CONFIG_PATH,
            checkpoint_path=CHECKPOINT_PATH,
            image_path=temp_path,
            generate_explanation=False,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    if result["status"] == "model_missing":
        payload: dict[str, Any] = {
            "status": "Training required",
            "stroke_type": "Unknown",
            "confidence_score": 0.0,
            "severity": "Unknown",
            "stroke_score": 0.0,
            "probabilities": [{"class": "No Stroke", "probability": 0.0}],
            "error": "The model checkpoint is not available yet. Train the model first.",
        }
    else:
        prediction = str(result.get("prediction", "")).strip().lower()
        probabilities = result.get("probabilities")
        prob_rows = []
        if hasattr(probabilities, "to_dict"):
            prob_rows = probabilities.to_dict("records")
        elif isinstance(probabilities, list):
            prob_rows = probabilities
        summary = build_prediction_summary(prediction, result.get("confidence", 0.0), image, prob_rows)
        lesion = result.get("lesion") or {}
        payload = {
            "status": "Prediction ready",
            "stroke_type": summary["stroke_type"],
            "confidence_score": summary["confidence_score"],
            "severity": summary["severity"],
            "stroke_score": summary["stroke_score"],
            "probabilities": summary["probabilities"],
            "lesion": lesion,
        }

    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode("ascii")
    payload["image_url"] = f"data:image/png;base64,{img_b64}"

    report_dir = ensure_dir(config["outputs"]["reports"])
    report_path = Path(report_dir) / "latest_clinical_report.pdf"
    report_result = dict(result)
    if result["status"] != "model_missing":
        report_result.update(
            {
                "stroke_type": payload["stroke_type"],
                "severity": payload["severity"],
            }
        )
    report_result["explanation_path"] = result.get("explanation_path")
    lesion = result.get("lesion") or {}
    report_result.update(
        {
            "affected_region": lesion.get("region"),
            "lesion_area": lesion.get("area_mm2"),
            "lesion_pct": lesion.get("pct_brain"),
            "lesion_centroid": lesion.get("centroid"),
            "lesion_overlay": lesion.get("overlay_path"),
        }
    )
    build_clinical_report(
        output_path=report_path,
        image_name=file.filename,
        result=report_result,
        explanation_path=result.get("explanation_path"),
    )
    payload["report_url"] = f"/reports/{report_path.name}"

    return JSONResponse(content=payload)


@app.get("/reports/{report_name}")
def download_report(report_name: str) -> FileResponse:
    report_path = REPORT_DIR / report_name
    if not report_path.exists():
        return JSONResponse(status_code=404, content={"error": "Report not found"})
    return FileResponse(path=report_path, filename=report_name, media_type="application/pdf")


if __name__ == "__main__":
    import os
    import socket

    def get_free_port(start_port: int = 8003) -> int:
        port = start_port
        while True:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                try:
                    sock.bind(("127.0.0.1", port))
                    return port
                except OSError:
                    port += 1

    port = int(os.environ.get("PORT", get_free_port()))
    print(f"Starting the updated Brain Stroke UI on http://127.0.0.1:{port}/")
    flask_app_instance.run(debug=True, host="127.0.0.1", port=port)
