from __future__ import annotations

import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, redirect, render_template, request, session, send_file, url_for
from PIL import Image
import pdfkit

from stroke_sam_swin.inference import analyze_scan, checkpoint_ready
from stroke_sam_swin.report import build_clinical_report
from stroke_sam_swin.utils import ensure_dir, load_config

app = Flask(__name__, static_folder="outputs/uploads", static_url_path="/uploads")
app.secret_key = "brain-stroke-ai-demo"
app.config["UPLOAD_FOLDER"] = "outputs/uploads"

CONFIG_PATH = Path("configs/default.yaml")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")
REPORT_DIR = Path("outputs/reports")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def login_required(view):
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    wrapped.__name__ = view.__name__
    return wrapped


def copy_to_uploads(source_path: str | Path | None) -> str | None:
    if not source_path:
        return None
    source = Path(source_path)
    if not source.exists():
        return None
    destination_dir = ensure_dir(app.config["UPLOAD_FOLDER"])
    destination = Path(destination_dir) / source.name
    if not destination.exists():
        shutil.copyfile(source, destination)
    return str(destination)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == "admin" and password == "admin123":
            session["logged_in"] = True
            return redirect(url_for("home"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route("/home")
@login_required
def home():
    return render_template("home.html")


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    result = None
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename:
            image = Image.open(file.stream).convert("RGB")
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                image.save(temp_file.name)
                temp_path = Path(temp_file.name)
            try:
                inference_result = analyze_scan(
                    config_path=CONFIG_PATH,
                    checkpoint_path=CHECKPOINT_PATH,
                    image_path=temp_path,
                    generate_explanation=True,
                )
            finally:
                temp_path.unlink(missing_ok=True)

            prediction = str(inference_result.get("prediction", "")).strip().lower()
            summary = {
                "stroke_type": "Unknown",
                "confidence_score": 0.0,
                "severity": "Unknown",
                "stroke_score": 0.0,
                "probabilities": [],
            }
            if inference_result.get("status") == "model_missing":
                summary.update({"stroke_type": "Unknown", "severity": "Unknown"})
            else:
                from app import build_prediction_summary
                summary = build_prediction_summary(prediction, inference_result.get("confidence", 0.0), image)

            result = {
                "prediction": summary["stroke_type"],
                "prediction_class": prediction,
                "confidence": summary["confidence_score"],
                "stroke_score": summary["stroke_score"],
                "stroke_type": summary["stroke_type"],
                "severity": summary["severity"],
                "probabilities": summary["probabilities"],
            }

            save_dir = ensure_dir(app.config["UPLOAD_FOLDER"])
            input_path = Path(save_dir) / f"{Path(file.filename).stem}_input.png"
            image.save(input_path)
            result["input_image"] = f"/uploads/{input_path.name}"
            result["heatmap_image"] = result["input_image"]
            result["overlay_image"] = result["input_image"]

            explanation_artifacts = inference_result.get("explanation_artifacts") or {}
            if explanation_artifacts.get("heatmap"):
                copied_heatmap = copy_to_uploads(explanation_artifacts["heatmap"])
                if copied_heatmap:
                    result["heatmap_image"] = f"/uploads/{Path(copied_heatmap).name}"
            if explanation_artifacts.get("overlay"):
                copied_overlay = copy_to_uploads(explanation_artifacts["overlay"])
                if copied_overlay:
                    result["overlay_image"] = f"/uploads/{Path(copied_overlay).name}"
            result["explanation_path"] = inference_result.get("explanation_path")

            session["last_result"] = result
            session["last_filename"] = file.filename

    return render_template("analyze.html", result=result)


@app.route("/download_pdf")
@login_required
def download_pdf():
    result = session.get("last_result")
    if not result:
        return redirect(url_for("analyze"))

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = session.get("last_filename", "scan")
    report_payload = {
        "status": "ok",
        "prediction": result.get("prediction", "stroke"),
        "confidence": result.get("confidence", 0.0) / 100.0,
        "probabilities": [
            {"class": key, "probability": value / 100.0} for key, value in result.get("probabilities", {}).items()
        ],
    }
    report_path = REPORT_DIR / "brain_stroke_report.pdf"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    build_clinical_report(
        output_path=report_path,
        image_name=filename,
        result=report_payload,
        explanation_path=result.get("explanation_path"),
    )
    if not report_path.exists():
        return redirect(url_for("analyze"))
    return send_file(report_path, as_attachment=True, download_name="brain_stroke_report.pdf")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8003)
