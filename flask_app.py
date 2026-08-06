from __future__ import annotations

import io
import os
import tempfile
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from flask import Flask, flash, redirect, render_template, request, send_file, session, url_for
from PIL import Image
from werkzeug.security import check_password_hash

from database import (
    create_patient,
    create_scan,
    delete_patient,
    get_all_scans,
    get_patient,
    get_patient_by_user_id,
    get_patients,
    get_scan,
    get_scans_by_patient,
    get_user_by_username,
    init_db,
    scan_counts,
    update_patient,
    update_scan_notes,
)
from stroke_sam_swin.inference import analyze_scan
from stroke_sam_swin.report import build_clinical_report
from stroke_sam_swin.utils import ensure_dir, load_config

init_db()

app = Flask(__name__, static_folder="outputs/uploads", static_url_path="/uploads")
app.secret_key = "brain-stroke-ai-demo"
app.config["UPLOAD_FOLDER"] = "outputs/uploads"

CONFIG_PATH = Path("configs/default.yaml")
CHECKPOINT_PATH = Path("outputs/checkpoints/best_model.pt")
REPORT_DIR = Path("outputs/reports")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def doctor_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        if session.get("role") != "doctor":
            flash("Access denied. Doctor account required.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

    return wrapped


def patient_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        if session.get("role") != "patient":
            flash("Access denied. Patient account required.")
            return redirect(url_for("login"))
        return view(*args, **kwargs)

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
        import shutil

        shutil.copyfile(source, destination)
    return str(destination)


@app.route("/")
def index():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    return render_template("landing.html")


@app.template_filter("basename")
def basename_filter(path: str | None) -> str:
    import os

    return os.path.basename(path or "")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("dashboard"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        role = request.form.get("role", "doctor")
        user = get_user_by_username(username)
        if user and user["role"] == role and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["role"] = user["role"]
            return redirect(url_for("dashboard"))
        error = "Invalid username, password, or role."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    if session["role"] == "doctor":
        return redirect(url_for("doctor_dashboard"))
    return redirect(url_for("patient_dashboard"))


def current_user() -> dict[str, Any] | None:
    from database import get_user

    return get_user(session.get("user_id"))


def summarize_inference(inference_result: dict[str, Any], image: Image.Image | None = None) -> dict[str, Any]:
    from app import build_prediction_summary

    if inference_result.get("status") == "model_missing":
        return {
            "prediction": "unknown",
            "stroke_type": "Unknown",
            "confidence_score": 0.0,
            "severity": "Unknown",
            "stroke_score": 0.0,
            "probabilities": [],
        }
    prediction = str(inference_result.get("prediction", "")).strip().lower()
    probabilities = inference_result.get("probabilities")
    prob_rows = []
    if hasattr(probabilities, "to_dict"):
        prob_rows = probabilities.to_dict("records")
    elif isinstance(probabilities, list):
        prob_rows = probabilities
    summary = build_prediction_summary(prediction, inference_result.get("confidence", 0.0), image, prob_rows)
    return {
        "prediction": prediction,
        "stroke_type": summary["stroke_type"],
        "confidence_score": summary["confidence_score"],
        "severity": summary["severity"],
        "stroke_score": summary["stroke_score"],
        "probabilities": prob_rows,
        "lesion": inference_result.get("lesion"),
    }


def save_scan_record(patient_id: int, summary: dict[str, Any], inference_result: dict[str, Any], filename: str) -> int:
    explanation_path = inference_result.get("explanation_path")
    copied = copy_to_uploads(explanation_path) if explanation_path else None
    lesion = summary.get("lesion") or {}
    lesion_overlay = None
    if lesion.get("overlay_path"):
        lesion_overlay = copy_to_uploads(lesion["overlay_path"])
    return create_scan(
        patient_id=patient_id,
        filename=filename,
        prediction=summary["prediction"],
        confidence=summary["confidence_score"],
        severity=summary["severity"],
        stroke_score=summary["stroke_score"],
        probabilities=summary["probabilities"],
        explanation_path=copied,
        affected_region=lesion.get("region"),
        lesion_area=lesion.get("area_mm2"),
        lesion_pct=lesion.get("pct_brain"),
        lesion_centroid=lesion.get("centroid"),
        lesion_overlay=lesion_overlay,
    )


# ---------------------------------------------------------------------------
# Doctor routes
# ---------------------------------------------------------------------------


@app.route("/doctor/dashboard")
@doctor_required
def doctor_dashboard():
    stats = scan_counts()
    recent_patients = get_patients()[:8]
    return render_template("doctor_dashboard.html", stats=stats, recent_patients=recent_patients)


@app.route("/doctor/patients")
@doctor_required
def doctor_patients():
    query = request.args.get("q", "").strip()
    patients = get_patients(query)
    return render_template("patients.html", patients=patients, query=query)


@app.route("/doctor/patients/new", methods=["GET", "POST"])
@doctor_required
def patient_new():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        blood_group = request.form.get("blood_group", "").strip()
        medical_history = request.form.get("medical_history", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        if not name or not username or not password:
            flash("Name, username, and password are required.")
            return render_template("patient_form.html", patient=None)
        if get_user_by_username(username):
            flash("Username already exists. Choose another.")
            return render_template("patient_form.html", patient=None)

        patient_id = create_patient(
            name=name,
            age=int(age) if age.isdigit() else None,
            gender=gender,
            phone=phone,
            blood_group=blood_group,
            medical_history=medical_history,
            username=username,
            password=password,
        )
        flash(f"Patient registered successfully (ID: {patient_id}).")
        return redirect(url_for("doctor_patients"))
    return render_template("patient_form.html", patient=None)


@app.route("/doctor/patients/<int:patient_id>")
@doctor_required
def patient_detail(patient_id: int):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.")
        return redirect(url_for("doctor_patients"))
    scans = get_scans_by_patient(patient_id)
    return render_template("patient_detail.html", patient=patient, scans=scans)


@app.route("/doctor/patients/<int:patient_id>/edit", methods=["GET", "POST"])
@doctor_required
def patient_edit(patient_id: int):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.")
        return redirect(url_for("doctor_patients"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        gender = request.form.get("gender", "").strip()
        phone = request.form.get("phone", "").strip()
        blood_group = request.form.get("blood_group", "").strip()
        medical_history = request.form.get("medical_history", "").strip()
        if not name:
            flash("Name is required.")
            return render_template("patient_form.html", patient=patient)
        update_patient(
            patient_id,
            name=name,
            age=int(age) if age.isdigit() else None,
            gender=gender,
            phone=phone,
            blood_group=blood_group,
            medical_history=medical_history,
        )
        flash("Patient updated.")
        return redirect(url_for("doctor_patients"))
    return render_template("patient_form.html", patient=patient)


@app.route("/doctor/patients/<int:patient_id>/delete", methods=["POST"])
@doctor_required
def patient_delete(patient_id: int):
    delete_patient(patient_id)
    flash("Patient deleted.")
    return redirect(url_for("doctor_patients"))


@app.route("/doctor/patients/<int:patient_id>/upload", methods=["GET", "POST"])
@doctor_required
def patient_upload(patient_id: int):
    patient = get_patient(patient_id)
    if not patient:
        flash("Patient not found.")
        return redirect(url_for("doctor_patients"))
    if request.method == "POST":
        file = request.files.get("file")
        if not file or not file.filename:
            flash("Please choose an image to upload.")
            return render_template("upload.html", patient=patient)
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

        summary = summarize_inference(inference_result, image)

        save_dir = ensure_dir(app.config["UPLOAD_FOLDER"])
        input_name = f"p{patient_id}_{Path(file.filename).stem}_input.png"
        image.save(Path(save_dir) / input_name)

        scan_id = save_scan_record(patient_id, summary, inference_result, file.filename)
        return redirect(url_for("doctor_report", scan_id=scan_id))

    return render_template("upload.html", patient=patient)


@app.route("/doctor/reports/<int:scan_id>", methods=["GET", "POST"])
@doctor_required
def doctor_report(scan_id: int):
    scan = get_scan(scan_id)
    if not scan:
        flash("Report not found.")
        return redirect(url_for("doctor_patients"))
    patient = get_patient(scan["patient_id"])
    if request.method == "POST":
        notes = request.form.get("doctor_notes", "").strip()
        update_scan_notes(scan_id, notes)
        scan["doctor_notes"] = notes
        flash("Doctor notes saved.")
    return render_template("report.html", scan=scan, patient=patient, role="doctor")


@app.route("/doctor/statistics")
@doctor_required
def doctor_statistics():
    stats = scan_counts()
    scans = get_all_scans()
    monthly: dict[str, int] = {}
    for scan in scans:
        month = scan["created_at"][:7]
        monthly[month] = monthly.get(month, 0) + 1
    months = sorted(monthly.keys())
    return render_template(
        "statistics.html",
        stats=stats,
        months=months,
        monthly_counts=[monthly[m] for m in months],
        recent_scans=scans[:10],
    )


# ---------------------------------------------------------------------------
# Patient routes
# ---------------------------------------------------------------------------


@app.route("/patient/dashboard")
@patient_required
def patient_dashboard():
    patient = get_patient_by_user_id(session["user_id"])
    if not patient:
        flash("Patient profile not found.")
        return redirect(url_for("logout"))
    scans = get_scans_by_patient(patient["id"])
    return render_template("patient_dashboard.html", patient=patient, scans=scans)


@app.route("/patient/reports/<int:scan_id>")
@patient_required
def patient_report(scan_id: int):
    scan = get_scan(scan_id)
    patient = get_patient_by_user_id(session["user_id"])
    if not scan or not patient or scan["patient_id"] != patient["id"]:
        flash("Report not found.")
        return redirect(url_for("patient_dashboard"))
    return render_template("report.html", scan=scan, patient=patient, role="patient")


# ---------------------------------------------------------------------------
# Report download
# ---------------------------------------------------------------------------


@app.route("/reports/<int:scan_id>/download")
@login_required
def download_scan_report(scan_id: int):
    scan = get_scan(scan_id)
    if not scan:
        flash("Report not found.")
        return redirect(url_for("dashboard"))
    patient = get_patient(scan["patient_id"])
    if session["role"] == "patient" and (not patient or patient["user_id"] != session["user_id"]):
        flash("Access denied.")
        return redirect(url_for("patient_dashboard"))

    report_payload = {
        "status": "ok",
        "prediction": scan.get("prediction", "unknown"),
        "confidence_score": scan.get("confidence", 0.0),
        "severity": scan.get("severity", "Unknown"),
        "stroke_score": scan.get("stroke_score", 0.0),
        "probabilities": scan.get("probabilities") or [],
        "affected_region": scan.get("affected_region"),
        "lesion_area": scan.get("lesion_area"),
        "lesion_pct": scan.get("lesion_pct"),
        "lesion_centroid": scan.get("lesion_centroid"),
        "lesion_overlay": scan.get("lesion_overlay"),
    }
    report_path = REPORT_DIR / f"report_{scan_id}.pdf"
    build_clinical_report(
        output_path=report_path,
        image_name=scan.get("filename", "scan"),
        result=report_payload,
        explanation_path=scan.get("explanation_path"),
        patient=patient,
        doctor_notes=scan.get("doctor_notes"),
        scan_date=scan.get("created_at"),
    )
    if not report_path.exists():
        flash("Could not generate the PDF report.")
        return redirect(url_for("dashboard"))
    return send_file(report_path, as_attachment=True, download_name=f"stroke_report_{scan_id}.pdf")


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8003)
