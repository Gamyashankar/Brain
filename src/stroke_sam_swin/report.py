from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .utils import ensure_dir


def build_clinical_report(
    output_path: str | Path,
    image_name: str,
    result: dict[str, Any],
    explanation_path: str | Path | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []

    story.append(Paragraph("Brain Stroke AI Diagnostic Report", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Paragraph(f"Input image: {image_name}", styles["Normal"]))
    story.append(Spacer(1, 12))

    status = result.get("status", "unknown")
    if status == "model_missing":
        summary = "Model checkpoint is not available. Train the Swin Transformer before using this report clinically."
    else:
        stroke_type = result.get("stroke_type") or result.get("prediction") or "Unknown"
        confidence_score = result.get("confidence_score")
        severity = result.get("severity") or "Unknown"
        if confidence_score is None:
            confidence_score = result.get("confidence")
            if confidence_score is not None:
                confidence_score = round(float(confidence_score) * 100, 2)
        summary = (
            f"Stroke type: {stroke_type}. Confidence: {confidence_score:.2f}%. "
            f"Severity: {severity}."
        )
    story.append(Paragraph(summary, styles["Heading2"]))
    story.append(Spacer(1, 12))

    probabilities = result.get("probabilities")
    if probabilities is not None:
        data = [["Class", "Probability"]]
        if hasattr(probabilities, "to_dict"):
            rows = probabilities.to_dict("records")
        elif isinstance(probabilities, list):
            rows = probabilities
        elif isinstance(probabilities, dict):
            rows = [{"class": key, "probability": value} for key, value in probabilities.items()]
        else:
            rows = []
        for row in rows:
            if isinstance(row, dict):
                data.append([row.get("class", ""), f"{float(row.get('probability', 0)) * 100:.2f}%"])
        table = Table(data, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 16))

    if explanation_path and Path(explanation_path).exists():
        story.append(Paragraph("Swin Visual Explanation", styles["Heading2"]))
        story.append(ReportImage(str(explanation_path), width=420, height=168))
        story.append(Spacer(1, 12))

    story.append(
        Paragraph(
            "Clinical note: This is a decision-support prototype and is not validated "
            "for independent diagnosis. Final interpretation must be made by a qualified clinician.",
            styles["Italic"],
        )
    )
    document.build(story)
    return output_path
