from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as ReportImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

from .utils import ensure_dir

NAVY = colors.HexColor("#0b2d52")
NAVY_LIGHT = colors.HexColor("#123a68")
TEAL = colors.HexColor("#00a3a1")
TEAL_DARK = colors.HexColor("#00807e")
TEAL_BRIGHT = colors.HexColor("#00c2bf")
BORDER = colors.HexColor("#d5e0eb")
MUTED = colors.HexColor("#6b7c8f")
WHITE = colors.white

SEMANTIC = {
    "normal": (colors.HexColor("#dcfce7"), colors.HexColor("#166534")),
    "ischemic": (colors.HexColor("#dbeafe"), colors.HexColor("#1e40af")),
    "hemorrhagic": (colors.HexColor("#fee2e2"), colors.HexColor("#991b1b")),
    "unknown": (colors.HexColor("#fef3c7"), colors.HexColor("#92400e")),
}


def _style(name: str, **kwargs) -> ParagraphStyle:
    base = {
        "fontName": "Helvetica",
        "fontSize": 10,
        "leading": 14,
        "textColor": colors.HexColor("#1c2b3a"),
    }
    base.update(kwargs)
    return ParagraphStyle(name, **base)


def _section_heading(text: str) -> Paragraph:
    return Paragraph(
        text,
        _style(
            "SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=10,
            spaceAfter=4,
            textColor=NAVY,
        ),
    )


def _badge(style: ParagraphStyle, text: str, fg: colors.Color, bg: colors.Color) -> Table:
    cell = Paragraph(text, style)
    t = Table([[cell]], hAlign="LEFT")
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg),
                ("BOX", (0, 0), (-1, -1), 0, colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return t


def build_clinical_report(
    output_path: str | Path,
    image_name: str,
    result: dict[str, Any],
    explanation_path: str | Path | None = None,
    patient: dict[str, Any] | None = None,
    doctor_notes: str | None = None,
    scan_date: str | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    getSampleStyleSheet()
    document = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []

    header_cell = Table(
        [
            [
                Paragraph(
                    "Neuro<font color='#00c2bf'>Scan</font>",
                    _style("Brand", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=WHITE),
                )
            ],
            [
                Paragraph(
                    "AI-Based Brain Stroke Detection & Classification System",
                    _style("BrandSub", fontName="Helvetica", fontSize=9, leading=12, textColor=WHITE),
                )
            ],
        ],
        colWidths=[160 * mm],
    )
    header_cell.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("LEFTPADDING", (0, 0), (-1, -1), 18),
            ]
        )
    )
    story.append(header_cell)

    story.append(
        Paragraph(
            f"Clinical Diagnostic Report &nbsp;|&nbsp; Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            _style("Meta", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=MUTED, spaceBefore=8),
        )
    )
    story.append(
        Paragraph(
            f"Input image: {image_name}",
            _style("Meta", fontName="Helvetica", fontSize=9, leading=12, textColor=MUTED),
        )
    )
    story.append(Spacer(1, 6))

    if patient:
        story.append(_section_heading("Patient Details"))
        patient_rows = [
            ["Patient Name", patient.get("name", "")],
            ["Patient ID", str(patient.get("id", ""))],
            ["Age", str(patient.get("age") or "")],
            ["Gender", patient.get("gender") or ""],
            ["Phone", patient.get("phone") or ""],
            ["Blood Group", patient.get("blood_group") or ""],
        ]
        if scan_date:
            patient_rows.append(["Scan Date", str(scan_date)[:10]])
        patient_table = Table(patient_rows, hAlign="LEFT", colWidths=[90, 350])
        patient_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                    ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("GRID", (0, 0), (-1, -1), 0.5, BORDER),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(patient_table)
        story.append(Spacer(1, 6))

    status = result.get("status", "unknown")
    if status == "model_missing":
        summary = "Model checkpoint is not available. Train the Swin Transformer before using this report clinically."
        key = "unknown"
        label = "Model unavailable"
    else:
        prediction = (result.get("prediction") or "Unknown").lower()
        stroke_type = (result.get("stroke_type") or prediction or "unknown").lower()
        confidence_score = result.get("confidence_score")
        severity = result.get("severity") or "Unknown"
        if confidence_score is None:
            confidence_score = result.get("confidence")
            if confidence_score is not None:
                confidence_score = round(float(confidence_score) * 100, 2)
        if prediction == "normal":
            key = "normal"
            label = "Normal"
            summary = f"No stroke detected &nbsp;|&nbsp; Confidence: {confidence_score:.2f}%"
        else:
            key = prediction if prediction in SEMANTIC else "unknown"
            label = {"ischemic": "Ischemic Stroke", "hemorrhagic": "Hemorrhagic Stroke"}.get(
                prediction, stroke_type.title() if stroke_type != "normal" else "Stroke"
            )
            summary = (
                f"{label} detected &nbsp;|&nbsp; Confidence: {confidence_score:.2f}% "
                f"&nbsp;|&nbsp; Severity: {severity}"
            )

    story.append(_section_heading("AI Prediction"))
    fg, bg = SEMANTIC.get(key, SEMANTIC["unknown"])
    label_style = _style("Badge", fontName="Helvetica-Bold", fontSize=11, leading=14, textColor=fg)
    pred_table = Table(
        [[_badge(label_style, label.upper(), fg, bg)], [Paragraph(summary, _style("Summary", fontSize=10, leading=15))]],
        hAlign="LEFT",
    )
    pred_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("BOX", (0, 0), (-1, -1), 1, BORDER),
                ("TOPPADDING", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(pred_table)
    story.append(Spacer(1, 8))

    lesion_region = result.get("affected_region")
    lesion_overlay = result.get("lesion_overlay")
    if lesion_region or result.get("lesion_area") is not None or lesion_overlay:
        story.append(_section_heading("Lesion Localization"))
        lesion_rows: list[list[str]] = [["Affected Region", lesion_region or "Not localized"]]
        if result.get("lesion_area") is not None:
            lesion_rows.append(["Lesion Area", f'{float(result["lesion_area"]):.1f} mm\u00b2'])
        if result.get("lesion_pct") is not None:
            lesion_rows.append(["Brain Affected", f'{float(result["lesion_pct"]):.2f}%'])
        if result.get("lesion_centroid"):
            lesion_rows.append(["Lesion Centroid (X, Y)", result["lesion_centroid"]])
        lesion_table = Table(lesion_rows, hAlign="LEFT", colWidths=[150, 290])
        lesion_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#fef2f2")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#991b1b")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#fecaca")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(lesion_table)
        if lesion_overlay and Path(lesion_overlay).exists():
            try:
                with PILImage.open(str(lesion_overlay)) as image:
                    width_px, height_px = image.size
                scale = min(1.0, 300.0 / float(width_px))
                story.append(
                    ReportImage(
                        str(lesion_overlay),
                        width=float(width_px) * scale,
                        height=float(height_px) * scale,
                    )
                )
            except Exception:
                story.append(ReportImage(str(lesion_overlay), width=280, height=280))
        story.append(Spacer(1, 6))

    if explanation_path and Path(explanation_path).exists():
        story.append(_section_heading("Explainable AI"))
        try:
            with PILImage.open(str(explanation_path)) as image:
                width_px, height_px = image.size
            scale = min(1.0, 440.0 / float(width_px))
            story.append(
                ReportImage(
                    str(explanation_path),
                    width=float(width_px) * scale,
                    height=float(height_px) * scale,
                )
            )
        except Exception:
            story.append(ReportImage(str(explanation_path), width=420, height=168))
        story.append(Spacer(1, 6))

    if doctor_notes:
        story.append(_section_heading("Doctor Notes"))
        notes_table = Table([[Paragraph(doctor_notes, _style("Notes", fontSize=10, leading=15))]], colWidths=[440])
        notes_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef9c3")),
                    ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#fde68a")),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(notes_table)
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Clinical note: This is a decision-support prototype and is not validated for independent "
            "diagnosis. Final interpretation must be made by a qualified clinician.",
            _style("Disclaimer", fontName="Helvetica-Oblique", fontSize=8.5, leading=12, textColor=MUTED),
        )
    )
    document.build(story)
    return output_path
