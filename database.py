from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from werkzeug.security import generate_password_hash

DB_PATH = Path(__file__).resolve().parent / "brain_stroke.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('doctor', 'patient')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    phone TEXT,
    blood_group TEXT,
    medical_history TEXT,
    emergency_contact TEXT,
    emergency_phone TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    filename TEXT,
    prediction TEXT,
    confidence REAL,
    severity TEXT,
    stroke_score REAL,
    probabilities TEXT,
    explanation_path TEXT,
    affected_region TEXT,
    lesion_area REAL,
    lesion_pct REAL,
    lesion_centroid TEXT,
    lesion_overlay TEXT,
    report_path TEXT,
    doctor_notes TEXT,
    created_at TEXT NOT NULL
);
"""

SCAN_COLUMN_MIGRATIONS = {
    "affected_region": "ALTER TABLE scans ADD COLUMN affected_region TEXT",
    "lesion_area": "ALTER TABLE scans ADD COLUMN lesion_area REAL",
    "lesion_pct": "ALTER TABLE scans ADD COLUMN lesion_pct REAL",
    "lesion_centroid": "ALTER TABLE scans ADD COLUMN lesion_centroid TEXT",
    "lesion_overlay": "ALTER TABLE scans ADD COLUMN lesion_overlay TEXT",
    "emergency_contact": "ALTER TABLE patients ADD COLUMN emergency_contact TEXT",
    "emergency_phone": "ALTER TABLE patients ADD COLUMN emergency_phone TEXT",
}


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)
        existing_scans = {row[1] for row in conn.execute("PRAGMA table_info(scans)").fetchall()}
        existing_patients = {row[1] for row in conn.execute("PRAGMA table_info(patients)").fetchall()}
        for column, ddl in SCAN_COLUMN_MIGRATIONS.items():
            if column in existing_scans or column in existing_patients:
                continue
            if "scans" in ddl and column not in existing_scans:
                conn.execute(ddl)
            elif "patients" in ddl and column not in existing_patients:
                conn.execute(ddl)
        admin = conn.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
        if admin is None:
            conn.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
                ("admin", generate_password_hash("admin123"), "doctor", datetime.now().isoformat(timespec="seconds")),
            )


def create_user(username: str, password: str, role: str) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), role, datetime.now().isoformat(timespec="seconds")),
        )
        return int(cursor.lastrowid)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        return dict(row) if row else None


def get_user(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def create_patient(
    name: str,
    age: int | None,
    gender: str,
    phone: str,
    blood_group: str,
    medical_history: str,
    username: str,
    password: str,
    emergency_contact: str = "",
    emergency_phone: str = "",
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, generate_password_hash(password), "patient", datetime.now().isoformat(timespec="seconds")),
        )
        user_id = int(cursor.lastrowid)
        patient_cursor = conn.execute(
            "INSERT INTO patients (user_id, name, age, gender, phone, blood_group, medical_history, emergency_contact, emergency_phone, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                user_id,
                name,
                age,
                gender,
                phone,
                blood_group,
                medical_history,
                emergency_contact,
                emergency_phone,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(patient_cursor.lastrowid)


def get_patients(search: str | None = None) -> list[dict[str, Any]]:
    query = (
        "SELECT p.*, u.username FROM patients p LEFT JOIN users u ON p.user_id = u.id "
    )
    params: tuple[Any, ...] = ()
    if search:
        term = f"%{search}%"
        query += "WHERE p.name LIKE ? OR p.phone LIKE ? OR CAST(p.id AS TEXT) LIKE ?"
        params = (term, term, term)
    query += " ORDER BY p.id DESC"
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def get_patient(patient_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT p.*, u.username FROM patients p LEFT JOIN users u ON p.user_id = u.id WHERE p.id = ?",
            (patient_id,),
        ).fetchone()
        return dict(row) if row else None


def get_patient_by_user_id(user_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM patients WHERE user_id = ?", (user_id,)).fetchone()
        return dict(row) if row else None


def update_patient(patient_id: int, name: str, age: int | None, gender: str, phone: str, blood_group: str, medical_history: str, emergency_contact: str = "", emergency_phone: str = "") -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE patients SET name = ?, age = ?, gender = ?, phone = ?, blood_group = ?, medical_history = ?, emergency_contact = ?, emergency_phone = ? "
            "WHERE id = ?",
            (name, age, gender, phone, blood_group, medical_history, emergency_contact, emergency_phone, patient_id),
        )


def delete_patient(patient_id: int) -> None:
    with get_connection() as conn:
        row = conn.execute("SELECT user_id FROM patients WHERE id = ?", (patient_id,)).fetchone()
        if row and row["user_id"]:
            conn.execute("DELETE FROM users WHERE id = ?", (row["user_id"],))
        conn.execute("DELETE FROM patients WHERE id = ?", (patient_id,))


def create_scan(patient_id: int, **fields: Any) -> int:
    filename = fields.get("filename")
    prediction = fields.get("prediction")
    confidence = fields.get("confidence")
    severity = fields.get("severity")
    stroke_score = fields.get("stroke_score")
    probabilities = json.dumps(fields.get("probabilities") or [])
    explanation_path = fields.get("explanation_path")
    affected_region = fields.get("affected_region")
    lesion_area = fields.get("lesion_area")
    lesion_pct = fields.get("lesion_pct")
    lesion_centroid = fields.get("lesion_centroid")
    lesion_overlay = fields.get("lesion_overlay")
    report_path = fields.get("report_path")
    doctor_notes = fields.get("doctor_notes")
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO scans (patient_id, filename, prediction, confidence, severity, stroke_score, "
            "probabilities, explanation_path, affected_region, lesion_area, lesion_pct, lesion_centroid, "
            "lesion_overlay, report_path, doctor_notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                patient_id,
                filename,
                prediction,
                confidence,
                severity,
                stroke_score,
                probabilities,
                explanation_path,
                affected_region,
                lesion_area,
                lesion_pct,
                lesion_centroid,
                lesion_overlay,
                report_path,
                doctor_notes,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cursor.lastrowid)


def get_scan(scan_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["probabilities"] = json.loads(result.get("probabilities") or "[]")
        return result


def get_scans_by_patient(patient_id: int) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM scans WHERE patient_id = ? ORDER BY id DESC", (patient_id,)).fetchall()
        return [dict(row) for row in rows]


def get_all_scans() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM scans ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]


def update_scan_notes(scan_id: int, notes: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE scans SET doctor_notes = ? WHERE id = ?", (notes, scan_id))


def scan_counts() -> dict[str, int]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
        patients = conn.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
        ischemic = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction LIKE '%ischemic%'").fetchone()[0]
        hemorrhagic = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction LIKE '%hemorrhagic%'").fetchone()[0]
        normal = conn.execute("SELECT COUNT(*) FROM scans WHERE prediction LIKE '%normal%'").fetchone()[0]
        today = datetime.now().strftime("%Y-%m-%d")
        todays = conn.execute("SELECT COUNT(*) FROM scans WHERE created_at LIKE ?", (f"{today}%",)).fetchone()[0]
        return {
            "total_patients": patients,
            "total_scans": total,
            "ischemic": ischemic,
            "hemorrhagic": hemorrhagic,
            "normal": normal,
            "stroke_cases": ischemic + hemorrhagic,
            "todays_scans": todays,
        }
