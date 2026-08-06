# AI-Based Brain Stroke Detection and Classification System using SAM and Swin Transformer with Explainable AI

**Project Report**

---

## Table of Contents

1. [Introduction](#1-introduction)
   - [1.1 Problem Statement](#11-problem-statement)
   - [1.2 Background of the Problem](#12-background-of-the-problem)
   - [1.3 Existing Issues / Challenges](#13-existing-issues--challenges)
   - [1.4 Need for the Proposed System (with Figure)](#14-need-for-the-proposed-system-with-figure)
2. [Objectives](#2-objectives)
   - [2.1 Primary Objective](#21-primary-objective)
   - [2.2 Secondary Objectives](#22-secondary-objectives)
   - [2.3 Expected Outcomes](#23-expected-outcomes)
3. [Scope of the Project](#3-scope-of-the-project)
   - [3.1 What Is Included](#31-what-is-included)
   - [3.2 What Is Excluded](#32-what-is-excluded)
   - [3.3 Target Users / Domain](#33-target-users--domain)
4. [System Architecture](#4-system-architecture)
   - [4.1 Architecture Diagram](#41-architecture-diagram)
   - [4.2 Major Modules](#42-major-modules)
   - [4.3 Data Flow Between Modules](#43-data-flow-between-modules)
   - [4.4 Technology Stack](#44-technology-stack)
5. [Detailed Module Design](#5-detailed-module-design)
   - [5.1 Data Ingestion and Preprocessing](#51-data-ingestion-and-preprocessing-module)
   - [5.2 SAM ROI Segmentation](#52-sam-roi-segmentation-module)
   - [5.3 Model Definition and Training](#53-model-definition-and-training-module)
   - [5.4 Inference and Classification](#54-inference-and-classification-module)
   - [5.5 Explainable AI (XAI)](#55-explainable-ai-xai-module)
   - [5.6 Clinical Report Generation](#56-clinical-report-generation-module)
   - [5.7 Web Application](#57-web-application-module)
   - [5.8 Database](#58-database-module)
   - [5.9 API Service](#59-api-service-module)
6. [Implementation Progress Status](#6-implementation-progress-status)
   - [6.1 Work Completed](#61-work-completed)
   - [6.2 Remaining Work](#62-remaining-work)
7. [Project Completion Plan](#7-project-completion-plan)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

Stroke is a neurological emergency caused by an interruption of blood supply to the brain. It is the **second leading cause of death** worldwide and a leading cause of long-term adult disability. In India, approximately **1.8 million people are affected by stroke every year**, and the burden continues to grow with an ageing population and rising prevalence of hypertension, diabetes, and sedentary lifestyles.

The clinical outcome of a stroke depends critically on **time to diagnosis and treatment**. Two major stroke subtypes — *ischemic* (blocked vessel, ~85% of cases) and *hemorrhagic* (ruptured vessel) — require **opposite** treatments: clot-dissolving therapy for ischemic stroke versus blood-pressure control and possible surgical intervention for hemorrhagic stroke. Misclassification between the two can be fatal. Computed Tomography (CT) is the first-line imaging modality because it quickly distinguishes hemorrhage from ischemia; Magnetic Resonance Imaging (MRI) provides more detailed tissue characterisation.

This project builds a **web-based clinical decision-support system** that uses deep learning to detect and classify brain strokes from CT/MRI scans, and — critically — explains *why* the model made its decision using multiple Explainable AI (XAI) techniques, so clinicians can trust and verify the output.

### 1.1 Problem Statement

Manual stroke interpretation from CT/MRI images is **time-sensitive, reader-dependent, and requires expert radiologists** who are scarce in rural and low-resource settings. Delays in reading scans and the visual subtlety of early ischemic lesions lead to missed or delayed diagnosis. There is a need for an automated assistant that:

- detects whether a scan shows a stroke,
- classifies the stroke type (ischemic vs. hemorrhagic) — the decision that drives treatment,
- localises the affected region, and
- produces a clinician-readable report, all within a hospital-friendly workflow.

### 1.2 Background of the Problem

- **Global burden:** Stroke accounts for ~11–12% of all deaths worldwide. Each year millions of new cases occur, and survivors often face severe disability.
- **The "golden window":** For ischemic stroke, thrombolysis must be administered within ~4.5 hours of onset. Diagnostic delay directly worsens outcomes.
- **Subtype-dependent treatment:** Administering anti-clotting drugs to a hemorrhagic patient, or vice-versa, is dangerous. Rapid, correct subtyping is therefore a clinical safety requirement, not just a convenience.
- **Imaging:** CT is fast, cheap, and available in most hospitals; MRI offers higher soft-tissue contrast. Both are noisy, high-dimensional, and interpretable mainly by trained specialists.
- **AI in radiology:** Deep learning models (especially convolutional and transformer networks) have shown expert-level performance in lesion detection, but their *opacity* limits clinical adoption. Explainable AI is the bridge that lets clinicians inspect model reasoning.

### 1.3 Existing Issues / Challenges

1. **Radiologist shortage and workload:** In many regions, there are no specialists available around the clock, causing treatment delays for stroke patients.
2. **Manual interpretation is subjective:** Detection of subtle ischemic changes can vary between readers and is prone to error under time pressure.
3. **Black-box AI:** Existing AI classifiers often return only a label and confidence with no justification, which is unacceptable in a medical setting where errors have life-or-death consequences.
4. **Missing end-to-end clinical workflow:** Most research systems stop at a model and accuracy number; they do not integrate patient records, report generation, or a usable interface for doctors.
5. **Imbalanced and limited datasets:** Stroke lesion datasets are often small and heavily skewed towards normal scans, which biases naive classifiers.
6. **No localisation/explanation:** Many systems classify but do not show *where* in the brain the evidence lies, limiting their usefulness and trustworthiness.

### 1.4 Need for the Proposed System (with Figure)

The proposed system directly addresses the challenges above by providing:

- a **3-class classifier** (normal / ischemic / hemorrhagic) built on a Swin Transformer,
- **SAM-assisted region-of-interest** segmentation to focus the model on clinically relevant brain regions,
- **multi-method explainable AI** (LIME, Grad-CAM, SHAP) plus a **consensus mask** that visually highlights the lesion-driving region,
- a **hospital web application** with doctor/patient roles, patient management, scan upload, statistics, and **downloadable PDF clinical reports**.

**Figure 1** shows the explainable-AI output produced by the system for a real scan. The six panels (original input, LIME, Grad-CAM, SHAP, combined mask, and final overlay) let a clinician verify that the model is focusing on the actual lesion rather than on irrelevant anatomy — the key trust requirement that distinguishes this system from black-box classifiers.

```
          ┌──────────────────────────────────────────────┐
          │  Figure 1: Multi-method Explainable AI output │
          │  ┌─────────┐  ┌──────────┐  ┌───────────┐    │
          │  │ Original│  │   LIME   │  │  Grad-CAM │    │
          │  │  Input  │  │ (superpx │  │  (class   │    │
          │  │         │  │   ridge) │  │ activation│    │
          │  └─────────┘  └──────────┘  └───────────┘    │
          │  ┌─────────┐  ┌──────────┐  ┌───────────┐    │
          │  │  SHAP   │  │ Combined │  │   Final   │    │
          │  │ (Shapley│  │   Mask   │  │  Overlay  │    │
          │  │ values) │  │ (consensus)│  │           │    │
          │  └─────────┘  └──────────┘  └───────────┘    │
          └──────────────────────────────────────────────┘
     (actual image: outputs/explanations/00_002652_normal_xai.png)
```

---

## 2. Objectives

### 2.1 Primary Objective

To design and develop an **AI-based system that detects and classifies brain strokes (normal, ischemic, hemorrhagic) from CT/MRI scans** using a Swin Transformer, with SAM-assisted segmentation and multi-method explainable AI, packaged as a web-based clinical decision-support tool that generates downloadable reports.

### 2.2 Secondary Objectives

1. Build an end-to-end, resumable **training pipeline** (class-weighted loss, cosine learning-rate schedule, checkpointing).
2. Train a **Swin Transformer** classifier and evaluate with accuracy, precision, recall, F1-score, and a confusion matrix.
3. Integrate **SAM** for ROI/mask-assisted preprocessing of brain scans.
4. Implement **LIME, Grad-CAM, and SHAP** pixel-level explanations and fuse them into a **consensus mask**.
5. Build a **Flask hospital web application** with role-based access (doctor/patient), patient management, scan upload, and analysis orchestration.
6. Generate a **branded, downloadable PDF clinical report** containing patient details, prediction, probabilities, XAI figure, and doctor notes.
7. Provide **statistics dashboards** (stroke-type distribution, scans over time) for the department.
8. Expose a **FastAPI service** for programmatic analysis and report retrieval.
9. Write **automated tests** covering the API and web flows.

### 2.3 Expected Outcomes

| Outcome | Description |
|---|---|
| Trained checkpoint | `outputs/checkpoints/best_model.pt` (3-class Swin Transformer) |
| High-accuracy classification | Validation accuracy target ≥ 90% on the TEKNOFEST hold-out set |
| Explainable output | 6-panel XAI figure (LIME, Grad-CAM, SHAP, consensus mask, overlay) per scan |
| Clinical report | Downloadable PDF with patient details, prediction, and doctor notes |
| Hospital web app | Doctor and patient portals with full CRUD + statistics |
| API | JSON analysis endpoint + PDF retrieval |
| Documentation | This report, README, and test suite |

---

## 3. Scope of the Project

### 3.1 What Is Included

- **Data:** TEKNOFEST brain CT dataset — **7,202 images** in 3 classes (normal **4,551**, ischemic **1,361**, hemorrhagic **1,290**), split **70/15/15** into train (5,039), validation (1,079), and test (1,084).
- **Model:** Swin Transformer Tiny (`swin_tiny_patch4_window7_224`) fine-tuned from ImageNet-pretrained weights, 224×224 input.
- **Training:** class-weighted cross-entropy for imbalance, AdamW + cosine annealing, resume support, per-epoch checkpointing (best + latest).
- **Segmentation:** SAM (ViT-B) automatic mask generation with area/centrality mask selection and ROI cropping.
- **Explainability:** LIME (superpixel + ridge surrogate), Grad-CAM (last Swin stage), SHAP (Shapley value sampling over superpixels), consensus combined mask, and overlay figure.
- **Application:** Flask hospital system (login, patient CRUD, upload, analysis, report, doctor notes, statistics), FastAPI analysis API, SQLite persistence, PDF generation, and the web UI (NeuroScan design system).
- **Testing:** automated pytest suite for API and Flask flows.

### 3.2 What Is Excluded

- **DICOM handling:** the system currently accepts PNG/JPG images; native DICOM parsing and CT windowing are out of scope (noted as future work).
- **Severity grading:** mild/moderate/severe grading is *architected* but requires severity labels in the dataset, which are not available; severity is currently inferred heuristically from confidence and scan statistics.
- **Clinical deployment / certification:** the system is a decision-support prototype, not validated for independent diagnosis, and explicitly disclaims clinical use without expert review.
- **Multi-modal input:** only single 2D scans are processed; volumetric (3D) CT stacks and MR sequences are excluded.
- **SAM in the live path:** SAM ROI preprocessing is implemented as a standalone module; the live inference pipeline currently classifies the full image. Wiring SAM into the live path is listed as remaining/future work.
- **Multi-center validation:** no external hospital datasets or radiologist double-reading are included.

### 3.3 Target Users / Domain

- **Primary users:** radiologists, neurologists, emergency physicians, and resident doctors who need a second-opinion assistant for CT/MRI stroke screening.
- **Secondary users:** patients, who can log in and view/download their own AI reports generated by their doctor.
- **Domain:** medical imaging / computer-aided diagnosis, hospital information systems, and AI-in-healthcare research (final-year engineering project).

---

## 4. System Architecture

### 4.1 Architecture Diagram

```
                     ┌───────────────────────────────────────────────┐
                     │                 NEUROSCAN SYSTEM               │
                     └───────────────────────────────────────────────┘

  ┌────────────┐     ┌─────────────────────────────────────────────────────────────┐
  │   Doctor   │────▶│  Flask Web App (flask_app.py)                               │
  │ (browser)  │     │  ┌────────────┐ ┌──────────────┐ ┌───────────────┐          │
  └────────────┘     │  │ Auth/roles │ │ Patient CRUD │ │ Upload/Report │          │
  ┌────────────┐     │  └─────┬──────┘ └──────┬───────┘ └───────┬───────┘          │
  │  Patient   │────▶│        │               │                 │                  │
  │ (browser)  │     └────────┼───────────────┼─────────────────┼──────────────────┘
  └────────────┘              │               │                 │
        │                     ▼               ▼                 ▼
  ┌─────────────┐     ┌────────────────┐  ┌──────────────┐  ┌──────────────┐
  │  FastAPI    │     │  SQLite        │  │  Uploads     │  │  Analysis    │
  │  (/analyze) │     │  brain_stroke  │  │  outputs/    │  │  Pipeline    │
  │  /reports/  │     │  .db (users,   │  │  uploads/    │  │              │
  └─────────────┘     │  patients,     │  └──────────────┘  │  ┌─────────┐ │
        │             │  scans)        │                    │  │  SAM ROI │ │
        │             └────────────────┘                    │  └─────────┘ │
        ▼                                                    │  ┌─────────┐ │
  ┌───────────────────────────────┐                         │  │  Swin   │ │
  │  PDF Report (ReportLab)       │◀────────────────────────│  │ Classifier│ │
  └───────────────────────────────┘                         │  └─────────┘ │
        │                                                    │  ┌─────────┐ │
        ▼                                                    │  │  XAI:   │ │
  ┌───────────────────────────────┐                         │  │ LIME/Grad│ │
  │  XAI 6-panel figure           │◀────────────────────────│  │ CAM/SHAP │ │
  │  + per-method artifacts       │                         │  └─────────┘ │
  └───────────────────────────────┘                         └──────────────┘
```

### 4.2 Major Modules

| # | Module | File(s) | Responsibility |
|---|---|---|---|
| 1 | Data Ingestion & Preprocessing | `src/stroke_sam_swin/data.py` | Load images by class, resize/normalize, batch, augment |
| 2 | SAM ROI Segmentation | `src/stroke_sam_swin/sam_roi.py` | Generate lesion ROI masks and crops |
| 3 | Model Definition & Training | `model.py`, `train.py` | Build Swin, train with class weights + cosine schedule, checkpoint |
| 4 | Inference & Classification | `inference.py` | Preprocess → forward → probabilities → prediction |
| 5 | Explainable AI | `explain.py` | LIME, Grad-CAM, SHAP, consensus mask, 6-panel figure |
| 6 | Clinical Report Generation | `report.py` | Branded PDF report (ReportLab) |
| 7 | Web Application | `flask_app.py`, `templates/` | Hospital workflow: auth, CRUD, upload, report, notes, stats |
| 8 | Database | `database.py` | SQLite schema + CRUD (users, patients, scans) |
| 9 | API Service | `app.py` | FastAPI `/analyze` + `/reports/{name}` |

### 4.3 Data Flow Between Modules

1. **Doctor registers/logs in** (Web App → Database) and **creates a patient** (Web App → Database).
2. **Doctor uploads a scan** (Web App → Uploads).
3. The **Analysis Pipeline** runs: *(optional)* SAM ROI → Swin classification → XAI (LIME/Grad-CAM/SHAP) → consensus mask.
4. Results are **persisted** (Web App → Database: `scans` record) and **copied into uploads** (XAI figure).
5. The **PDF Report** is generated (Report Builder ← result + patient + doctor notes).
6. **Doctor views/annotates** the report and **adds doctor notes** (Web App → Database).
7. **Patient logs in** and views/downloads their own report (Web App → Database → PDF).
8. **Statistics page** aggregates counts by type and month (Web App → Database → Chart.js).

### 4.4 Technology Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Deep learning | PyTorch, torchvision, timm (Swin Transformer), segment-anything (SAM) |
| Explainability | captum (Grad-CAM / SHAP Shapley-value), hand-rolled LIME surrogate |
| Image processing | OpenCV, Pillow |
| Web backend | Flask (hospital app), FastAPI + uvicorn (API) |
| Frontend | Jinja2 templates, HTML/CSS (Open Sans, NeuroScan design system), Chart.js |
| Database | SQLite (`brain_stroke.db`) |
| Reports | ReportLab (PDF) |
| Data / metrics | NumPy, pandas, scikit-learn |
| Testing | pytest |
| Dev environment | Windows, CPU-only PyTorch (torch 2.13.0+cpu), `requirements.txt` |

---

## 5. Detailed Module Design

### 5.1 Data Ingestion and Preprocessing Module

**File:** `src/stroke_sam_swin/data.py`

- **Inputs:** images under `data/raw/{train,val,test}/{hemorrhagic,ischemic,normal}/`; config (image size = 224, batch size = 8).
- **Processing:**
  - Read PNG/JPG, convert to RGB, resize to 224×224.
  - Normalize with ImageNet mean/std `[0.485,0.456,0.406]` / `[0.229,0.224,0.225]`.
  - Build `torch.utils.data.DataLoader` for train/val/test.
- **Outputs:** `DataLoader` objects; class mapping `0=hemorrhagic, 1=ischemic, 2=normal`.
- **Dataset statistics (TEKNOFEST):**

| Class | Train | Val | Test | Total |
|---|---|---|---|---|
| normal | 3,185 | 682 | 684 | 4,551 |
| ischemic | 952 | 204 | 205 | 1,361 |
| hemorrhagic | 902 | 193 | 195 | 1,290 |
| **Total** | **5,039** | **1,079** | **1,084** | **7,202** |

### 5.2 SAM ROI Segmentation Module

**File:** `src/stroke_sam_swin/sam_roi.py`

- **Inputs:** raw image; SAM ViT-B checkpoint (`checkpoints/sam_vit_b_01ec64.pth`); config `sam.{min_mask_area_ratio, max_mask_area_ratio, crop_padding}`.
- **Processing:**
  - Automatic mask generation via `SamAutomaticMaskGenerator`.
  - Candidate masks filtered by area ratio (`min_mask_area_ratio`–`max_mask_area_ratio`).
  - Best mask selected by centrality + area heuristics.
  - ROI cropped with padding; mask saved alongside.
- **Outputs:** ROI crop and mask written to `data/processed/sam_roi/`.
- **Status:** implemented as a standalone CLI (`python -m stroke_sam_swin.sam_roi --config configs/default.yaml`); optional integration into the live pipeline (listed as remaining work).

### 5.3 Model Definition and Training Module

**Files:** `src/stroke_sam_swin/model.py`, `train.py`

- **Inputs:** config (`model.name = swin_tiny_patch4_window7_224`, pretrained = true), dataset loaders, optional existing checkpoint for resume.
- **Processing:**
  - Build Swin Transformer Tiny with a 3-class head.
  - **Class-weighted cross-entropy** loss (counts inverse-frequency to counter imbalance).
  - **AdamW** optimizer (lr 1e-4, weight decay 0.01) + **CosineAnnealingLR**.
  - Validation per epoch; save `best_model.pt` and per-epoch `latest_model.pt` (with optimizer/scheduler state for resumable training).
  - `--resume --epochs N` continues from `latest_model.pt`; a class-mismatch guard raises an error instead of silently corrupting the model.
- **Outputs:** `outputs/checkpoints/best_model.pt` (~330 MB), `latest_model.pt`, `outputs/reports/training_history.json`.
- **Current results (Round 1, 5 epochs on CPU, ~16 min/epoch):**

| Epoch | Train Acc | Train F1 | Val Acc | Val F1 |
|---|---|---|---|---|
| 1 | 0.680 | 0.635 | 0.843 | 0.755 |
| 2 | 0.833 | 0.807 | 0.876 | 0.842 |
| 3 | 0.883 | 0.861 | 0.900 | 0.871 |
| **4** | 0.898 | 0.883 | **0.926** | **0.912** |
| 5 | 0.920 | 0.906 | 0.907 | 0.893 |

Best checkpoint (epoch 4): **validation accuracy 92.6%, macro-F1 91.2%**.

### 5.4 Inference and Classification Module

**File:** `src/stroke_sam_swin/inference.py`

- **Inputs:** image path, config, checkpoint path; `generate_explanation` flag.
- **Processing:**
  - Checkpoint availability check → `model_missing` status if absent.
  - Preprocess, forward pass, softmax over the 3 classes.
  - Top-1 class + confidence; probabilities as a pandas DataFrame.
  - Optional: run the full XAI pipeline.
  - Model is cached in memory (`_MODEL_CACHE`) to avoid reloading per request.
- **Outputs:** dict `{status, prediction, confidence, probabilities, explanation_path, explanation_artifacts}`.

### 5.5 Explainable AI (XAI) Module

**File:** `src/stroke_sam_swin/explain.py`

- **Inputs:** input image tensor, model, target class, device, config (`xai.lime`, `xai.grad_cam`, `xai.shap`).
- **Processing (four methods):**
  1. **LIME** — grid superpixels (6×6) + distance-weighted ridge regression surrogate over the target-class probability; fitted coefficients → attribution map.
  2. **Grad-CAM** — forward/backward hooks on the last Swin stage's `norm2`; channel-weighted activation sum, upsampled to 224×224.
  3. **SHAP** — Shapley value sampling (captum) over an 8×8 superpixel grid; per-pixel attribution.
  4. **Consensus mask** — mean of LIME+Grad-CAM+SHAP, thresholded at 50% of max; red overlay on the scan.
- **Outputs:** a **6-panel figure** (`outputs/explanations/<scan>_<class>_xai.png`) plus individual `*_heatmap.png`, `*_overlay.png`, `*_combined_mask.png`, `*_combined_overlay.png` artifacts. Average runtime ≈ 2 min/scan on CPU.
- **Figure 1 (Section 1.4)** is the canonical output of this module.

### 5.6 Clinical Report Generation Module

**File:** `src/stroke_sam_swin/report.py`

- **Inputs:** output path, image name, analysis result, explanation figure, optional patient record, doctor notes, scan date.
- **Processing:** build a ReportLab `SimpleDocTemplate` with the NeuroScan design system: navy/teal header band, patient-details table, semantic color-coded prediction badge (green=normal, blue=ischemic, red=hemorrhagic, amber=model-missing), class-probability table, aspect-aware XAI figure embed, amber doctor-notes box, and a disclaimer footer.
- **Outputs:** branded PDF saved as `outputs/reports/report_<scan_id>.pdf` (and `latest_clinical_report.pdf` via the API).

### 5.7 Web Application Module

**File:** `flask_app.py` + `templates/*.html`

- **Inputs:** HTTP requests (login, CRUD forms, file upload, report views).
- **Processing:**
  - **Auth & roles:** `login_required`, `doctor_required`, `patient_required` decorators; password hashing (`werkzeug`); demo doctor `admin/admin123`.
  - **Doctor flows:** dashboard (stat cards), patient add/edit/search/delete, scan upload → `analyze_scan` → report, doctor notes, statistics (Chart.js pie + bar).
  - **Patient flow:** login → dashboard listing only their own scans → view + PDF download.
  - **Static serving:** `outputs/uploads` at `/uploads`.
- **Outputs:** rendered HTML pages, persisted database records, uploaded files.

### 5.8 Database Module

**File:** `database.py`

- **Schema (SQLite, auto-created as `brain_stroke.db`):**
  - `users (id, username, password_hash, role)`
  - `patients (id, name, age, gender, phone, blood_group, user_id → users)`
  - `scans (id, patient_id, filename, prediction, confidence, severity, stroke_score, probabilities_json, explanation_path, report_path, doctor_notes, created_at)`
- **Functions:** `init_db`, `create_user`, `get_user_by_username`, `create_patient`, `get_patients`, `get_patient`, `update_patient`, `delete_patient`, `create_scan`, `get_scan`, `get_scans_by_patient`, `get_all_scans`, `update_scan_notes`, `scan_counts`.
- **Outputs:** persistent relational storage for users, patients, and scan records.

### 5.9 API Service Module

**File:** `app.py`

- **Endpoints:**
  - `POST /analyze` — multipart upload → `analyze_scan` (explanation off for speed) → JSON `{status, stroke_type, confidence_score, severity, stroke_score, probabilities, image_url, report_url}` + generates `latest_clinical_report.pdf`.
  - `GET /reports/{name}` — download a generated PDF.
  - `GET /` — serves the frontend (React build if present, else `templates/index.html`).
- **Outputs:** JSON responses and file downloads for programmatic/integration use.

---

## 6. Implementation Progress Status

### 6.1 Work Completed

| Area | Status | Details |
|---|---|---|
| Dataset | ✅ Done | TEKNOFEST 7,202 CT images; 70/15/15 split; class-balanced loader |
| Training pipeline | ✅ Done | Class-weighted loss, AdamW, cosine schedule, resume, per-epoch checkpoints |
| Model training (Round 1) | ✅ Done | 5 epochs; best val_acc **0.926**, val_F1 **0.912** (epoch 4) |
| Model training (Round 2) | ✅ Done | Resumed to **10 epochs**; best val_acc **0.953**, val_F1 **0.942** (epoch 8) |
| Model training (Round 3) | 🔄 In progress | Resumed to **15 epochs** (`--resume --epochs 15`); epochs 9–10 not better than epoch 8 |
| Full test-set evaluation | ✅ Done | **Test acc 0.935**, macro-F1 **0.918** on 1,084 test images; confusion matrix saved |
| SAM ROI module | ✅ Done | `load_sam`, mask selection, ROI crop CLI |
| Swin inference | ✅ Done | Cached model, probabilities, model_missing guard |
| XAI | ✅ Done | LIME + Grad-CAM + SHAP + consensus mask + 6-panel figure |
| Lesion localization | ✅ Done | Red-contour overlay, affected brain region, area (mm²), % brain, centroid |
| Stroke score + severity | ✅ Done | Real softmax sum ×100; Mild / Moderate / Severe labels |
| PDF report | ✅ Done | NeuroScan-branded, patient/notes/probabilities/XAI/disclaimer + Lesion Localization section |
| Hospital web app | ✅ Done | Roles, patient CRUD, upload/analyze, report, notes, statistics |
| Database | ✅ Done | `brain_stroke.db`, users/patients/scans CRUD + lesion columns |
| API service | ✅ Done | `/analyze`, `/reports/{name}` (FastAPI) + Flask app |
| UI design system | ✅ Done | NeuroScan (navy/teal, Open Sans, semantic badges) + violet/cyan gradients |
| UI animations | ✅ Done | Scroll reveals, count-up stats, scan loader, ECG/pulse/halo animations; `prefers-reduced-motion` |
| Tests | ✅ Done | **11 pytest tests passing** (API + Flask flows + lesion localization) |

### 6.2 Remaining Work

| Area | Status | Details |
|---|---|---|
| Training Round 3 | 🔄 In progress | 15-epoch run in background; final `best_model.pt` + re-evaluation when done |
| SAM in live pipeline | ⏳ Pending | Optional: wire `sam_roi` ROI preprocessing into `analyze_scan` |
| Clinical validation | ⏳ Pending | Region atlas is heuristic (not MNI/Harvard-Oxford registered); area assumes ~0.5 mm/px |
| DICOM support | ⏳ Pending | CT windowing + DICOM parsing |
| Deployment | ⏳ Pending | Docker packaging, gunicorn/uvicorn production config |

---

## 7. Project Completion Plan

| Phase | Task | Timeline | Milestone |
|---|---|---|---|
| 1 | Complete Round 2 training to 10 epochs | ✅ Done | 10-epoch checkpoint (val_acc 0.953, epoch 8) |
| 2 | Complete Round 3 training to 15 epochs | In progress | Final `best_model.pt` |
| 3 | Full test-set evaluation + confusion matrix | ✅ Done | Test acc 0.935, macro-F1 0.918 |
| 4 | Wire SAM ROI into the live analysis pipeline (optional) | Pending | SAM-enhanced analysis |
| 5 | Clinical scenario demo: doctor creates patient, uploads scan, annotates, downloads PDF | ✅ Done | End-to-end demo (lesion overlay + clinical summary) |
| 6 | Patient portal verification + statistics | ✅ Done | Verified workflows |
| 7 | Documentation, README, final report, presentation | In progress | Project submission |

---

## 8. Conclusion

This project delivers a complete, end-to-end AI system for brain stroke detection and classification that addresses the three core gaps in existing work: **accuracy** (Swin Transformer, 93.5% test accuracy and 0.918 macro-F1 at the epoch-8 checkpoint, with further training scheduled), **trustworthiness** (LIME, Grad-CAM, SHAP and a consensus mask explain every prediction, plus lesion localization with affected region, area and centroid), and **deployability** (a hospital web application with patient management, role-based access, statistics, and branded PDF reports).

The system demonstrates that transformer-based classifiers combined with multi-method explainability can form a practical decision-support workflow for stroke triage, particularly valuable in settings with limited access to specialist radiologists. The clinical summary produced for each stroke scan — type, severity, affected region, lesion area and XAI evidence — gives referring clinicians an interpretable starting point. Remaining work focuses on finishing the 15-epoch training schedule and re-evaluation, optionally integrating SAM ROI pre-processing into the live path, and clinical validation — after which the system can be positioned as a research-grade assistant rather than a diagnostic tool.

**Key limitation to reiterate:** this is a decision-support prototype. It is not validated for independent diagnosis, and final interpretation must always be made by a qualified clinician.

---

## 9. References

1. **Kirillov, A., Mintun, E., Ravi, N., et al.** *Segment Anything.* ICCV 2023. arXiv:2304.02643.
2. **Liu, Z., Lin, Y., Cao, Y., et al.** *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.* ICCV 2021. arXiv:2103.14030.
3. **Selvaraju, R. R., Cogswell, M., Das, A., et al.** *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* CVPR 2017. arXiv:1610.02391.
4. **Ribeiro, M. T., Singh, S., Guestrin, C.** *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.* KDD 2016. arXiv:1602.04938.
5. **Lundberg, S. M., Lee, S.-I.** *A Unified Approach to Interpreting Model Predictions.* NeurIPS 2017. arXiv:1705.07874.
6. **Kokhlikyan, N., Miglani, V., Martin, M., et al.** *Captum: A Unified and Generic Model Interpretability Library for PyTorch.* arXiv:2009.07829.
7. **Rw, G. W., et al.** *Fastai/PyTorch Image Models (timm).* GitHub: https://github.com/huggingface/pytorch-image-models
8. **World Health Organization.** *Stroke fact sheet.* https://www.who.int
9. **Indian Stroke Association / WHO India stroke burden reports** (annual stroke incidence ~1.8M in India).
10. **TEKNOFEST 2021 Brain CT image dataset** (competition dataset for stroke classification).
11. **OpenCV** documentation. https://opencv.org
12. **ReportLab** documentation. https://www.reportlab.com
13. **Flask** documentation. https://flask.palletsprojects.com
14. **FastAPI** documentation. https://fastapi.tiangolo.com
