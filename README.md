# Brain Stroke Detection and Classification using SAM and Swin Transformer

This project is a practical starter pipeline for brain stroke image classification.
It uses:

- **SAM** for ROI/mask-assisted preprocessing.
- **Swin Transformer** for image classification.
- **Swin-based explanation heatmaps** for visual decision support (LIME, Grad-CAM, SHAP).
- **Lesion localization** with affected region, area, and centroid for clinical summaries.

The default task assumes 3 classes:

- `normal`
- `ischemic`
- `hemorrhagic`

You can rename or extend classes by changing the dataset folders and `configs/default.yaml`.

## Project Structure

```text
brain/
  configs/default.yaml
  data/
    raw/
      train/<class_name>/*.png
      val/<class_name>/*.png
      test/<class_name>/*.png
    processed/
  outputs/
    checkpoints/
    explanations/
    reports/
  src/stroke_sam_swin/
    data.py
    evaluate.py
    explain.py
    inference.py
    localization.py
    model.py
    report.py
    sam_roi.py
    train.py
    utils.py
  app.py            # FastAPI service (port 8004)
  flask_app.py      # Flask hospital web app (port 8003)
  database.py       # SQLite ORM + schema migrations
  templates/        # Web UI (base/landing/login/dashboard/report/...)
  tests/
```

## Setup

Create an environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

SAM is optional until you run ROI preprocessing. To use SAM, download a checkpoint from Meta AI's Segment Anything release page and set its path in `configs/default.yaml`.

## Dataset Layout

Place images like this:

```text
data/raw/train/normal/image_001.png
data/raw/train/ischemic/image_002.png
data/raw/train/hemorrhagic/image_003.png

data/raw/val/normal/...
data/raw/test/normal/...
```

Use PNG/JPG images. If you have DICOM files, convert them to PNG first with consistent windowing.

## 1. Optional SAM ROI Preprocessing

This step generates cropped ROI images and masks:

```powershell
python -m stroke_sam_swin.sam_roi --config configs/default.yaml
```

It writes to:

```text
data/processed/sam_roi/
```

After preprocessing, change `dataset.root` in `configs/default.yaml` from `data/raw` to `data/processed/sam_roi`.

## 2. Train Swin Transformer

```powershell
python -m stroke_sam_swin.train --config configs/default.yaml
```

Best checkpoint:

```text
outputs/checkpoints/best_model.pt
```

## 3. Evaluate

```powershell
python -m stroke_sam_swin.evaluate --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Reports are written to:

```text
outputs/reports/
```

## 4. Generate Swin Explanation Heatmaps

```powershell
python -m stroke_sam_swin.explain --config configs/default.yaml --checkpoint outputs/checkpoints/best_model.pt --image data/raw/test/ischemic/example.png
```

Outputs are written to:

```text
outputs/explanations/
```

## 5. Run the Web Clinical Tool

Two interfaces are available:

```powershell
# Flask hospital web app (recommended for the demo)
python flask_app.py        # http://127.0.0.1:8003/  (login: admin / admin123)

# FastAPI service
python app.py              # http://127.0.0.1:8004/docs
```

The web app supports:

- Doctor / patient role-based access and patient management.
- CT/MRI image upload and prediction with confidence and severity.
- Stroke score, class probability table, and affected brain region.
- XAI heatmaps (LIME, Grad-CAM, SHAP) plus a consensus mask.
- Lesion localization: red contour overlay, lesion area (mm²), % of brain affected, centroid.
- Downloadable NeuroScan-branded PDF clinical report.

If `outputs/checkpoints/best_model.pt` is not available yet, the app opens in training-required mode.

## 6. Tests

```powershell
python -m pytest -q
```

## Notes for a Strong Final-Year or Research Project

- Compare raw-image Swin training with SAM ROI-assisted Swin training.
- Report accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Validate separately for CT and MRI if your dataset contains both.
- Include radiologist review of SAM masks and Swin explanation heatmaps if possible.
- Do not claim clinical readiness unless validated on external hospital data.
- The lobe/region atlas is a heuristic approximation (not MNI/Harvard-Oxford registered); area assumes ~0.5 mm/pixel.
