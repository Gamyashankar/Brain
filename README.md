# Brain Stroke Detection and Classification using SAM and Swin Transformer

This project is a practical starter pipeline for brain stroke image classification.
It uses:

- **SAM** for ROI/mask-assisted preprocessing.
- **Swin Transformer** for image classification.
- **Swin-based explanation heatmaps** for visual decision support.

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
    model.py
    sam_roi.py
    train.py
    utils.py
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

```powershell
streamlit run app.py
```

The web app supports:

- CT/MRI image upload.
- Prediction and confidence display after training.
- Class probability table.
- Swin explanation heatmap display after training.
- Downloadable PDF clinical-style report.

If `outputs/checkpoints/best_model.pt` is not available yet, the app opens in training-required mode.

## Notes for a Strong Final-Year or Research Project

- Compare raw-image Swin training with SAM ROI-assisted Swin training.
- Report accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Validate separately for CT and MRI if your dataset contains both.
- Include radiologist review of SAM masks and Swin explanation heatmaps if possible.
- Do not claim clinical readiness unless validated on external hospital data.
