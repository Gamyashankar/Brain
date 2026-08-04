# Brain Stroke Detection and Classification using SAM and Swin Transformer Models

## Abstract

Stroke is the second leading cause of death globally and remains a critical medical emergency in India, where nearly 1.8 million people are affected every year. Accurate and timely diagnosis is essential, but current workflows depend heavily on manual interpretation of MRI and CT scans by radiologists. This creates a major barrier in low-resource and rural healthcare settings, where specialist availability is limited.

This project proposes an AI system for automated brain stroke detection, classification, and severity assessment using standard hospital imaging modalities. The system integrates two core model components: a Segment Anything Model (SAM)-assisted lesion segmentation module and a Swin Transformer-based classification and visual explanation module. The classification module is designed to distinguish ischemic and hemorrhagic stroke and can be extended to grade severity as mild, moderate, or severe when severity labels are available.

The proposed system is intended for deployment as a web-based clinical decision-support tool. A doctor can upload an MRI or CT scan and receive a diagnostic output that includes predicted stroke type, confidence score, segmented scan, visual explanation, and a downloadable clinical report suitable for hospital documentation workflows. By addressing black-box AI limitations, missing severity grading, and lack of deployment-ready clinical interfaces, this work aims to bridge the gap between AI research and practical diagnostic support, especially in low-resource healthcare environments.

## Problem Statement

Manual stroke identification from CT or MRI scans is time-sensitive and reader-dependent. In many regions, access to expert radiologists is limited, causing diagnostic delays. A deep learning system can assist clinicians by providing fast classification, lesion localization, and visual explanations that help users inspect whether the model is focusing on clinically plausible regions.

## Objectives

- Build an AI system for brain stroke detection and classification from MRI or CT images.
- Use SAM-assisted segmentation to identify lesion or region-of-interest areas.
- Train a Swin Transformer classifier to classify stroke type.
- Extend the classifier for severity grading as mild, moderate, or severe when severity labels are available.
- Generate Swin-based visual explanation heatmaps for model interpretation.
- Produce a downloadable diagnostic report for clinical documentation workflows.
- Evaluate using accuracy, precision, recall, F1-score, confusion matrix, and clinical review where possible.

## Methodology

1. Data collection:
   Organize images into train, validation, and test folders by class.

2. Preprocessing:
   Resize images, normalize them, and optionally use SAM to generate ROI crops.

3. SAM segmentation:
   SAM generates candidate masks. The pipeline selects a mask using area and centrality heuristics, then crops the ROI with padding.

4. Classification:
   A Swin Transformer model is trained using cross-entropy loss and AdamW optimization.

5. Visual explanation:
   The implementation uses Swin Transformer prediction behavior to generate heatmaps that highlight image regions influencing the classification result. No additional explainability model is required.

6. Evaluation:
   The trained model is evaluated on validation or test data using classification report metrics and a confusion matrix.

## Expected Outputs

- Trained Swin Transformer checkpoint.
- Classification report and confusion matrix.
- SAM ROI crops and masks.
- Swin explanation heatmap overlays for selected images.
- Confidence score for each prediction.
- Downloadable clinical-style report.

## Keywords

Brain Stroke Detection, Segment Anything Model, SAM, Swin Transformer, Ischemic Stroke, Hemorrhagic Stroke, Medical Image Segmentation, Clinical Decision Support.

## Limitations

- SAM is a general segmentation model and may need prompt tuning or medical fine-tuning for best lesion localization.
- Small or imbalanced datasets can cause unreliable performance.
- The model should not be used clinically without external validation and expert review.

## Future Improvements

- Add DICOM preprocessing with CT windowing.
- Compare SAM ROI preprocessing against raw-image training.
- Add transformer attention rollout if deeper Swin-specific interpretation is required.
- Use external multi-center datasets for stronger validation.
- Fine-tune medical segmentation models for stroke lesion masks.
