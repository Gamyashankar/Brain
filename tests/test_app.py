import io
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
import flask_app as flask_app_module
from stroke_sam_swin.localization import clean_lesion_mask, compute_lesion_stats, localize_lesion, map_brain_region


def test_root_page_returns_html():
    client = TestClient(app_module.app)
    response = client.get('/')
    assert response.status_code == 200
    assert 'Brain Stroke AI' in response.text


def test_infer_stroke_type_uses_heuristic_for_binary_stroke_prediction():
    image = Image.new('RGB', (64, 64), color='white')
    assert app_module.infer_stroke_type('stroke', image) == 'Hemorrhagic Stroke'


def test_build_prediction_summary_uses_stroke_type_severity_and_simplified_probabilities():
    image = Image.new('RGB', (64, 64), color='white')
    summary = app_module.build_prediction_summary(
        'ischemic', 0.92, image,
        probabilities=[
            {'class': 'normal', 'probability': 0.03},
            {'class': 'ischemic', 'probability': 0.92},
            {'class': 'hemorrhagic', 'probability': 0.05},
        ],
    )

    assert summary['stroke_type'] == 'Ischemic Stroke'
    assert summary['confidence_score'] == 92.0
    assert summary['severity'] == 'Severe'
    assert summary['stroke_score'] == 97.0
    assert summary['probabilities'] == []


def test_build_prediction_summary_stroke_score_for_normal_is_low():
    summary = app_module.build_prediction_summary(
        'normal', 0.98,
        probabilities=[
            {'class': 'normal', 'probability': 0.98},
            {'class': 'ischemic', 'probability': 0.015},
            {'class': 'hemorrhagic', 'probability': 0.005},
        ],
    )

    assert summary['stroke_type'] == 'No Stroke Detected'
    assert summary['severity'] == 'None'
    assert summary['stroke_score'] == 2.0


def test_flask_analysis_requests_explanation_generation(monkeypatch):
    captured = {}

    def fake_analyze_scan(*args, **kwargs):
        captured.update(kwargs)
        return {
            'status': 'ok',
            'prediction': 'normal',
            'confidence': 0.57,
            'probabilities': [
                {'class': 'normal', 'probability': 0.57},
                {'class': 'ischemic', 'probability': 0.28},
                {'class': 'hemorrhagic', 'probability': 0.15},
            ],
            'explanation_path': None,
            'explanation_artifacts': {
                'input': 'outputs/explanations/sample_input.png',
                'heatmap': 'outputs/explanations/sample_heatmap.png',
                'overlay': 'outputs/explanations/sample_overlay.png',
            },
        }

    monkeypatch.setattr(flask_app_module, 'analyze_scan', fake_analyze_scan)

    patient_id = flask_app_module.create_patient(
        'XAI Test', 45, 'Female', '999', 'A+', '', 'xaitest', 'pass123'
    )

    image_bytes = io.BytesIO()
    Image.new('RGB', (32, 32), color='white').save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    try:
        client = flask_app_module.app.test_client()
        with client.session_transaction() as session:
            session['user_id'] = 1
            session['role'] = 'doctor'
        response = client.post(
            f'/doctor/patients/{patient_id}/upload',
            data={'file': (io.BytesIO(image_bytes), 'scan.png')},
            content_type='multipart/form-data',
        )

        assert response.status_code == 302
        assert captured.get('generate_explanation') is True
    finally:
        flask_app_module.delete_patient(patient_id)
        input_file = Path('outputs/uploads') / f'p{patient_id}_scan_input.png'
        if input_file.exists():
            input_file.unlink()


def test_analyze_endpoint_returns_simplified_stroke_prediction(monkeypatch):
    def fake_analyze_scan(*args, **kwargs):
        return {
            'status': 'ok',
            'prediction': 'ischemic',
            'confidence': 0.92,
            'probabilities': [
                {'class': 'normal', 'probability': 0.03},
                {'class': 'ischemic', 'probability': 0.92},
                {'class': 'hemorrhagic', 'probability': 0.05},
            ],
            'explanation_path': None,
        }

    def fake_build_report(*args, **kwargs):
        return None

    monkeypatch.setattr(app_module, 'analyze_scan', fake_analyze_scan)
    monkeypatch.setattr(app_module, 'build_clinical_report', fake_build_report)
    monkeypatch.setattr(app_module, 'load_config', lambda path: {'outputs': {'reports': 'outputs/reports'}})

    image_bytes = io.BytesIO()
    Image.new('RGB', (32, 32), color='white').save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    client = TestClient(app_module.app)
    response = client.post('/analyze', files={'file': ('scan.png', image_bytes, 'image/png')})

    assert response.status_code == 200
    data = response.json()
    assert data['stroke_type'] == 'Ischemic Stroke'
    assert data['confidence_score'] == 92.0
    assert data['stroke_score'] == 97.0


def _synthetic_brain_and_lesion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = np.full((224, 224, 3), 20, dtype=np.uint8)
    cv = np.zeros((224, 224), dtype=np.uint8)
    cv[40:190, 40:190] = 200
    image[cv > 0] = (200, 200, 200)
    brain = cv > 0
    lesion = np.zeros((224, 224), dtype=bool)
    lesion[100:130, 60:90] = True
    return image, brain, lesion


def test_lesion_stats_compute_area_and_percentage():
    image, brain, lesion = _synthetic_brain_and_lesion()
    stats = compute_lesion_stats(lesion, brain, pixel_mm=0.5)
    assert stats['area_mm2'] == 30 * 30 * 0.25
    assert stats['pct_brain'] == round(30 * 30 / brain.sum() * 100.0, 2)
    assert stats['centroid_x'] == 74.5
    assert stats['centroid_y'] == 114.5


def test_lesion_region_mapping_is_anatomical():
    image, brain, lesion = _synthetic_brain_and_lesion()
    frontal = np.zeros_like(lesion)
    frontal[60:90, 70:100] = True
    stats = compute_lesion_stats(frontal, brain)
    region = map_brain_region(stats, brain)
    assert region == 'Left Frontal Lobe'


def test_localize_lesion_returns_overlay_and_stats(tmp_path):
    image, brain, lesion = _synthetic_brain_and_lesion()
    result = localize_lesion(image, lesion, tmp_path, 'synthetic_ischemic')
    assert result is not None
    assert result['detected'] is True
    assert result['area_mm2'] > 0
    assert result['pct_brain'] > 0
    assert result['overlay_path']
    assert Path(result['overlay_path']).exists()
    saved = np.asarray(Image.open(result['overlay_path']).convert('RGB'))
    assert tuple(saved[115, 75]) == (255, 0, 0) or tuple(saved[115, 75]) != tuple(image[115, 75])


def test_localize_lesion_returns_none_without_mask(tmp_path):
    image, brain, lesion = _synthetic_brain_and_lesion()
    empty = np.zeros_like(lesion)
    assert localize_lesion(image, empty, tmp_path, 'none') is None


def test_clean_lesion_mask_keeps_largest_component():
    image, brain, lesion = _synthetic_brain_and_lesion()
    noisy = lesion.copy()
    noisy[200:210, 200:210] = True
    cleaned = clean_lesion_mask(noisy)
    assert not cleaned[200:210, 200:210].any()
    assert cleaned.sum() > lesion.sum() * 0.95
