import io

from fastapi.testclient import TestClient
from PIL import Image

import app as app_module
import flask_app as flask_app_module


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
    summary = app_module.build_prediction_summary('ischemic', 0.92, image)

    assert summary['stroke_type'] == 'Ischemic Stroke'
    assert summary['confidence_score'] == 92.0
    assert summary['severity'] == 'High'
    assert summary['probabilities'] == [{'class': 'Stroke', 'probability': 92.0}]


def test_flask_analysis_requests_explanation_generation(monkeypatch):
    captured = {}

    def fake_analyze_scan(*args, **kwargs):
        captured.update(kwargs)
        return {
            'status': 'ok',
            'prediction': 'stroke',
            'confidence': 0.57,
            'probabilities': [
                {'class': 'normal', 'probability': 0.43},
                {'class': 'stroke', 'probability': 0.57},
            ],
            'explanation_path': None,
            'explanation_artifacts': {
                'input': 'outputs/explanations/sample_input.png',
                'heatmap': 'outputs/explanations/sample_heatmap.png',
                'overlay': 'outputs/explanations/sample_overlay.png',
            },
        }

    monkeypatch.setattr(flask_app_module, 'analyze_scan', fake_analyze_scan)

    image_bytes = io.BytesIO()
    Image.new('RGB', (32, 32), color='white').save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()

    client = flask_app_module.app.test_client()
    with client.session_transaction() as session:
        session['logged_in'] = True
    response = client.post('/analyze', data={'file': (io.BytesIO(image_bytes), 'scan.png')}, content_type='multipart/form-data')

    assert response.status_code == 200
    assert captured.get('generate_explanation') is True


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
    assert data['stroke_score'] == 94.64
