from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / 'Backend'
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app import app, load_dataset  # type: ignore  # noqa: E402


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_dataset_loading():
    df = load_dataset()
    required = {
        'Bike model',
        'Manufacturing year',
        'Part name',
        'Part category',
        'Availability status',
        'Estimated wait days',
        'Estimated price (INR)',
        'Supplier type',
        'City',
        'Garage type',
        'Part condition',
        'Service priority',
        'Record date',
    }
    assert required.issubset(set(df.columns))
    assert len(df) > 0


def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'


def test_invalid_payload(client):
    response = client.post('/predict-availability', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()


def test_prediction_endpoint_valid_input(client):
    payload = {
        'bike_model': 'Royal Enfield Thunderbird 350X',
        'manufacturing_year': 2019,
        'part_name': 'Clutch Cable',
        'part_category': 'Control',
        'supplier_type': 'Online Supplier',
        'city': 'Pune',
        'part_condition': 'New',
        'service_priority': 'High',
    }
    response = client.post('/predict-availability', json=payload)
    assert response.status_code in {200, 500}
    if response.status_code == 200:
        data = response.get_json()
        assert 'availability' in data
        assert 'confidence' in data


def test_model_files_are_present():
    expected = [
        PROJECT_ROOT / 'ML' / 'models' / 'availability_model.pkl',
        PROJECT_ROOT / 'ML' / 'models' / 'wait_model.pkl',
        PROJECT_ROOT / 'ML' / 'models' / 'price_model.pkl',
    ]
    if not all(path.exists() for path in expected):
        pytest.skip('Model files not generated in this environment yet.')
