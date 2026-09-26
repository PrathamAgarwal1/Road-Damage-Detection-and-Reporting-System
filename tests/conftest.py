"""
Test setup: runs the real app against a throwaway local MongoDB database.
External services (Gemini, SMTP) are disabled; Open-Meteo / Nominatim are stubbed.
Tests are skipped when no local MongoDB is reachable.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

TEST_DB_URI = os.getenv('TEST_MONGO_URI', 'mongodb://localhost:27017/roadsight_pytest')
os.environ.update(MONGO_URI=TEST_DB_URI, GEMINI_API_KEY='', SMTP_HOST='', ALERT_EMAILS='',
                  ADMIN_EMAIL='admin@test.local', ADMIN_PASSWORD='adminpass', FLASK_DEBUG='false')

from pymongo import MongoClient  # noqa: E402

_client = MongoClient(TEST_DB_URI, serverSelectionTimeoutMS=1500)
try:
    _client.admin.command('ping')
    MONGO_OK = True
except Exception:
    MONGO_OK = False

DATASET = os.path.join(ROOT, 'road_damage_dataset')


def dataset_image(folder, index=0):
    files = sorted(f for f in os.listdir(os.path.join(DATASET, folder)) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    return os.path.join(DATASET, folder, files[index])


@pytest.fixture(scope='session')
def app_module():
    if not MONGO_OK:
        pytest.skip('Local MongoDB not reachable (set TEST_MONGO_URI)')
    _client.drop_database(_client.get_default_database().name)
    import services
    # Deterministic, offline external calls
    services.compute_weather_risk = lambda lat, lon: {
        'available': lat is not None, 'riskScore': 0.3 if lat is not None else 0.0, 'pastRainMm': 12.0,
        'forecastRainMm': 20.0, 'heavyRainDays': 0, 'freezeThawCycles': 0, 'avgTempSwingC': 9.0}
    services.geocode_address = lambda address: (26.85, 75.80) if address else (None, None)
    services.reverse_geocode = lambda lat, lon: 'Test Street, Jaipur'
    import app as app_mod
    app_mod.app.config['TESTING'] = True
    yield app_mod
    _client.drop_database(_client.get_default_database().name)
    # remove files this test run created
    for folder in (services.UPLOAD_FOLDER, services.GENERATED_FOLDER):
        for name in os.listdir(folder):
            if 'pytest' in name:
                os.remove(os.path.join(folder, name))


@pytest.fixture()
def client(app_module):
    app_module.services.rate_limited.__globals__['_rate_hits'].clear()
    return app_module.app.test_client()


@pytest.fixture()
def admin(app_module):
    c = app_module.app.test_client()
    assert c.post('/api/admin/login', json={'email': 'admin@test.local', 'password': 'adminpass'}).get_json()['success']
    return c


def analyze(client, path, name='pytest.jpg'):
    with open(path, 'rb') as f:
        return client.post('/analyze', data={'image': (f, name)}, content_type='multipart/form-data')
