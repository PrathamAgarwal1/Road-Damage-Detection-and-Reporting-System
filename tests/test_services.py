"""Pure-function tests for services.py (no database needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services  # noqa: E402
from services import compute_health_forecast, compute_priority, hamming, haversine_m, normalize_condition  # noqa: E402


def test_normalize_condition():
    assert normalize_condition('Very Poor') == 'very_poor'
    assert normalize_condition(' very-poor ') == 'very_poor'
    assert services.severity_for('Good')['level'] == 'good'
    assert services.severity_for('nonsense')['level'] == 'moderate'


def test_haversine():
    assert haversine_m(0, 0, 0, 0) == 0
    # 0.001 deg of latitude is ~111 m
    assert 105 < haversine_m(26.0, 75.0, 26.001, 75.0) < 116


def test_hamming():
    assert hamming('ffff', 'ffff') == 0
    assert hamming('0000', '000f') == 4
    assert hamming(None, 'ff') == 64


def test_priority_critical_always_high():
    level, score = compute_priority(0.9, 0.0, 0, 0, 'critical')
    assert level == 'High' and score >= 0.9


def test_priority_crowd_evidence_raises_score():
    _, alone = compute_priority(0.4, 0.2, 0, 0, 'moderate')
    _, crowded = compute_priority(0.4, 0.2, 6, 3, 'moderate')
    assert crowded > alone


def test_health_forecast_bad_weather_is_faster():
    calm = compute_health_forecast(0.4, 0.0)
    stormy = compute_health_forecast(0.4, 1.0)
    assert stormy['daysToNextLevel'] < calm['daysToNextLevel']
    assert stormy['roadHealthIndex'] < calm['roadHealthIndex']
    assert calm['nextLevel'] == 'poor'


def test_health_forecast_critical_has_no_next_level():
    f = compute_health_forecast(0.9, 0.5)
    assert f['nextLevel'] is None and 'failure' in f['summary']


def test_rate_limiter():
    key = 'pytest-bucket'
    results = [services.rate_limited(key, 3, 60) for _ in range(5)]
    assert results == [False, False, False, True, True]


def test_image_hash_stable(tmp_path):
    from PIL import Image
    img = Image.linear_gradient('L').resize((64, 48))
    a, b = tmp_path / 'a.png', tmp_path / 'b.jpg'
    img.save(a)
    img.convert('RGB').save(b, quality=70)
    assert hamming(services.image_dhash(str(a)), services.image_dhash(str(b))) <= 4
