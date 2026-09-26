"""End-to-end API tests against the real models and a throwaway MongoDB."""
import io

from conftest import analyze, dataset_image


def test_pages_and_health(client):
    for path in ['/', '/map', '/user/login', '/user/signup', '/admin/login', '/static/js/script.js']:
        assert client.get(path).status_code == 200, path
    health = client.get('/api/health').get_json()
    assert health['database'] and health['model']


def test_auth_guards(client):
    assert client.get('/admin').status_code == 302
    assert client.get('/api/reports').status_code == 401
    assert client.get('/api/reports/export.csv').status_code == 401
    assert client.get('/api/user/reports').status_code == 401


def test_analyze_rejects_bad_input(client):
    r = client.post('/analyze', data={'image': (io.BytesIO(b'hello'), 'x.txt')}, content_type='multipart/form-data')
    assert r.status_code == 400
    r = client.post('/analyze', data={'image': (io.BytesIO(b'not an image'), 'pytest_fake.jpg')},
                    content_type='multipart/form-data')
    assert r.status_code == 400


def test_analyze_returns_probabilities_and_safe_filename(client):
    r = analyze(client, dataset_image('very_poor'), '../../pytest evil.jpg')
    d = r.get_json()
    assert r.status_code == 200, d
    assert '..' not in d['image_url'] and ' ' not in d['image_url']
    assert abs(sum(d['probabilities'].values()) - 100) < 0.5
    assert d['condition'] in d['probabilities']
    assert 0 <= d['road_health_index'] <= 100


def test_submit_duplicate_and_confirmation_flow(client, admin):
    d = analyze(client, dataset_image('very_poor', 1), 'pytest_a.jpg').get_json()
    payload = {'image_url': d['image_url'], 'email': 'first@example.com', 'name': '<script>x</script>',
               'location': {'address': 'MG Road', 'latitude': 26.9124, 'longitude': 75.7873}}
    r = client.post('/submit-report', json=payload)
    first = r.get_json()
    assert r.status_code == 200 and first['success'], first
    assert first['priority'] == 'High' and first['forecast']['roadHealthIndex'] <= 20

    # Same photo again -> rejected as reuse
    r = client.post('/submit-report', json=payload)
    assert r.status_code == 409 and r.get_json()['reason'] == 'same_image'

    # Client can't forge an unknown image
    r = client.post('/submit-report', json={**payload, 'image_url': '/static/uploads/nope.jpg'})
    assert r.status_code == 400

    reports = admin.get('/api/reports').get_json()['reports']
    assert any(x['id'] == first['reportId'] for x in reports)
    assert all('imageHash' not in x for x in reports)


def test_address_only_submission_is_geocoded(client):
    d = analyze(client, dataset_image('poor', 5), 'pytest_b.jpg').get_json()
    r = client.post('/submit-report', json={'image_url': d['image_url'], 'location': {'address': 'Tonk Road, Jaipur'}})
    assert r.status_code == 200, r.get_json()
    pub = client.get('/api/public/reports').get_json()['reports']
    assert any(x['address'] == 'Tonk Road, Jaipur' and x['latitude'] == 26.85 for x in pub)
    assert all('reporter' not in x for x in pub), 'public API must not leak reporter details'


def test_admin_workflow(admin):
    rid = admin.get('/api/reports').get_json()['reports'][0]['id']
    assert admin.post('/api/reports/bad-id/status', json={'status': 'Resolved'}).status_code == 400
    assert admin.post(f'/api/reports/{rid}/status', json={'status': 'Bogus'}).status_code == 400
    r = admin.post(f'/api/reports/{rid}/assign', json={'unit': 'PWD Alpha', 'scheduledFor': '2030-01-15'}).get_json()
    assert r['success'] and r['status'] == 'Scheduled'
    assert admin.post(f'/api/reports/{rid}/status', json={'status': 'Resolved', 'note': 'patched'}).get_json()['success']
    timeline = admin.get(f'/api/reports/{rid}/timeline').get_json()
    assert [m['status'] for m in timeline['milestones']][:2] == ['Resolved', 'Scheduled']
    assert len(timeline['assignments']) == 1

    stats = admin.get('/api/admin/stats').get_json()
    assert stats['by_status']['Resolved'] >= 1 and stats['avg_resolution_days'] is not None
    analytics = admin.get('/api/admin/analytics?days=14').get_json()
    assert len(analytics['labels']) == 14 and sum(analytics['resolved']) >= 1
    assert admin.get('/api/reports?q=MG%20Road').get_json()['reports']
    assert admin.get('/api/reports?start=garbage').status_code == 200


def test_csv_export(admin):
    r = admin.get('/api/reports/export.csv?status=Resolved')
    assert r.status_code == 200 and r.mimetype == 'text/csv'
    lines = r.get_data(as_text=True).strip().splitlines()
    assert lines[0].lstrip('﻿').startswith('id,created_at,status')
    assert all(',Resolved,' in line for line in lines[1:])


def test_delete_report(admin):
    before = admin.get('/api/reports').get_json()['reports']
    rid = before[-1]['id']
    assert admin.delete(f'/api/reports/{rid}').get_json()['success']
    after = admin.get('/api/reports').get_json()['reports']
    assert len(after) == len(before) - 1
    assert admin.delete(f'/api/reports/{rid}').status_code == 404


def test_user_signup_login(client):
    assert client.post('/api/user/signup', json={'email': 'u@x.com', 'password': '123'}).status_code == 400
    assert client.post('/api/user/signup', json={'email': 'U@X.com', 'password': 'secret1', 'name': 'U'}).get_json()['success']
    assert client.post('/api/user/login', json={'email': 'u@x.com', 'password': 'wrong'}).status_code == 401
    assert client.post('/api/user/login', json={'email': 'u@x.com', 'password': 'secret1'}).get_json()['success']
    assert client.get('/user/dashboard').status_code == 200


def test_login_rate_limited(client):
    codes = [client.post('/api/admin/login', json={'email': 'a@b.c', 'password': 'x'}).status_code for _ in range(12)]
    assert codes[-1] == 429


def test_share_image_and_traversal(client):
    assert client.post('/generate-shareable-image', json={'original_filename': '../../app.py'}).status_code == 404
    d = analyze(client, dataset_image('good'), 'pytest_share.jpg').get_json()
    s = client.post('/generate-shareable-image', json={'original_filename': d['original_filename'],
                                                       'condition': d['condition'], 'address': 'MG Road, Jaipur'}).get_json()
    assert s['success'] and '#RoadSafety' in s['post_text']
    assert client.get(s['shareable_image_url']).status_code == 200


def test_upload_restored_from_gridfs(client, app_module):
    import os
    d = analyze(client, dataset_image('poor', 9), 'pytest_persist.jpg').get_json()
    assert client.post('/submit-report', json={'image_url': d['image_url'], 'location': {'address': 'Persist Rd'}}).status_code == 200
    local = os.path.join(app_module.UPLOAD_FOLDER, os.path.basename(d['image_url']))
    os.remove(local)  # simulate a redeploy wiping the disk
    assert client.get(d['image_url']).status_code == 200
    assert os.path.isfile(local)


def _document_image(path):
    from PIL import Image, ImageDraw, ImageFont
    doc = Image.new('RGB', (900, 1200), 'white')
    d = ImageDraw.Draw(doc)
    for i in range(30):
        d.text((60, 60 + i * 36), f'Q{i + 1}. Explain the working of a transistor amplifier.', fill='black',
               font=ImageFont.load_default(size=26))
    doc.save(path)
    return path


def test_non_road_rejected_and_not_submittable(client, app_module, tmp_path):
    import os
    before = set(os.listdir(app_module.UPLOAD_FOLDER))
    r = analyze(client, _document_image(str(tmp_path / 'pytest_doc.jpg')), 'pytest_doc.jpg')
    assert r.status_code == 400 and 'road' in r.get_json()['error']
    leftover = set(os.listdir(app_module.UPLOAD_FOLDER)) - before
    assert not any('pytest_doc' in f for f in leftover), 'rejected upload must be deleted'


def test_unanalysed_upload_cannot_be_submitted(client, app_module):
    import os, shutil
    name = 'road_pytest_unanalysed.jpg'
    shutil.copy(dataset_image('very_poor', 20), os.path.join(app_module.UPLOAD_FOLDER, name))
    r = client.post('/submit-report', json={'image_url': f'/static/uploads/{name}', 'location': {'address': 'X Rd'}})
    assert r.status_code == 400


def test_borderline_photo_flagged_for_review(client, admin, app_module, monkeypatch):
    real = app_module.vision.analyze
    monkeypatch.setattr(app_module.vision, 'analyze', lambda p: {**real(p), 'road_status': 'review', 'road_probability': 0.3})
    d = analyze(client, dataset_image('poor', 30), 'pytest_border.jpg').get_json()
    assert d['success'] and d['road_check']['uncertain']
    rid = client.post('/submit-report', json={'image_url': d['image_url'],
                                               'location': {'address': 'Border Rd'}}).get_json()['reportId']
    flagged = admin.get('/api/reports?status=needs_review').get_json()['reports']
    assert [x['id'] for x in flagged] == [rid]
    admin.post(f'/api/reports/{rid}/status', json={'status': 'Scheduled'})
    assert admin.get('/api/reports?status=needs_review').get_json()['reports'] == []


def test_gemini_cooldown_after_rate_limit(app_module, monkeypatch):
    import requests

    class Resp:
        status_code = 429

        def raise_for_status(self):
            raise requests.HTTPError('429')
    calls = []
    monkeypatch.setattr(app_module, 'GEMINI_API_KEY', 'x')
    monkeypatch.setattr(app_module.requests, 'post', lambda *a, **k: calls.append(1) or Resp())
    monkeypatch.setattr(app_module, '_gemini_cooldown_until', 0.0)
    for _ in range(3):
        try:
            app_module.gemini_generate([{'text': 'hi'}])
        except Exception:
            pass
    assert len(calls) == 1, 'must stop calling Gemini during the cooldown'


def test_planets_and_other_non_roads_rejected(client):
    """Regression: a solar-system picture used to be rated a 'very poor road'."""
    import glob
    negatives = sorted(glob.glob('training/negatives/*.jpg'))
    if not negatives:
        import pytest
        pytest.skip('run training/make_negatives.py first')
    import services
    for path in negatives:
        services._rate_hits.clear()  # this test uploads more than the per-minute limit
        r = analyze(client, path, 'pytest_neg.jpg')
        assert r.status_code == 400, f'{path} was accepted as a road: {r.get_json()}'


def test_model_quality_recorded(app_module):
    m = app_module.vision.metrics
    assert m['test_accuracy'] >= 0.95
    assert m['gate']['non_road_images_passed'].startswith('0/')
