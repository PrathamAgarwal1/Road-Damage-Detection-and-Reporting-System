import os
import cv2
import numpy as np
import json
import secrets
import csv
import io
import re
import textwrap
import time
import uuid
from functools import wraps
from datetime import datetime, timedelta

from dotenv import load_dotenv

load_dotenv()

import services  # noqa: E402  (also makes console logging UTF-8 safe)
import requests
from flask import Flask, Response, request, jsonify, session, redirect, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont, ImageOps
from bson import ObjectId
from bson.errors import InvalidId

from services import (reports_col, assignments_col, users_col, milestones_col,
                      UPLOAD_FOLDER, GENERATED_FOLDER, BASE_DIR)

FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

# --- Configuration ---
app = Flask(__name__, static_folder=os.path.join(FRONTEND_DIR, 'static'), static_url_path='/static')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 15 * 1024 * 1024  # 15 MB uploads
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
# The Vercel frontend proxies API calls (same origin), so Lax cookies work everywhere.
# Secure cookies are enabled automatically on Render / when COOKIE_SECURE=true.
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = (os.getenv('COOKIE_SECURE', 'true' if os.getenv('RENDER') else 'false').lower() == 'true')

# Only allow credentialed cross-origin calls from explicitly configured origins
_origins = [o.strip() for o in os.getenv('ALLOWED_ORIGINS', '').split(',') if o.strip()]
if _origins:
    from flask_cors import CORS
    CORS(app, supports_credentials=True, origins=_origins)

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

services.ensure_indexes()

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
_gemini_cooldown_until = 0.0  # after a 429 we stop calling Gemini for a while instead of failing every request
if not GEMINI_API_KEY:
    print("⚠️ GEMINI_API_KEY not set — Gemini validation/descriptions disabled, using local fallbacks.")


def gemini_generate(parts, timeout=15):
    """Call Gemini and return the text response. Raises on failure."""
    global _gemini_cooldown_until
    if not GEMINI_API_KEY:
        raise RuntimeError('GEMINI_API_KEY not configured')
    if time.monotonic() < _gemini_cooldown_until:
        raise RuntimeError('Gemini rate-limited; cooling down')
    res = requests.post(GEMINI_API_URL,
                        headers={'Content-Type': 'application/json', 'x-goog-api-key': GEMINI_API_KEY},
                        data=json.dumps({'contents': [{'parts': parts}]}), timeout=timeout)
    if res.status_code == 429:
        _gemini_cooldown_until = time.monotonic() + 60
        print("⚠️ Gemini quota exceeded (429) — using the local road check for the next 60 s")
    res.raise_for_status()
    return res.json()['candidates'][0]['content']['parts'][0]['text'].strip()


# --- Vision model: road gate + condition classifier (see vision.py, training/train.py) ---
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'roadsight_clip.pt')
vision = None
try:
    from vision import RoadVision
    vision = RoadVision(MODEL_PATH)
    _m = vision.metrics
    print(f"✅ Vision model loaded ({_m.get('model')}, held-out test accuracy {_m.get('test_accuracy', 0):.1%})")
except Exception as _e:
    print(f"❌ MODEL ERROR: could not load {MODEL_PATH}: {_e}")
model_loaded = vision is not None


def analyze_image(image_path):
    """Road gate + condition for one photo (see RoadVision.analyze). None if the model fails."""
    try:
        return vision.analyze(image_path)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def predict_image(image_path):
    """(condition_key, confidence %) for a road photo."""
    result = analyze_image(image_path) if model_loaded else None
    return (result['condition'], result['confidence']) if result else (None, None)


# --- YOLOv8 segmentation (only for the road-surface visualisation overlay) ---
yolo_model = None
try:
    from ultralytics import YOLO as _YOLO
    yolo_model = _YOLO(os.path.join(BASE_DIR, 'yolov8n-seg.pt'))  # auto-downloads on first run (~6 MB)
    print("✅ YOLOv8-seg model loaded")
except Exception as _e:
    print(f"⚠️  YOLOv8 unavailable ({_e}). Run: pip install ultralytics")

# COCO class IDs that are NOT the road surface (vehicles, people, animals …)
_NON_ROAD_COCO = {0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                  24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 56, 57, 58, 59, 60}


def segment_road_region(image_path):
    """
    Visualise the road surface in *image_path* (does not affect the classification):
      1. Perspective-aware trapezoid mask  (road ≈ lower portion of scene)
      2. Asphalt colour filter             (low-saturation, mid-to-dark brightness)
      3. YOLOv8-seg object subtraction     (erase vehicles / people)

    Returns (viz_filename | None, road_coverage_pct, road_found)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0, False
        h, w = img.shape[:2]

        # --- 1. Perspective trapezoid ---
        top_ratio = 0.28           # road width fraction at the horizon
        horizon_y = int(h * 0.38)  # row where road perspective vanishes
        pts = np.array([
            [int(w * 0.01), h - 1],
            [int(w * 0.99), h - 1],
            [int(w * (0.5 + top_ratio / 2)), horizon_y],
            [int(w * (0.5 - top_ratio / 2)), horizon_y],
        ], np.int32)
        persp = np.zeros((h, w), np.uint8)
        cv2.fillPoly(persp, [pts], 255)

        # --- 2. Asphalt colour profile (HSV) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, (0, 0, 15), (179, 68, 220))
        road_mask = cv2.bitwise_and(persp, color_mask)

        # Morphological cleanup
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

        # --- 3. YOLO object subtraction ---
        if yolo_model is not None:
            try:
                res = yolo_model.predict(image_path, verbose=False)[0]
                if res.masks is not None:
                    for cls_id, raw_mask in zip(res.boxes.cls.cpu().numpy().astype(int),
                                                res.masks.data.cpu().numpy()):
                        if int(cls_id) in _NON_ROAD_COCO:
                            obj = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                            road_mask[obj > 0.5] = 0
            except Exception as ye:
                print(f"⚠️  YOLO inference skipped: {ye}")

        coverage = float(np.sum(road_mask > 0)) / (h * w) * 100
        road_found = coverage >= 5.0
        if not road_found:
            # Colour mask failed on a confirmed road photo: outline the lower frame instead
            road_mask[:] = 0
            road_mask[int(h * 0.45):, :] = 255
            coverage = 0.0   # report 0 so UI can warn the user

        # --- Build visualisation overlay ---
        viz = img.copy()
        dimmed = (img * 0.25).astype(np.uint8)
        non_road = cv2.bitwise_not(road_mask)
        viz[non_road > 0] = dimmed[non_road > 0]
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, contours, -1, (0, 220, 80), 3)

        base = os.path.splitext(os.path.basename(image_path))[0]
        viz_name = f"seg_{base}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, viz_name), viz)
        return viz_name, round(coverage, 1), road_found
    except Exception as e:
        print(f"❌ segment_road_region error: {e}")
        return None, 0.0, False


def normalize_upload(path, max_side=1600):
    """Apply EXIF rotation and downscale huge phone photos (faster inference, smaller storage)."""
    with Image.open(path) as img:
        fmt = img.format
        fixed = ImageOps.exif_transpose(img)
        needs_resize = max(fixed.size) > max_side
        if not needs_resize and fixed is img:
            return
        if needs_resize:
            fixed.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if fmt == 'JPEG' or fixed.mode not in ('RGB', 'RGBA', 'L'):
            fixed = fixed.convert('RGB')
        fixed.save(path, format=fmt or 'JPEG', **({'quality': 90} if fmt in (None, 'JPEG', 'WEBP') else {}))


def _exif_rational(v):
    try:
        return float(v)
    except TypeError:
        return v[0] / v[1] if v[1] else 0.0


def extract_exif(image_path):
    """Return (lat, lon, captured_at_iso) from the photo's EXIF metadata when present."""
    lat = lon = captured = None
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            gps = exif.get_ifd(0x8825)
            if gps and 2 in gps and 4 in gps:
                def dms(vals, ref):
                    d, m, s = (_exif_rational(x) for x in vals)
                    val = d + m / 60 + s / 3600
                    return -val if ref in ('S', 'W') else val
                lat = dms(gps[2], gps.get(1, 'N'))
                lon = dms(gps[4], gps.get(3, 'E'))
                lat, lon = services.valid_coords(lat, lon)
                if lat == 0 and lon == 0:
                    lat = lon = None
            raw_time = exif.get_ifd(0x8769).get(36867) or exif.get(306)
            if raw_time:
                captured = datetime.strptime(str(raw_time).strip(), '%Y:%m:%d %H:%M:%S').isoformat()
    except Exception:
        pass
    return lat, lon, captured


def serialize_doc(doc):
    doc = dict(doc)
    doc['id'] = str(doc.pop('_id'))
    for k, v in list(doc.items()):
        if isinstance(v, ObjectId):
            doc[k] = str(v)
    return doc


def parse_oid(rid):
    try:
        return ObjectId(rid)
    except (InvalidId, TypeError):
        return None


def upload_path_from_url(image_url):
    """Map '/static/uploads/<name>' to a safe local path inside UPLOAD_FOLDER (or None)."""
    if not image_url or not image_url.startswith('/static/uploads/'):
        return None
    name = secure_filename(os.path.basename(image_url))
    path = os.path.join(UPLOAD_FOLDER, name)
    return path if name and os.path.isfile(path) else None


# --- Pages & static files ---
def page(name):
    return send_from_directory(FRONTEND_DIR, name)


@app.route('/')
def index():
    return page('index.html')


@app.route('/map')
def public_map():
    return page('map.html')


@app.route('/static/uploads/<path:filename>')
def uploaded_file(filename):
    name = secure_filename(os.path.basename(filename))
    services.restore_image(name, UPLOAD_FOLDER)  # re-hydrate from GridFS after a redeploy
    return send_from_directory(UPLOAD_FOLDER, name)


@app.route('/api/health')
def health():
    try:
        services.mongo_client.admin.command('ping')
        db_ok = True
    except Exception:
        db_ok = False
    return jsonify({'success': db_ok and model_loaded, 'database': db_ok, 'model': model_loaded,
                    'yolo': yolo_model is not None, 'gemini': bool(GEMINI_API_KEY),
                    'email': services.smtp_configured()}), (200 if db_ok and model_loaded else 503)


def client_ip():
    fwd = request.headers.get('X-Forwarded-For', '')
    return fwd.split(',')[0].strip() if fwd else (request.remote_addr or '?')


def too_many(bucket, limit, window_s=60):
    if services.rate_limited(f'{bucket}:{client_ip()}', limit, window_s):
        return jsonify({'success': False, 'error': 'Too many requests — please wait a minute and try again.'}), 429
    return None


@app.route('/static/generated/<path:filename>')
def generated_file(filename):
    return send_from_directory(GENERATED_FOLDER, filename)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({'success': False, 'error': 'Image is too large (max 15 MB).'}), 413


# --- Analysis ---
@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({'success': False, 'error': 'Unsupported file type. Please upload a JPG, PNG or WEBP image.'}), 400
    if not model_loaded:
        return jsonify({'success': False, 'error': 'Classification model is not available on the server.'}), 503
    limited = too_many('analyze', 15)
    if limited:
        return limited
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        stem = secure_filename(os.path.splitext(file.filename)[0])[:40] or 'photo'
        filename = f"road_{timestamp}_{uuid.uuid4().hex[:6]}_{stem}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        try:
            with Image.open(filepath) as probe:
                probe.verify()
        except Exception:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'The uploaded file is not a valid image.'}), 400
        exif_lat, exif_lon, captured_at = extract_exif(filepath)  # before re-encoding strips EXIF
        normalize_upload(filepath)

        # --- Stage 1 + 2: road gate and condition from one MobileCLIP2 pass ---
        result = analyze_image(filepath)
        if result is None:
            os.remove(filepath)
            return jsonify({'success': False, 'error': 'Failed to analyze image'}), 500
        road_check = {'is_road': result['road_status'] != 'not_road', 'uncertain': result['road_status'] == 'review',
                      'method': 'clip', 'score': result['road_probability']}
        if not road_check['is_road']:
            os.remove(filepath)  # never keep (or allow submitting) photos that failed the road check
            print(f"🚫 Rejected as not a road (P(road)={result['road_probability']})")
            return jsonify({'success': False, 'error': 'This photo does not appear to show a road or street. '
                                                       'Please upload a photo of the damaged road.'}), 400
        probs = result['probabilities']
        condition, confidence = result['condition'], result['confidence']

        # --- Stage 3: road-surface visualisation (YOLOv8 + colour filter) ---
        viz_name, road_coverage, road_found = segment_road_region(filepath)

        severity = services.severity_for(condition)
        severity['icon'] = {'good': 'check-circle', 'moderate': 'exclamation-triangle',
                            'poor': 'exclamation-circle', 'critical': 'times-circle'}[severity['level']]
        forecast = services.compute_health_forecast(severity['score'], 0.0)
        services.db.get_collection('analyses').insert_one({
            'filename': filename, 'roadCheck': road_check, 'condition': condition,
            'confidence': round(confidence, 2), 'createdAt': datetime.utcnow()})
        return jsonify({
            'success':           True,
            'condition':         condition.replace('_', ' ').title(),
            'confidence':        round(confidence, 2),
            'uncertain':         confidence < 55,
            'probabilities':     {k.replace('_', ' ').title(): v for k, v in probs.items()},
            'severity':          severity,
            'road_health_index': forecast['roadHealthIndex'],
            'image_url':         f'/static/uploads/{filename}',
            'segmented_url':     f'/static/uploads/{viz_name}' if viz_name else None,
            'road_coverage':     road_coverage,
            'road_found':        road_found,
            'road_check':        road_check,
            'original_filename': filename,
            'exif_location':     {'latitude': exif_lat, 'longitude': exif_lon} if exif_lat is not None else None,
            'captured_at':       captured_at,
        })
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': 'Failed to analyze image'}), 500


@app.route('/api/geocode/reverse', methods=['GET'])
def api_reverse_geocode():
    lat, lon = services.valid_coords(request.args.get('lat'), request.args.get('lon'))
    if lat is None:
        return jsonify({'success': False, 'error': 'Invalid coordinates'}), 400
    address = services.reverse_geocode(lat, lon)
    return jsonify({'success': bool(address), 'address': address})


# --- Report submission (duplicate filtering, weather forecasting, priority, alerts) ---
@app.route('/submit-report', methods=['POST'])
def submit_report():
    limited = too_many('submit', 8)
    if limited:
        return limited
    data = request.get_json(silent=True) or {}
    now = datetime.utcnow()
    location = data.get('location') or {}
    address = (location.get('address') or '').strip()[:300]
    latitude, longitude = services.valid_coords(location.get('latitude'), location.get('longitude'))
    description = (data.get('description') or '').strip()[:2000]

    if not address and latitude is None:
        return jsonify({'success': False, 'error': 'A location address or GPS position is required.'}), 400

    image_path = upload_path_from_url(data.get('image_url'))
    analysis = services.db.get_collection('analyses').find_one(
        {'filename': os.path.basename(image_path)}) if image_path else None
    if not image_path or not analysis:
        return jsonify({'success': False, 'error': 'Please analyze a photo before submitting a report.'}), 400
    road_check = analysis.get('roadCheck') or {}
    image_url = f"/static/uploads/{os.path.basename(image_path)}"

    # Never trust client-sent classification: re-run the classifier on the stored image
    condition_key, confidence = predict_image(image_path)
    if condition_key is None:
        return jsonify({'success': False, 'error': 'Failed to verify the image.'}), 500
    severity_info = services.severity_for(condition_key)

    if session.get('user'):
        reporter_email = session['user'].get('email')
        reporter_name = session['user'].get('name') or 'Anonymous'
    else:
        reporter_email = (data.get('email') or '').strip().lower()[:200] or 'anonymous@citizen.local'
        reporter_name = (data.get('name') or '').strip()[:100] or 'Anonymous'

    # Automatic geotagging: resolve missing coordinates from the address
    if latitude is None and address:
        latitude, longitude = services.geocode_address(address)
    if not address and latitude is not None:
        address = services.reverse_geocode(latitude, longitude) or f'{latitude:.5f}, {longitude:.5f}'

    # --- Duplicate & fraud filtering ---
    image_hash = services.image_dhash(image_path)
    dup, reason = services.find_duplicate(image_hash, latitude, longitude, reporter_email, now)
    if dup:
        if reason == 'similar_image_nearby' and reporter_email not in services.ANONYMOUS_EMAILS \
                and reporter_email != (dup.get('reporter') or {}).get('email'):
            # An independent citizen confirmed the same damage — strengthen the original report
            reports_col.update_one({'_id': dup['_id']}, {'$addToSet': {'confirmedBy': reporter_email},
                                                          '$set': {'updatedAt': now}})
            services.refresh_priority(dup['_id'])
        reports_col.database.get_collection('rejected_reports').insert_one({
            'reason': reason, 'duplicateOf': dup['_id'], 'imageUrl': image_url, 'imageHash': image_hash,
            'reporterEmail': reporter_email, 'latitude': latitude, 'longitude': longitude, 'createdAt': now})
        print(f"🚫 Duplicate report rejected ({reason}) — original {dup['_id']}")
        return jsonify({'success': False, 'duplicate': True, 'reason': reason,
                        'error': services.DUPLICATE_MESSAGES[reason],
                        'duplicateOf': str(dup['_id'])}), 409

    # --- Weather-driven risk, road health forecast, crowd density, priority ---
    weather = services.compute_weather_risk(latitude, longitude)
    risk = weather['riskScore']
    forecast = services.compute_health_forecast(severity_info['score'], risk)
    density = services.compute_density(latitude, longitude)
    priority, score = services.compute_priority(severity_info['score'], risk, density, 0, severity_info['level'])

    doc = {
        'imageUrl': image_url,
        'imageHash': image_hash,
        'location': {
            'address': address,
            'latitude': latitude,
            'longitude': longitude,
            'geo': {'type': 'Point', 'coordinates': [longitude, latitude]} if latitude is not None else None,
        },
        'category': 'RoadDamage',
        'condition': condition_key.replace('_', ' ').title(),
        'confidence': round(confidence, 2),
        'severity': severity_info,
        'weather': weather,
        'predictiveRisk': risk,
        'forecast': forecast,
        'reportDensity': density,
        'confirmedBy': [],
        'priority': priority,
        'priorityScore': score,
        'status': 'New',
        'reporter': {'name': reporter_name, 'email': reporter_email},
        'description': description,
        'capturedAt': data.get('captured_at'),
        'roadCheck': road_check,
        'needsReview': bool(road_check.get('uncertain')),
        'createdAt': now,
        'updatedAt': now,
    }
    services.persist_image(image_path)
    inserted = reports_col.insert_one(doc)
    services.add_milestone(inserted.inserted_id, 'New', 'Report submitted',
                           f"Classified as {doc['condition']} ({doc['confidence']}% confidence). {forecast['summary']}.",
                           when=now)
    services.refresh_neighbors(latitude, longitude, exclude_id=inserted.inserted_id)

    # Real-time alert to authorities for high-priority damage
    if priority == 'High' and services.ALERT_EMAILS:
        maps = f"https://www.openstreetmap.org/?mlat={latitude}&mlon={longitude}#map=18/{latitude}/{longitude}" \
            if latitude is not None else 'n/a'
        services.send_email_async(
            services.ALERT_EMAILS,
            f"[RoadSight] HIGH priority road damage: {address}",
            f"A high-priority road damage report was submitted.\n\n"
            f"Location: {address}\nMap: {maps}\nCondition: {doc['condition']} ({doc['confidence']}%)\n"
            f"Road Health Index: {forecast['roadHealthIndex']}/100\nForecast: {forecast['summary']}\n"
            f"Weather risk: {risk:.2f}\nPriority score: {score:.2f}\nPhoto: {image_url}\n"
            f"Report ID: {inserted.inserted_id}\n")

    print(f"📋 New report {inserted.inserted_id}: {doc['condition']} at {address} — priority {priority} ({score})")
    return jsonify({'success': True, 'message': 'Report submitted successfully!',
                    'reportId': str(inserted.inserted_id), 'priority': priority,
                    'forecast': forecast, 'weather': {k: v for k, v in weather.items() if k != 'fetchedAt'}})


# --- Auth Helpers ---
def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('admin'):
            return redirect('/admin/login')
        return fn(*args, **kwargs)
    return wrapper


def require_admin_api(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('admin'):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        return fn(*args, **kwargs)
    return wrapper


def require_user(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return redirect('/user/login')
        return fn(*args, **kwargs)
    return wrapper


def require_user_api(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        return fn(*args, **kwargs)
    return wrapper


# Seed the admin account. An explicitly configured password always wins so rotating it works.
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@example.com').strip().lower()
_admin_hash = os.getenv('ADMIN_PASSWORD_HASH') or (
    generate_password_hash(os.getenv('ADMIN_PASSWORD')) if os.getenv('ADMIN_PASSWORD') else None)
try:
    if _admin_hash:
        users_col.update_one({'email': ADMIN_EMAIL},
                             {'$set': {'passwordHash': _admin_hash, 'role': 'admin'},
                              '$setOnInsert': {'email': ADMIN_EMAIL, 'createdAt': datetime.utcnow()}}, upsert=True)
    else:
        print("⚠️ ADMIN_PASSWORD not set — seeding default admin password 'admin123'. Set it in production!")
        users_col.update_one({'email': ADMIN_EMAIL}, {'$setOnInsert': {
            'email': ADMIN_EMAIL, 'passwordHash': generate_password_hash('admin123'),
            'role': 'admin', 'createdAt': datetime.utcnow()}}, upsert=True)
except Exception as e:
    print(f"⚠️ Could not seed admin user: {e}")


def _credentials():
    data = request.get_json(silent=True) or request.form or {}
    return (data.get('email') or '').strip().lower(), data.get('password') or '', (data.get('name') or '').strip()[:100]


# --- User Authentication ---
@app.route('/user/signup', methods=['GET'])
def user_signup_view():
    return page('user_signup.html')


@app.route('/api/user/signup', methods=['POST'])
def user_signup():
    limited = too_many('signup', 5)
    if limited:
        return limited
    email, password, name = _credentials()
    if not email or not password:
        return jsonify({'success': False, 'error': 'Email and password are required'}), 400
    if '@' not in email or len(email) > 200:
        return jsonify({'success': False, 'error': 'Please enter a valid email address'}), 400
    if len(password) < 6:
        return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
    if users_col.find_one({'email': email}):
        return jsonify({'success': False, 'error': 'Email already registered'}), 400
    users_col.insert_one({'email': email, 'passwordHash': generate_password_hash(password),
                          'name': name, 'role': 'user', 'createdAt': datetime.utcnow()})
    session.permanent = True
    session['user'] = {'email': email, 'name': name, 'role': 'user'}
    return jsonify({'success': True, 'user': session['user']})


@app.route('/user/login', methods=['GET'])
def user_login_view():
    return page('user_login.html')


@app.route('/api/user/login', methods=['POST'])
def user_login():
    limited = too_many('login', 10)
    if limited:
        return limited
    email, password, _ = _credentials()
    user = users_col.find_one({'email': email, 'role': 'user'})
    if user and check_password_hash(user.get('passwordHash', ''), password):
        session.permanent = True
        session['user'] = {'email': user['email'], 'name': user.get('name', ''), 'role': 'user'}
        return jsonify({'success': True, 'user': session['user']})
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401


@app.route('/user/logout')
@app.route('/api/user/logout')
def user_logout():
    session.pop('user', None)
    if request.headers.get('Accept') == 'application/json':
        return jsonify({'success': True})
    return redirect('/')


@app.route('/user/dashboard')
@require_user
def user_dashboard():
    return page('user_dashboard.html')


# --- Admin Views ---
@app.route('/admin/login', methods=['GET'])
def admin_login_view():
    return page('admin_login.html')


@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    limited = too_many('login', 10)
    if limited:
        return limited
    email, password, _ = _credentials()
    user = users_col.find_one({'email': email, 'role': 'admin'})
    if user and check_password_hash(user.get('passwordHash', ''), password):
        session.permanent = True
        session['admin'] = {'email': user['email'], 'role': 'admin'}
        return jsonify({'success': True, 'admin': session['admin']})
    return jsonify({'success': False, 'error': 'Invalid credentials'}), 401


@app.route('/admin/logout')
@app.route('/api/admin/logout')
def admin_logout():
    session.pop('admin', None)
    if request.headers.get('Accept') == 'application/json':
        return jsonify({'success': True})
    return redirect('/admin/login')


@app.route('/admin')
@require_admin
def admin_dashboard():
    return page('admin.html')


@app.route('/api/admin/check', methods=['GET'])
def api_admin_check():
    admin = session.get('admin')
    return jsonify({'success': bool(admin), 'admin': admin})


# --- Admin/Reports APIs ---
VALID_STATUSES = ['New', 'Scheduled', 'In Progress', 'Resolved']


def _parse_date(value, end_of_day=False):
    try:
        d = datetime.fromisoformat(value)
        if end_of_day and len(value) <= 10:
            d += timedelta(days=1) - timedelta(microseconds=1)
        return d
    except (TypeError, ValueError):
        return None


def _report_query():
    q = {}
    severity = request.args.get('severity')
    status = request.args.get('status')
    priority = request.args.get('priority')
    if severity:
        q['severity.level'] = severity
    if status == 'needs_review':
        q['needsReview'] = True
    elif status:
        q['status'] = status
    if priority:
        q['priority'] = priority
    start = _parse_date(request.args.get('start'))
    end = _parse_date(request.args.get('end'), end_of_day=True)
    if start or end:
        q['createdAt'] = {}
        if start:
            q['createdAt']['$gte'] = start
        if end:
            q['createdAt']['$lte'] = end
    text = (request.args.get('q') or '').strip()[:100]
    if text:
        rx = {'$regex': re.escape(text), '$options': 'i'}
        q['$or'] = [{'location.address': rx}, {'reporter.name': rx}, {'reporter.email': rx},
                    {'description': rx}, {'assignedUnit': rx}]
    return q


def _report_sort():
    sort = request.args.get('sort')
    if sort in ('newest_only', 'oldest_only'):
        return [('createdAt', -1 if sort == 'newest_only' else 1)]
    return [('priorityScore', -1), ('createdAt', 1 if sort == 'oldest' else -1)]


@app.route('/api/reports', methods=['GET'])
@require_admin_api
def api_reports_list():
    docs = [serialize_doc(d) for d in reports_col.find(_report_query(), {'imageHash': 0}).sort(_report_sort())]
    return jsonify({'success': True, 'reports': docs})


@app.route('/api/reports/export.csv', methods=['GET'])
@require_admin_api
def api_reports_export():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(['id', 'created_at', 'status', 'priority', 'priority_score', 'condition', 'confidence',
                'road_health_index', 'forecast', 'weather_risk', 'nearby_reports', 'confirmations',
                'address', 'latitude', 'longitude', 'assigned_unit', 'scheduled_for', 'resolved_at',
                'reporter_name', 'reporter_email', 'description'])
    for d in reports_col.find(_report_query(), {'imageHash': 0}).sort(_report_sort()):
        loc, f = d.get('location') or {}, d.get('forecast') or {}
        w.writerow([d['_id'], d.get('createdAt'), d.get('status'), d.get('priority'), d.get('priorityScore'),
                    d.get('condition'), d.get('confidence'), f.get('roadHealthIndex'), f.get('summary'),
                    d.get('predictiveRisk'), d.get('reportDensity'), len(d.get('confirmedBy') or []),
                    loc.get('address'), loc.get('latitude'), loc.get('longitude'), d.get('assignedUnit'),
                    d.get('scheduledFor'), d.get('resolvedAt'), (d.get('reporter') or {}).get('name'),
                    (d.get('reporter') or {}).get('email'), d.get('description')])
    name = f"roadsight_reports_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    return Response('\ufeff' + buf.getvalue(), mimetype='text/csv',
                    headers={'Content-Disposition': f'attachment; filename={name}'})


@app.route('/api/reports/<rid>', methods=['DELETE'])
@require_admin_api
def api_delete_report(rid):
    """Remove spam / invalid reports (kept in the rejected log for auditing)."""
    oid = parse_oid(rid)
    if not oid:
        return jsonify({'success': False, 'error': 'Invalid report id'}), 400
    report = reports_col.find_one({'_id': oid})
    if not report:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    reports_col.database.get_collection('rejected_reports').insert_one({
        'reason': 'removed_by_admin', 'removedBy': session['admin']['email'], 'report': report,
        'createdAt': datetime.utcnow()})
    reports_col.delete_one({'_id': oid})
    milestones_col.delete_many({'reportId': oid})
    assignments_col.delete_many({'reportId': oid})
    loc = report.get('location') or {}
    services.refresh_neighbors(loc.get('latitude'), loc.get('longitude'))
    return jsonify({'success': True})


@app.route('/api/reports/<rid>/timeline', methods=['GET'])
@require_admin_api
def api_report_timeline(rid):
    oid = parse_oid(rid)
    if not oid:
        return jsonify({'success': False, 'error': 'Invalid report id'}), 400
    milestones = [serialize_doc(m) for m in milestones_col.find({'reportId': oid}).sort([('createdAt', -1)])]
    assignments = [serialize_doc(a) for a in assignments_col.find({'reportId': oid}).sort([('createdAt', -1)])]
    return jsonify({'success': True, 'milestones': milestones, 'assignments': assignments})


def _notify_reporter(report, new_status, note=''):
    email = (report.get('reporter') or {}).get('email')
    if not email or email in services.ANONYMOUS_EMAILS:
        return
    address = (report.get('location') or {}).get('address', 'Unknown')
    body = (f"Hello {(report.get('reporter') or {}).get('name') or ''},\n\n"
            f"Your reported road issue at {address} is now '{new_status}'.\n"
            + (f"\nUpdate: {note}\n" if note else '')
            + ("\nThe repair is complete — thank you for helping improve our roads!\n" if new_status == 'Resolved'
               else "\nThank you for helping improve our roads.\n"))
    services.send_email_async(email, f"Your RoadSight report status is now: {new_status}", body)


@app.route('/api/reports/<rid>/status', methods=['POST'])
@require_admin_api
def api_update_status(rid):
    oid = parse_oid(rid)
    if not oid:
        return jsonify({'success': False, 'error': 'Invalid report id'}), 400
    data = request.get_json(silent=True) or {}
    new_status = data.get('status')
    note = (data.get('note') or '').strip()[:500]
    if new_status not in VALID_STATUSES:
        return jsonify({'success': False, 'error': 'Invalid status'}), 400
    report = reports_col.find_one({'_id': oid})
    if not report:
        return jsonify({'success': False, 'error': 'Not found'}), 404

    old_status = report.get('status', 'New')
    if old_status == new_status and not note:
        return jsonify({'success': True, 'unchanged': True})
    now = datetime.utcnow()
    update = {'status': new_status, 'updatedAt': now}
    if new_status == 'Resolved':
        update['resolvedAt'] = now
    if new_status != 'New':
        update['needsReview'] = False  # an admin acted on it
    reports_col.update_one({'_id': oid}, {'$set': update})
    services.add_milestone(oid, new_status,
                           f'Status updated: {old_status} → {new_status}' if old_status != new_status else f'Update: {new_status}',
                           note or f'Road repair status changed to {new_status}',
                           previous_status=old_status, created_by=session['admin']['email'])
    if old_status != new_status:
        loc = report.get('location') or {}
        services.refresh_neighbors(loc.get('latitude'), loc.get('longitude'))
        _notify_reporter(report, new_status, note)
    return jsonify({'success': True})


@app.route('/api/reports/<rid>/assign', methods=['POST'])
@require_admin_api
def api_assign_task(rid):
    oid = parse_oid(rid)
    if not oid:
        return jsonify({'success': False, 'error': 'Invalid report id'}), 400
    data = request.get_json(silent=True) or {}
    unit = (data.get('unit') or '').strip()[:100]
    if not unit:
        return jsonify({'success': False, 'error': 'unit required'}), 400
    scheduled_for = _parse_date(data.get('scheduledFor')) if data.get('scheduledFor') else None
    note = (data.get('note') or '').strip()[:500] or f'Assigned to {unit}' + (
        f", scheduled for {scheduled_for.strftime('%d %b %Y')}" if scheduled_for else '')

    report = reports_col.find_one({'_id': oid})
    if not report:
        return jsonify({'success': False, 'error': 'Report not found'}), 404

    now = datetime.utcnow()
    assignments_col.insert_one({'reportId': oid, 'unit': unit, 'note': note, 'scheduledFor': scheduled_for,
                                'createdAt': now, 'createdBy': session['admin']['email']})
    old_status = report.get('status', 'New')
    new_status = old_status if old_status in ('In Progress', 'Resolved') else 'Scheduled'
    reports_col.update_one({'_id': oid}, {'$set': {'status': new_status, 'assignedUnit': unit,
                                                   'scheduledFor': scheduled_for, 'updatedAt': now}})
    services.add_milestone(oid, new_status, f'Assigned to Field Unit: {unit}', note,
                           previous_status=old_status, created_by=session['admin']['email'])
    if new_status != old_status:
        _notify_reporter(report, new_status, note)
    return jsonify({'success': True, 'status': new_status})


@app.route('/api/admin/stats', methods=['GET'])
@require_admin_api
def api_admin_stats():
    def counts(field, values):
        return {v: reports_col.count_documents({field: v}) for v in values}
    return jsonify({
        'success': True,
        'total': reports_col.count_documents({}),
        'by_priority': counts('priority', ['High', 'Medium', 'Low']),
        'by_status': counts('status', VALID_STATUSES),
        'by_severity': counts('severity.level', ['critical', 'poor', 'moderate', 'good']),
        'needs_review': reports_col.count_documents({'needsReview': True}),
        'duplicates_blocked': reports_col.database.get_collection('rejected_reports').count_documents(
            {'reason': {'$ne': 'removed_by_admin'}}),
        'avg_resolution_days': _avg_resolution_days(),
        'hotspots': len(services.compute_hotspots()),
    })


def _avg_resolution_days():
    res = list(reports_col.aggregate([
        {'$match': {'status': 'Resolved', 'resolvedAt': {'$type': 'date'}, 'createdAt': {'$type': 'date'}}},
        {'$group': {'_id': None, 'avg': {'$avg': {'$subtract': ['$resolvedAt', '$createdAt']}}}}]))
    return round(res[0]['avg'] / 86400000, 1) if res and res[0]['avg'] is not None else None


@app.route('/api/admin/analytics', methods=['GET'])
@require_admin_api
def api_admin_analytics():
    """Daily reported / resolved counts for the last N days plus condition and status breakdowns."""
    try:
        days = min(max(int(request.args.get('days', 30)), 7), 365)
    except ValueError:
        days = 30
    start = (datetime.utcnow() - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)

    def per_day(field):
        rows = reports_col.aggregate([
            {'$match': {field: {'$gte': start}}},
            {'$group': {'_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$' + field}}, 'n': {'$sum': 1}}}])
        return {r['_id']: r['n'] for r in rows}

    reported, resolved = per_day('createdAt'), per_day('resolvedAt')
    labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    by_condition = {r['_id'] or 'unknown': r['n'] for r in reports_col.aggregate(
        [{'$group': {'_id': '$severity.level', 'n': {'$sum': 1}}}])}
    by_status = {r['_id'] or 'unknown': r['n'] for r in reports_col.aggregate(
        [{'$group': {'_id': '$status', 'n': {'$sum': 1}}}])}
    avg_rhi = list(reports_col.aggregate([
        {'$match': {'status': {'$ne': 'Resolved'}, 'forecast.roadHealthIndex': {'$type': 'number'}}},
        {'$group': {'_id': None, 'v': {'$avg': '$forecast.roadHealthIndex'}}}]))
    return jsonify({'success': True, 'labels': labels,
                    'reported': [reported.get(d, 0) for d in labels],
                    'resolved': [resolved.get(d, 0) for d in labels],
                    'by_condition': by_condition, 'by_status': by_status,
                    'avg_open_rhi': round(avg_rhi[0]['v']) if avg_rhi else None})


# --- Public (anonymised) community map ---
@app.route('/api/public/reports', methods=['GET'])
def api_public_reports():
    docs = reports_col.find({'location.latitude': {'$type': 'number'}},
                            {'location.address': 1, 'location.latitude': 1, 'location.longitude': 1,
                             'condition': 1, 'severity.level': 1, 'status': 1, 'priority': 1,
                             'forecast.roadHealthIndex': 1, 'forecast.summary': 1, 'imageUrl': 1,
                             'confirmedBy': 1, 'createdAt': 1, 'resolvedAt': 1}).sort([('createdAt', -1)]).limit(2000)
    out = []
    for d in docs:
        loc = d.get('location') or {}
        out.append({'id': str(d['_id']), 'address': loc.get('address'), 'latitude': loc.get('latitude'),
                    'longitude': loc.get('longitude'), 'condition': d.get('condition'),
                    'severity': (d.get('severity') or {}).get('level'), 'status': d.get('status'),
                    'priority': d.get('priority'), 'imageUrl': d.get('imageUrl'),
                    'roadHealthIndex': (d.get('forecast') or {}).get('roadHealthIndex'),
                    'forecast': (d.get('forecast') or {}).get('summary'),
                    'confirmations': len(d.get('confirmedBy') or []),
                    'createdAt': d.get('createdAt'), 'resolvedAt': d.get('resolvedAt')})
    return jsonify({'success': True, 'reports': out})


@app.route('/api/public/stats', methods=['GET'])
def api_public_stats():
    return jsonify({'success': True, 'total': reports_col.count_documents({}),
                    'resolved': reports_col.count_documents({'status': 'Resolved'}),
                    'in_progress': reports_col.count_documents({'status': {'$in': ['Scheduled', 'In Progress']}}),
                    'avg_resolution_days': _avg_resolution_days()})


@app.route('/api/admin/hotspots', methods=['GET'])
@require_admin_api
def api_admin_hotspots():
    try:
        radius = min(max(int(request.args.get('radius', services.HOTSPOT_RADIUS_M)), 50), 5000)
    except ValueError:
        radius = services.HOTSPOT_RADIUS_M
    return jsonify({'success': True, 'hotspots': services.compute_hotspots(radius_m=radius)})


# --- User APIs ---
@app.route('/api/user/reports', methods=['GET'])
@require_user_api
def api_user_reports():
    user_email = session['user']['email']
    docs = [serialize_doc(d) for d in reports_col.find({'reporter.email': user_email}, {'imageHash': 0})
            .sort([('createdAt', -1)])]
    return jsonify({'success': True, 'reports': docs})


@app.route('/api/user/check', methods=['GET'])
def api_user_check():
    user = session.get('user')
    return jsonify({'success': bool(user), 'user': user})


@app.route('/api/user/milestones/<rid>', methods=['GET'])
@require_user_api
def api_user_milestones(rid):
    oid = parse_oid(rid)
    if not oid:
        return jsonify({'success': False, 'error': 'Invalid report id'}), 400
    report = reports_col.find_one({'_id': oid, 'reporter.email': session['user']['email']})
    if not report:
        return jsonify({'success': False, 'error': 'Report not found'}), 404
    milestones = [serialize_doc(m) for m in milestones_col.find({'reportId': oid}).sort([('createdAt', -1)])]
    return jsonify({'success': True, 'milestones': milestones})


# --- Gemini helpers ---
@app.route('/generate-description', methods=['POST'])
def generate_description():
    data = request.get_json(silent=True) or {}
    condition = str(data.get('condition') or 'damaged')[:40]
    address = str(data.get('address') or 'an unspecified location')[:300]
    prompt = (f"You are an official at a municipal corporation. A citizen has reported a road with a condition "
              f"classified as '{condition}' at '{address}'. Write a concise, formal, and descriptive report "
              f"(around 40-50 words) for the Public Works Department. The tone should be urgent but professional. "
              f"Start the description directly, without any preamble.")
    try:
        return jsonify({'success': True, 'description': gemini_generate([{'text': prompt}])})
    except Exception as e:
        print(f"⚠️ Gemini description unavailable, using template: {e}")
        return jsonify({'success': True, 'fallback': True, 'description': (
            f"The road at {address} has been assessed as being in {condition.lower()} condition. "
            f"The deteriorated surface poses a risk to motorists and pedestrians and may worsen with rainfall "
            f"and traffic. The Public Works Department is requested to inspect and schedule repairs at the earliest.")})


def _load_font(size):
    for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except IOError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


@app.route('/generate-shareable-image', methods=['POST'])
def generate_shareable_image():
    data = request.get_json(silent=True) or {}
    original_filename = secure_filename(os.path.basename(data.get('original_filename') or ''))
    condition = str(data.get('condition') or 'a damaged')[:40]
    address = str(data.get('address') or 'our area')[:300]

    if not original_filename:
        return jsonify({'success': False, 'error': 'Original filename not provided.'}), 400
    original_path = os.path.join(UPLOAD_FOLDER, original_filename)
    if not os.path.isfile(original_path):
        return jsonify({'success': False, 'error': 'Original image not found.'}), 404

    city = address.split(',')[-1].strip() if ',' in address else 'OurCity'
    city_tag = ''.join(ch for ch in city if ch.isalnum()) or 'OurCity'
    prompt_text = (f"You are a social media manager for a civic activism app. A user reported a road in '{condition}' "
                   f"condition at '{address}'. Generate an inspiring, tweet-length social media post (under 280 "
                   f"characters) to raise awareness. Include #RoadSafety, #{city_tag}, and #CivicAction. The tone "
                   f"should be positive and action-oriented, even for poor conditions (e.g., 'Let's get this fixed!'). "
                   f"Return only the post text.")
    social_post_text = f"A citizen reported a road in {condition.lower()} condition. Let's get it fixed! #RoadSafety #{city_tag} #CivicAction"
    try:
        social_post_text = gemini_generate([{'text': prompt_text}])
    except Exception as e:
        print(f"⚠️ Gemini social post unavailable, using template: {e}")

    try:
        with Image.open(original_path) as base:
            base = ImageOps.fit(ImageOps.exif_transpose(base), (1080, 1080), Image.Resampling.LANCZOS,
                                centering=(0.5, 0.5)).convert("RGBA")
            txt_img = Image.new("RGBA", base.size, (255, 255, 255, 0))
            font = _load_font(int(base.width / 25))
            draw = ImageDraw.Draw(txt_img)

            avg_char_width = sum(font.getbbox(ch)[2] for ch in 'abcdefghijklmnopqrstuvwxyz') / 26
            wrapper = textwrap.TextWrapper(width=max(10, int((base.width * 0.9) / avg_char_width)))
            lines = wrapper.wrap(social_post_text)
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            total_text_height = sum(line_heights) + (len(lines) - 1) * 10

            banner_height = total_text_height + 70
            draw.rectangle(((0, base.height - banner_height), (base.width, base.height)), fill=(0, 0, 0, 170))
            y_text = base.height - banner_height + 25
            for i, line in enumerate(lines):
                draw.text(((base.width - font.getbbox(line)[2]) / 2, y_text), line, font=font, fill="white")
                y_text += line_heights[i] + 10

            draw.text((base.width - 190, base.height - 30), "Generated by RoadSight", font=_load_font(18),
                      fill=(255, 255, 255, 170))

            combined = Image.alpha_composite(base, txt_img)
            new_filename = f"share_{os.path.splitext(original_filename)[0]}.jpeg"
            combined.convert("RGB").save(os.path.join(GENERATED_FOLDER, new_filename), "JPEG", quality=90)
            return jsonify({'success': True, 'shareable_image_url': f'/static/generated/{new_filename}',
                            'post_text': social_post_text})
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate shareable image.'}), 500


# --- Admin Route to Generate Test Reports ---
@app.route('/api/admin/generate-reports', methods=['POST'])
@require_admin_api
def admin_generate_reports():
    """Generate random test reports from the dataset (demo data)."""
    try:
        num_reports = min(max(int((request.get_json(silent=True) or {}).get('count', 20)), 1), 100)
        from generate_random_reports import generate_random_reports
        generated = generate_random_reports(num_reports)
        return jsonify({'success': True, 'message': f'Generated {generated} test reports'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'true').lower() == 'true')
