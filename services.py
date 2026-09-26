"""
RoadSight core services — shared by the Flask API (app.py) and the
test-data generator (generate_random_reports.py).

Implements the non-ML parts of the RoadSight pipeline:
  * MongoDB connection + indexes
  * Weather-driven deterioration risk (Open-Meteo, past 7 + next 7 days)
  * Road Health Index and deterioration forecast
  * Duplicate / fraud filtering (perceptual image hash + GPS radius + reporter/time)
  * Report density, priority scoring and hotspot clustering
  * Geocoding (OpenStreetMap Nominatim) and e-mail notifications
"""
import os
import sys
import math
import smtplib
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import gridfs
import requests
from dotenv import load_dotenv
from PIL import Image
from pymongo import MongoClient, ASCENDING, DESCENDING, GEOSPHERE

load_dotenv()

# Emoji log lines must not crash on Windows consoles / redirected output (cp1252)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
GENERATED_FOLDER = os.path.join(BASE_DIR, 'static', 'generated')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

HTTP_HEADERS = {'User-Agent': 'RoadSight/1.0 (road damage reporting system)'}

# --- MongoDB ---
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/roadsight')
mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=10000)
try:
    db = mongo_client.get_default_database()
except Exception:
    db = mongo_client['roadsight']
reports_col = db.get_collection('reports')
assignments_col = db.get_collection('assignments')
users_col = db.get_collection('users')
milestones_col = db.get_collection('milestones')


# Report photos are persisted in GridFS: hosts like Render wipe the local disk on every deploy.
image_store = gridfs.GridFS(db, collection='images')


def persist_image(path):
    """Copy a local upload into GridFS (idempotent). Returns True on success."""
    name = os.path.basename(path)
    try:
        if not image_store.exists({'filename': name}):
            with open(path, 'rb') as f:
                image_store.put(f, filename=name, uploadedAt=datetime.utcnow())
        return True
    except Exception as e:
        print(f"⚠️ Could not persist image {name}: {e}")
        return False


def restore_image(name, folder):
    """Restore a GridFS image to local disk if it is missing. Returns True if the file is available."""
    path = os.path.join(folder, name)
    if os.path.isfile(path):
        return True
    try:
        grid_out = image_store.find_one({'filename': name})
        if grid_out is None:
            return False
        with open(path, 'wb') as f:
            f.write(grid_out.read())
        return True
    except Exception as e:
        print(f"⚠️ Could not restore image {name}: {e}")
        return False


def delete_image(name):
    try:
        for g in image_store.find({'filename': name}):
            image_store.delete(g._id)
    except Exception as e:
        print(f"⚠️ Could not delete image {name}: {e}")
    try:
        os.remove(os.path.join(UPLOAD_FOLDER, name))
    except OSError:
        pass


# --- Rate limiting (in-memory, per client IP + bucket) ---
_rate_hits = defaultdict(deque)
_rate_lock = threading.Lock()


def rate_limited(key, limit, window_s=60):
    """Sliding-window limiter. Returns True if this call exceeds `limit` calls per `window_s`."""
    now = time.monotonic()
    with _rate_lock:
        hits = _rate_hits[key]
        while hits and now - hits[0] > window_s:
            hits.popleft()
        if len(hits) >= limit:
            return True
        hits.append(now)
        if len(_rate_hits) > 10000:  # bound memory
            for k in [k for k, v in _rate_hits.items() if not v][:5000]:
                del _rate_hits[k]
        return False


def ensure_indexes():
    try:
        reports_col.create_index([('createdAt', ASCENDING)])
        reports_col.create_index([('status', ASCENDING)])
        reports_col.create_index([('category', ASCENDING)])
        reports_col.create_index([('severity.level', ASCENDING)])
        reports_col.create_index([('priorityScore', DESCENDING)])
        reports_col.create_index([('imageHash', ASCENDING)])
        reports_col.create_index([('location.geo', GEOSPHERE)])
        assignments_col.create_index([('reportId', ASCENDING)])
        users_col.create_index([('email', ASCENDING)], unique=True)
        milestones_col.create_index([('reportId', ASCENDING)])
        milestones_col.create_index([('createdAt', ASCENDING)])
        db.get_collection('analyses').create_index([('filename', ASCENDING)])
        db.get_collection('analyses').create_index([('createdAt', ASCENDING)], expireAfterSeconds=7 * 86400)
    except Exception as e:
        print(f"⚠️ Could not create MongoDB indexes (is MongoDB reachable?): {e}")


# --- Severity model ---
# Model output class -> severity level + numeric damage score (0 = perfect, 1 = failed)
SEVERITY_MAP = {
    'good':         {'level': 'good',     'score': 0.1},
    'satisfactory': {'level': 'moderate', 'score': 0.4},
    'poor':         {'level': 'poor',     'score': 0.7},
    'very_poor':    {'level': 'critical', 'score': 0.9},
}
# Upper damage-score boundary of each level, used for the deterioration forecast
LEVEL_BOUNDS = [(0.25, 'good'), (0.55, 'moderate'), (0.80, 'poor'), (1.00, 'critical')]
LEVEL_LABELS = {'good': 'Good', 'moderate': 'Satisfactory', 'poor': 'Poor', 'critical': 'Very Poor'}


def normalize_condition(condition):
    """'Very Poor' / 'very-poor' / 'very_poor' -> 'very_poor'."""
    return (condition or '').strip().lower().replace(' ', '_').replace('-', '_')


def severity_for(condition):
    return dict(SEVERITY_MAP.get(normalize_condition(condition), SEVERITY_MAP['satisfactory']))


# --- Geo helpers ---
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(a))


def valid_coords(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None, None
    if -90 <= lat <= 90 and -180 <= lon <= 180:
        return lat, lon
    return None, None


def geocode_address(address):
    """Forward-geocode an address with OpenStreetMap Nominatim. Returns (lat, lon) or (None, None)."""
    if not address:
        return None, None
    try:
        r = requests.get('https://nominatim.openstreetmap.org/search',
                         params={'q': address, 'format': 'json', 'limit': 1},
                         headers=HTTP_HEADERS, timeout=6)
        r.raise_for_status()
        hits = r.json()
        if hits:
            return float(hits[0]['lat']), float(hits[0]['lon'])
    except Exception as e:
        print(f"⚠️ Geocoding failed: {e}")
    return None, None


def reverse_geocode(lat, lon):
    """Reverse-geocode coordinates to a human-readable address (or None)."""
    try:
        r = requests.get('https://nominatim.openstreetmap.org/reverse',
                         params={'lat': lat, 'lon': lon, 'format': 'json', 'zoom': 18},
                         headers=HTTP_HEADERS, timeout=6)
        r.raise_for_status()
        return r.json().get('display_name')
    except Exception as e:
        print(f"⚠️ Reverse geocoding failed: {e}")
        return None


# --- Weather-driven deterioration risk ---
def compute_weather_risk(latitude, longitude):
    """
    Deterioration Risk Score (0..1) from the preceding 7 days and next 7 days of weather.

    Components (pavement studies show rainfall can double damage progression;
    freeze-thaw and large thermal swings open cracks into potholes):
      * cumulative rainfall            45 %
      * heavy-rain days (>= 20 mm)     20 %
      * freeze-thaw cycles             20 %
      * mean daily temperature swing   15 %
    """
    result = {'available': False, 'riskScore': 0.0}
    if latitude is None or longitude is None:
        return result
    try:
        params = {
            'latitude': latitude, 'longitude': longitude,
            'daily': 'precipitation_sum,temperature_2m_max,temperature_2m_min',
            'past_days': 7, 'forecast_days': 7, 'timezone': 'auto',
        }
        r = requests.get('https://api.open-meteo.com/v1/forecast', params=params, timeout=8)
        r.raise_for_status()
        daily = r.json().get('daily', {})
        dates = daily.get('time', [])
        rain = [x or 0.0 for x in daily.get('precipitation_sum', [])]
        tmax = daily.get('temperature_2m_max', [])
        tmin = daily.get('temperature_2m_min', [])
        today = datetime.utcnow().date().isoformat()
        past_idx = [i for i, d in enumerate(dates) if d < today]
        future_idx = [i for i, d in enumerate(dates) if d >= today]

        past_rain = sum(rain[i] for i in past_idx)
        forecast_rain = sum(rain[i] for i in future_idx)
        heavy_days = sum(1 for x in rain if x >= 20)
        pairs = [(hi, lo) for hi, lo in zip(tmax, tmin) if hi is not None and lo is not None]
        freeze_thaw = sum(1 for hi, lo in pairs if hi > 0 and lo < 0)
        swing = sum(hi - lo for hi, lo in pairs) / len(pairs) if pairs else 0.0

        rain_c = min((past_rain + forecast_rain) / 150.0, 1.0)
        heavy_c = min(heavy_days / 3.0, 1.0)
        ft_c = min(freeze_thaw / 4.0, 1.0)
        swing_c = min(max(swing - 8.0, 0.0) / 12.0, 1.0)
        risk = 0.45 * rain_c + 0.20 * heavy_c + 0.20 * ft_c + 0.15 * swing_c

        return {
            'available': True,
            'riskScore': round(min(max(risk, 0.0), 1.0), 3),
            'pastRainMm': round(past_rain, 1),
            'forecastRainMm': round(forecast_rain, 1),
            'heavyRainDays': heavy_days,
            'freezeThawCycles': freeze_thaw,
            'avgTempSwingC': round(swing, 1),
            'fetchedAt': datetime.utcnow(),
        }
    except Exception as e:
        print(f"⚠️ Weather risk unavailable: {e}")
        return result


def compute_health_forecast(severity_score, risk_score):
    """
    Road Health Index (0-100, higher is healthier) and a deterioration forecast.

    Damage is modelled as growing linearly at a base rate of ~0.06 score/month,
    scaled by (1 + weather risk) so the worst weather doubles the rate.
    """
    base_rate_per_day = 0.002
    rate = base_rate_per_day * (1.0 + risk_score)
    rhi = round(100 * (1.0 - severity_score) * (1.0 - 0.3 * risk_score))
    projected_score = min(severity_score + rate * 30, 1.0)
    rhi_30 = round(100 * (1.0 - projected_score) * (1.0 - 0.3 * risk_score))

    current_level = next(lvl for bound, lvl in LEVEL_BOUNDS if severity_score <= bound)
    next_bound = next(bound for bound, lvl in LEVEL_BOUNDS if severity_score <= bound)
    idx = [lvl for _, lvl in LEVEL_BOUNDS].index(current_level)
    if idx + 1 < len(LEVEL_BOUNDS):
        next_level = LEVEL_BOUNDS[idx + 1][1]
        days = max(1, round((next_bound - severity_score) / rate))
        summary = f"Likely to deteriorate to '{LEVEL_LABELS[next_level]}' in ~{days} days"
    else:
        next_level = None
        days = max(1, round((1.0 - severity_score) / rate))
        summary = f"Surface failure / pothole formation expected in ~{days} days without repair"
    return {
        'roadHealthIndex': rhi,
        'roadHealthIndex30d': rhi_30,
        'deteriorationRatePerMonth': round(rate * 30, 3),
        'nextLevel': next_level,
        'daysToNextLevel': days,
        'summary': summary,
    }


# --- Duplicate & fraud filtering ---
def image_dhash(image_path, hash_size=8):
    """64-bit difference hash (perceptual fingerprint) as a 16-char hex string."""
    with Image.open(image_path) as img:
        img = img.convert('L').resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        px = list(img.getdata())
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f'{bits:016x}'


def hamming(h1, h2):
    try:
        return bin(int(h1, 16) ^ int(h2, 16)).count('1')
    except (TypeError, ValueError):
        return 64


DUPLICATE_RADIUS_M = 50
SAME_IMAGE_MAX_DIST = 4      # near-identical file (re-upload / resized copy)
SIMILAR_IMAGE_MAX_DIST = 12  # same scene photographed again
ANONYMOUS_EMAILS = {'', None, 'anonymous@citizen.local'}


def find_duplicate(image_hash, lat, lon, reporter_email, now=None):
    """
    Returns (existing_report, reason) if the new report duplicates an existing one, else (None, None).
      * same_image              – the same photo was already submitted anywhere (reuse / fraud)
      * similar_image_nearby    – visually similar photo within 50 m of an open report
      * same_reporter_location  – same reporter, within 50 m, within 24 h
    """
    now = now or datetime.utcnow()
    if image_hash:
        for doc in reports_col.find({'imageHash': {'$exists': True}}, {'imageHash': 1}):
            if hamming(image_hash, doc.get('imageHash')) <= SAME_IMAGE_MAX_DIST:
                return reports_col.find_one({'_id': doc['_id']}), 'same_image'

    if lat is None or lon is None:
        return None, None

    nearby = reports_col.find({
        'status': {'$ne': 'Resolved'},
        'createdAt': {'$gte': now - timedelta(days=30)},
        'location.geo': {'$nearSphere': {
            '$geometry': {'type': 'Point', 'coordinates': [lon, lat]},
            '$maxDistance': DUPLICATE_RADIUS_M,
        }},
    })
    for doc in nearby:
        if image_hash and hamming(image_hash, doc.get('imageHash')) <= SIMILAR_IMAGE_MAX_DIST:
            return doc, 'similar_image_nearby'
        same_reporter = (reporter_email not in ANONYMOUS_EMAILS
                         and (doc.get('reporter') or {}).get('email') == reporter_email)
        if same_reporter and doc.get('createdAt') and now - doc['createdAt'] <= timedelta(hours=24):
            return doc, 'same_reporter_location'
    return None, None


DUPLICATE_MESSAGES = {
    'same_image': 'This photo has already been submitted. Duplicate or reused images are not accepted.',
    'similar_image_nearby': 'This damage has already been reported at this location. Your sighting has been added as a confirmation.',
    'same_reporter_location': 'You already reported damage at this location in the last 24 hours.',
}


# --- Density, priority, hotspots ---
HOTSPOT_RADIUS_M = 500


def compute_density(latitude, longitude, exclude_id=None, radius_m=HOTSPOT_RADIUS_M):
    """Number of other open reports within radius_m, plus their independent confirmations."""
    if latitude is None or longitude is None:
        return 0
    query = {
        'status': {'$ne': 'Resolved'},
        'location.geo': {'$nearSphere': {
            '$geometry': {'type': 'Point', 'coordinates': [longitude, latitude]},
            '$maxDistance': radius_m,
        }},
    }
    if exclude_id is not None:
        query['_id'] = {'$ne': exclude_id}
    try:
        return sum(1 + len(d.get('confirmedBy') or []) for d in reports_col.find(query, {'confirmedBy': 1}))
    except Exception as e:
        print(f"⚠️ Density calculation error: {e}")
        return 0


def compute_priority(severity_score, risk_score, density_count, confirmations=0, severity_level=None):
    """Priority = 60 % severity + 25 % weather risk + 15 % crowd evidence (nearby reports + confirmations)."""
    crowd = min(density_count + confirmations, 10) / 10.0
    score = round(severity_score * 0.6 + risk_score * 0.25 + crowd * 0.15, 3)
    if severity_level == 'critical':
        return 'High', max(score, 0.9)
    if score >= 0.55:
        return 'High', score
    if score >= 0.25:
        return 'Medium', score
    return 'Low', score


def refresh_priority(report_id):
    """Recompute density-dependent fields of a stored report."""
    doc = reports_col.find_one({'_id': report_id})
    if not doc:
        return
    loc = doc.get('location') or {}
    density = compute_density(loc.get('latitude'), loc.get('longitude'), exclude_id=doc['_id'])
    sev = doc.get('severity') or {}
    priority, score = compute_priority(sev.get('score', 0.4), doc.get('predictiveRisk', 0.0), density,
                                       len(doc.get('confirmedBy') or []), sev.get('level'))
    reports_col.update_one({'_id': doc['_id']}, {'$set': {
        'reportDensity': density, 'priority': priority, 'priorityScore': score}})


def refresh_neighbors(latitude, longitude, exclude_id=None, radius_m=HOTSPOT_RADIUS_M):
    """A new report raises the crowd evidence of its neighbours — update them too."""
    if latitude is None or longitude is None:
        return
    query = {'status': {'$ne': 'Resolved'}, 'location.geo': {'$nearSphere': {
        '$geometry': {'type': 'Point', 'coordinates': [longitude, latitude]}, '$maxDistance': radius_m}}}
    try:
        for d in reports_col.find(query, {'_id': 1}).limit(50):
            if d['_id'] != exclude_id:
                refresh_priority(d['_id'])
    except Exception as e:
        print(f"⚠️ Neighbour refresh failed: {e}")


def compute_hotspots(radius_m=HOTSPOT_RADIUS_M, min_reports=2):
    """
    Greedy geographic clustering of open reports. Each cluster whose report count
    (including independent confirmations) reaches min_reports is a hotspot.
    """
    docs = [d for d in reports_col.find(
        {'status': {'$ne': 'Resolved'}, 'location.latitude': {'$ne': None}, 'location.longitude': {'$ne': None}},
        {'location': 1, 'severity': 1, 'priority': 1, 'priorityScore': 1, 'confirmedBy': 1})
        if isinstance((d.get('location') or {}).get('latitude'), (int, float))]
    docs.sort(key=lambda d: d.get('priorityScore') or 0, reverse=True)
    used, hotspots = set(), []
    for seed in docs:
        if seed['_id'] in used:
            continue
        s_lat, s_lon = seed['location']['latitude'], seed['location']['longitude']
        members = [d for d in docs if d['_id'] not in used and
                   haversine_m(s_lat, s_lon, d['location']['latitude'], d['location']['longitude']) <= radius_m]
        weight = sum(1 + len(d.get('confirmedBy') or []) for d in members)
        if weight < min_reports:
            continue
        for d in members:
            used.add(d['_id'])
        sev = [(d.get('severity') or {}).get('score', 0.4) for d in members]
        hotspots.append({
            'latitude': sum(d['location']['latitude'] for d in members) / len(members),
            'longitude': sum(d['location']['longitude'] for d in members) / len(members),
            'reports': len(members),
            'weight': weight,
            'avgSeverity': round(sum(sev) / len(sev), 2),
            'highPriority': sum(1 for d in members if d.get('priority') == 'High'),
            'reportIds': [str(d['_id']) for d in members],
            'address': members[0]['location'].get('address'),
        })
    hotspots.sort(key=lambda h: (h['weight'], h['avgSeverity']), reverse=True)
    return hotspots


# --- Milestones ---
def add_milestone(report_id, status, title, description, previous_status=None, created_by='system', when=None):
    milestones_col.insert_one({
        'reportId': report_id,
        'status': status,
        'previousStatus': previous_status,
        'title': title,
        'description': description,
        'createdAt': when or datetime.utcnow(),
        'createdBy': created_by,
    })


# --- E-mail notifications ---
SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '0') or 0)
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
SMTP_FROM = os.getenv('SMTP_FROM') or SMTP_USER or 'no-reply@roadsight.local'
# Comma-separated list of authority inboxes that receive high-priority alerts
ALERT_EMAILS = [e.strip() for e in os.getenv('ALERT_EMAILS', '').split(',') if e.strip()]


def smtp_configured():
    return bool(SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS)


def send_email(to_emails, subject, body):
    if isinstance(to_emails, str):
        to_emails = [to_emails]
    to_emails = [e for e in to_emails if e and not e.endswith('.local')]
    if not to_emails:
        return
    if not smtp_configured():
        print(f"ℹ️ Email skipped (set SMTP_* env vars): {subject}")
        return
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_FROM
    msg['To'] = ', '.join(to_emails)
    try:
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_FROM, to_emails, msg.as_string())
        else:  # 587 / 25: STARTTLS
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.starttls()
                server.login(SMTP_USER, SMTP_PASS)
                server.sendmail(SMTP_FROM, to_emails, msg.as_string())
        print(f"📧 Email sent to {', '.join(to_emails)}: {subject}")
    except Exception as e:
        print(f"❌ Email failed: {e}")


def send_email_async(to_emails, subject, body):
    threading.Thread(target=send_email, args=(to_emails, subject, body), daemon=True).start()
