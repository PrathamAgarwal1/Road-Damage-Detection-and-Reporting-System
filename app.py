import os
import cv2
import numpy as np
import base64
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, jsonify, session, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient, ASCENDING, GEOSPHERE
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime, timedelta
import secrets
import requests
import json
import textwrap
import random
import shutil
import math
from bson import ObjectId
from urllib.parse import urlencode

# --- Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(16)
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# --- MongoDB Configuration ---
MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/roadsight')
mongo_client = MongoClient(MONGO_URI)
db = mongo_client.get_default_database()
reports_col = db.get_collection('reports')
assignments_col = db.get_collection('assignments')
users_col = db.get_collection('users')
milestones_col = db.get_collection('milestones')

# Ensure indexes
reports_col.create_index([('createdAt', ASCENDING)])
reports_col.create_index([('status', ASCENDING)])
reports_col.create_index([('category', ASCENDING)])
reports_col.create_index([('severity.level', ASCENDING)])
reports_col.create_index([('location.geo', GEOSPHERE)])
assignments_col.create_index([('reportId', ASCENDING)])
users_col.create_index([('email', ASCENDING)], unique=True)
milestones_col.create_index([('reportId', ASCENDING)])
milestones_col.create_index([('createdAt', ASCENDING)])

# --- Gemini API Configuration ---
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

# --- Model Configuration ---
CLASS_NAMES = ['good', 'poor', 'satisfactory', 'very_poor']
IMG_SIZE = 224
MODEL_PATH = 'road_damage_model.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(CLASS_NAMES))

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    print("✅ ResNet18 model loaded successfully")
else:
    print(f"❌ MODEL ERROR: The model file '{MODEL_PATH}' was not found.")

# --- YOLOv8 Road-Segmentation Model ---
yolo_model = None
try:
    from ultralytics import YOLO as _YOLO
    _YOLO_PATH = 'yolov8n-seg.pt'   # auto-downloads on first run (~6 MB)
    yolo_model = _YOLO(_YOLO_PATH)
    print("✅ YOLOv8-seg model loaded")
except Exception as _e:
    print(f"⚠️  YOLOv8 unavailable ({_e}). Run: pip install ultralytics")

# COCO class IDs that are NOT the road surface (vehicles, people, animals …)
_NON_ROAD_COCO = {0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                  24,25,26,27,28,29,30,31,32,33,56,57,58,59,60}

def is_road_image(image_path):
    """Use Gemini Vision to definitively check if the image is a road/street."""
    try:
        with open(image_path, "rb") as f:
            img_data = base64.b64encode(f.read()).decode("utf-8")
        
        ext = os.path.splitext(image_path)[1].lower()
        mime = "image/png" if ext == ".png" else ("image/webp" if ext == ".webp" else "image/jpeg")

        prompt = (
            "Examine this image carefully. "
            "Does it show an ACTUAL outdoor road, street, highway, or pavement meant for vehicles? "
            "If the image is a test paper, document, text, screenshot, indoor floor, or anything else, you MUST answer NO. "
            "Answer strictly with only YES or NO."
        )

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime, "data": img_data}}
                ]
            }]
        }
        res = requests.post(GEMINI_API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload), timeout=10)
        res.raise_for_status()
        text = res.json()['candidates'][0]['content']['parts'][0]['text'].strip().upper()
        print(f"🤖 Gemini Vision Validation -> {text}")
        return 'YES' in text
    except Exception as e:
        if isinstance(e, requests.exceptions.HTTPError) and e.response is not None and e.response.status_code == 400:
            print(f"❌ Gemini API rejected request (Likely Invalid API Key). Cannot validate road. Error: {e.response.text}")
            return False
        print(f"⚠️ Gemini road validation failed (fallback to heuristics): {e}")
        return True  # Fallback to True only if it's a timeout or network error

def segment_road_region(image_path):
    """
    Detect road surface in *image_path* with a robust multi-stage pipeline:
      0. Gemini Vision validation (rejects test papers, indoor photos, etc.)
      1. Perspective-aware trapezoid mask  (road ≈ lower portion of scene)
      2. Asphalt colour filter             (low-saturation, mid-to-dark brightness)
      3. YOLOv8-seg object subtraction     (erase vehicles / people / sky)

    Returns (viz_filename | None, road_coverage_pct, road_found)
    """
    try:
        if not is_road_image(image_path):
            print("🚫 Gemini rejected image: Not a road.")
            return None, 0.0, False

        img = cv2.imread(image_path)
        if img is None:
            return None, 0.0, False
        h, w = img.shape[:2]

        # --- 1. Perspective trapezoid ---
        top_ratio = 0.28           # road width fraction at the horizon
        horizon_y = int(h * 0.38)  # row where road perspective vanishes
        pts = np.array([
            [int(w * 0.01),                  h - 1],
            [int(w * 0.99),                  h - 1],
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
        road_mask = cv2.morphologyEx(road_mask,
            cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))
        road_mask = cv2.morphologyEx(road_mask,
            cv2.MORPH_OPEN,  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10)))

        # --- 3. YOLO object subtraction ---
        if yolo_model is not None:
            try:
                res = yolo_model.predict(image_path, verbose=False)[0]
                if res.masks is not None:
                    for cls_id, raw_mask in zip(
                        res.boxes.cls.cpu().numpy().astype(int),
                        res.masks.data.cpu().numpy()
                    ):
                        if int(cls_id) in _NON_ROAD_COCO:
                            obj = cv2.resize(raw_mask, (w, h),
                                             interpolation=cv2.INTER_NEAREST)
                            road_mask[obj > 0.5] = 0
            except Exception as ye:
                print(f"⚠️  YOLO inference skipped: {ye}")

        coverage = float(np.sum(road_mask > 0)) / (h * w) * 100
        road_found = coverage >= 5.0

        if not road_found:
            # Graceful fallback: lower 55 % of frame
            road_mask[:] = 0
            road_mask[int(h * 0.45):, :] = 255
            coverage = 0.0   # report 0 so UI can warn the user

        # --- Build visualisation overlay ---
        viz = img.copy()
        dimmed = (img * 0.25).astype(np.uint8)
        non_road = cv2.bitwise_not(road_mask)
        viz[non_road > 0] = dimmed[non_road > 0]
        # Green contour around road region
        contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(viz, contours, -1, (0, 220, 80), 3)

        base = os.path.splitext(os.path.basename(image_path))[0]
        viz_name = f"seg_{base}.jpg"
        cv2.imwrite(os.path.join(UPLOAD_FOLDER, viz_name), viz)
        return viz_name, round(coverage, 1), road_found

    except Exception as e:
        print(f"❌ segment_road_region error: {e}")
        return None, 0.0, False

# --- Image Preprocessing for PyTorch Model ---
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, predicted_idx = torch.max(probabilities, 0)
            predicted_class = CLASS_NAMES[predicted_idx.item()]
            confidence_score = confidence.item() * 100
        return predicted_class, confidence_score
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None, None

# --- Utility Functions ---
def compute_predictive_risk(latitude, longitude):
    try:
        if latitude is None or longitude is None:
            return 0.0
        # Open-Meteo free API (no key required)
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'hourly': 'precipitation,temperature_2m',
            'forecast_days': 1
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urlencode(params)}"
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        precip = sum(data.get('hourly', {}).get('precipitation', [])[:12])  # next 12 hours
        temp_avg = 0.0
        temps = data.get('hourly', {}).get('temperature_2m', [])[:12]
        if temps:
            temp_avg = sum(temps) / len(temps)
        # Heuristic risk: more rain -> higher risk; extreme temps slightly higher
        rain_component = min(precip / 10.0, 1.0)  # 0..1
        temp_component = 0.0
        if temp_avg <= 0 or temp_avg >= 38:
            temp_component = 0.2
        return max(0.0, min(rain_component + temp_component, 1.0))
    except Exception:
        return 0.0

def compute_density(latitude, longitude, category, report_id=None, radius_m=500):
    try:
        if latitude is None or longitude is None:
            return 0
        
        # Build query to find nearby reports
        # Using aggregation with $geoNear for better performance
        pipeline = [
            {
                '$geoNear': {
                    'near': {'type': 'Point', 'coordinates': [longitude, latitude]},
                    'distanceField': 'distance',
                    'maxDistance': radius_m,
                    'spherical': True,
                    'query': {'category': category}
                }
            },
            {'$count': 'count'}
        ]
        
        try:
            result = list(reports_col.aggregate(pipeline))
            count = result[0]['count'] if result else 0
            # Exclude current report if report_id is provided
            if report_id:
                current_report = reports_col.find_one({'_id': report_id})
                if current_report and current_report.get('location', {}).get('geo'):
                    # Check if current report is in the results
                    count = max(0, count - 1)
            return count
        except Exception as e:
            # Fallback to simple count if aggregation fails
            print(f"⚠️ GeoNear aggregation failed, using fallback: {e}")
            # Simple distance-based check (less accurate but works)
            all_reports = reports_col.find({'category': category, 'location.geo': {'$exists': True}})
            count = 0
            for report in all_reports:
                if report_id and str(report['_id']) == str(report_id):
                    continue
                loc = report.get('location', {}).get('geo', {}).get('coordinates')
                if loc and len(loc) == 2:
                    # Simple distance calculation (Haversine formula)
                    lat2, lon2 = math.radians(loc[1]), math.radians(loc[0])
                    lat1, lon1 = math.radians(latitude), math.radians(longitude)
                    dlon, dlat = lon2 - lon1, lat2 - lat1
                    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                    c = 2 * math.asin(math.sqrt(a))
                    distance = 6371 * c * 1000  # Convert to meters
                    if distance <= radius_m:
                        count += 1
            return count
    except Exception as e:
        print(f"⚠️ Density calculation error: {e}")
        return 0

def recompute_priority(severity_score, predictive_risk, density_count, severity_level=None):
    # Critical/very_poor should always be High priority
    if severity_level == 'critical':
        return 'High', 0.9
    
    # Calculate priority score
    priority_score = (severity_score * 0.6) + (predictive_risk * 0.25) + (min(density_count, 10) / 10.0 * 0.15)
    
    # Adjusted thresholds for better distribution
    if priority_score >= 0.55:  # High priority
        return 'High', priority_score
    elif priority_score >= 0.25:  # Medium priority
        return 'Medium', priority_score
    else:  # Low priority
        return 'Low', priority_score

def serialize_report(doc):
    doc = dict(doc)
    doc['id'] = str(doc.pop('_id'))
    return doc

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"road_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # --- Stage 1: Road-region segmentation (YOLOv8 + colour filter) ---
        viz_name, road_coverage, road_found = segment_road_region(filepath)

        # If it's definitely not a road, reject it immediately
        if not road_found and viz_name is None:
            return jsonify({'success': False, 'error': 'Image does not appear to show a road or street. Please upload a valid photo.'}), 400

        # --- Stage 2: Classify condition with ResNet18 (on original image) ---
        condition, confidence = predict_image(filepath)
        if condition is None:
            return jsonify({'success': False, 'error': 'Failed to analyze image'}), 500

        severity_map = {
            'good':        {'level': 'good',     'icon': 'check-circle'},
            'satisfactory':{'level': 'moderate', 'icon': 'exclamation-triangle'},
            'poor':        {'level': 'poor',     'icon': 'exclamation-circle'},
            'very_poor':   {'level': 'critical', 'icon': 'times-circle'},
        }
        severity = severity_map.get(condition, severity_map['satisfactory'])
        return jsonify({
            'success':           True,
            'condition':         condition.replace('_', ' ').title(),
            'confidence':        round(confidence, 2),
            'severity':          severity,
            'image_url':         f'/static/uploads/{filename}',
            'segmented_url':     f'/static/uploads/{viz_name}' if viz_name else None,
            'road_coverage':     road_coverage,
            'road_found':        road_found,
            'original_filename': filename,
        })
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit-report', methods=['POST'])
def submit_report():
    data = request.json
    now = datetime.utcnow()
    location = data.get('location') or {}
    latitude = location.get('latitude')
    longitude = location.get('longitude')
    address = location.get('address') or 'Unknown'
    condition = data.get('condition')
    confidence = data.get('confidence')
    image_url = data.get('image_url')
    description = data.get('description')
    # Use logged-in user info if available, otherwise use form data
    if session.get('user'):
        user_info = session.get('user')
        reporter_email = user_info.get('email')
        reporter_name = user_info.get('name', 'Anonymous')
    else:
        reporter_email = data.get('email')
        reporter_name = data.get('name', 'Anonymous')

    # Map condition to severity (matching model output: good, poor, satisfactory, very_poor)
    severity_map = {
        'good': {'level': 'good', 'score': 0.1},
        'satisfactory': {'level': 'moderate', 'score': 0.4},
        'poor': {'level': 'poor', 'score': 0.7},
        'very_poor': {'level': 'critical', 'score': 0.9},
        'very poor': {'level': 'critical', 'score': 0.9}
    }
    # Normalize condition: "Very Poor" -> "very_poor", "Good" -> "good", etc.
    condition_normalized = (condition or '').lower().replace(' ', '_').replace('-', '_')
    severity_info = severity_map.get(condition_normalized, {'level': 'moderate', 'score': 0.4})

    # Placeholders for predictive risk and density; will be recomputed server-side later
    predictive_risk = 0.0
    density_count = 0

    # Basic priority rule (to be refined once risk/density are computed)
    priority_score = (severity_info['score'] * 0.6) + (predictive_risk * 0.25) + (min(density_count, 10) / 10.0 * 0.15)
    if priority_score >= 0.66:
        priority = 'High'
    elif priority_score >= 0.33:
        priority = 'Medium'
    else:
        priority = 'Low'

    doc = {
        'imageUrl': image_url,
        'location': {
            'address': address,
            'latitude': latitude,
            'longitude': longitude,
            'geo': {'type': 'Point', 'coordinates': [longitude, latitude]} if latitude is not None and longitude is not None else None
        },
        'category': data.get('damage_category') or 'RoadDamage',
        'condition': condition,
        'confidence': confidence,
        'severity': severity_info,
        'predictiveRisk': predictive_risk,
        'reportDensity': density_count,
        'priority': priority,
        'status': 'New',
        'reporter': {
            'name': reporter_name,
            'email': reporter_email
        },
        'description': description,
        'createdAt': now,
        'updatedAt': now
    }

    inserted = reports_col.insert_one(doc)
    # After insert, compute live risk and density and update
    predictive_risk = compute_predictive_risk(latitude, longitude)
    density_count = compute_density(latitude, longitude, doc['category'], inserted.inserted_id)
    priority, score = recompute_priority(severity_info['score'], predictive_risk, density_count, severity_info['level'])
    reports_col.update_one({'_id': inserted.inserted_id}, {'$set': {
        'predictiveRisk': predictive_risk,
        'reportDensity': density_count,
        'priority': priority,
        'priorityScore': score,
        'updatedAt': datetime.utcnow()
    }})

    print("\n" + "="*50 + "\n📋 NEW ROAD DAMAGE REPORT\n" + "="*50)
    print(f"📍 Location: {address}")
    print(f"🗺️  Coordinates: {latitude}, {longitude}")
    print(f"🚧 Condition: {condition}")
    print(f"📊 Confidence: {confidence}%")
    print(f"🎯 Severity: {severity_info}")
    print(f"🏷️  Priority: {priority}")
    print(f"💬 Description: {description or 'N/A'}")
    print(f"👤 Reporter: {reporter_name} ({reporter_email or 'Anonymous'})")
    print(f"⏰ Timestamp: {now.strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n")
    return jsonify({'success': True, 'message': 'Report submitted successfully!', 'reportId': str(inserted.inserted_id)})

# --- Auth Helpers ---
def require_admin(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('admin'): return redirect(url_for('admin_login_view'))
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
            return redirect(url_for('user_login_view'))
        return fn(*args, **kwargs)
    return wrapper

def require_user_api(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get('user'):
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401
        return fn(*args, **kwargs)
    return wrapper

# Seed default admin if not exists (email+password hash via env)
ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@example.com')
ADMIN_PASSWORD_HASH = os.getenv('ADMIN_PASSWORD_HASH')
if ADMIN_PASSWORD_HASH is None:
    # Default temporary password: admin123 (should be overridden in env)
    ADMIN_PASSWORD_HASH = generate_password_hash(os.getenv('ADMIN_PASSWORD', 'admin123'))
users_col.update_one({'email': ADMIN_EMAIL}, {'$setOnInsert': {
    'email': ADMIN_EMAIL,
    'passwordHash': ADMIN_PASSWORD_HASH,
    'role': 'admin',
    'createdAt': datetime.utcnow()
}}, upsert=True)

# --- User Authentication ---
@app.route('/user/signup', methods=['GET'])
def user_signup_view():
    return render_template('user_signup.html')

@app.route('/user/signup', methods=['POST'])
def user_signup():
    data = request.form or request.json or {}
    email = data.get('email')
    password = data.get('password')
    name = data.get('name', '')
    
    if not email or not password:
        return render_template('user_signup.html', error='Email and password are required')
    
    # Check if user exists
    if users_col.find_one({'email': email}):
        return render_template('user_signup.html', error='Email already registered')
    
    # Create user
    password_hash = generate_password_hash(password)
    users_col.insert_one({
        'email': email,
        'passwordHash': password_hash,
        'name': name,
        'role': 'user',
        'createdAt': datetime.utcnow()
    })
    
    session.permanent = True
    session['user'] = {'email': email, 'name': name, 'role': 'user'}
    return redirect(url_for('user_dashboard'))

@app.route('/user/login', methods=['GET'])
def user_login_view():
    return render_template('user_login.html')

@app.route('/user/login', methods=['POST'])
def user_login():
    data = request.form or request.json or {}
    email = data.get('email')
    password = data.get('password')
    
    user = users_col.find_one({'email': email, 'role': 'user'})
    if user and check_password_hash(user.get('passwordHash', ''), password):
        session.permanent = True
        session['user'] = {'email': user['email'], 'name': user.get('name', ''), 'role': 'user'}
        return redirect(url_for('user_dashboard'))
    return render_template('user_login.html', error='Invalid credentials')

@app.route('/user/logout')
def user_logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/user/dashboard')
@require_user
def user_dashboard():
    return render_template('user_dashboard.html')

# --- Admin Views ---
@app.route('/admin/login', methods=['GET'])
def admin_login_view():
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login():
    data = request.form or request.json or {}
    email = data.get('email')
    password = data.get('password')
    # Only allow users with role='admin' to access the admin panel
    user = users_col.find_one({'email': email, 'role': 'admin'})
    if user and check_password_hash(user.get('passwordHash', ''), password):
        session.permanent = True
        session['admin'] = {'email': user['email'], 'role': user.get('role', 'admin')}
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html', error='Invalid credentials')

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('admin_login_view'))

@app.route('/admin')
@require_admin
def admin_dashboard():
    return render_template('admin.html')

# --- Admin/Reports APIs ---
@app.route('/api/reports', methods=['GET'])
@require_admin_api
def api_reports_list():
    # filters: severity.level, status, from, to, sort
    q = {}
    severity = request.args.get('severity')
    status = request.args.get('status')
    start = request.args.get('start')
    end = request.args.get('end')
    if severity:
        q['severity.level'] = severity
    if status:
        q['status'] = status
    if start or end:
        q['createdAt'] = {}
        if start: q['createdAt']['$gte'] = datetime.fromisoformat(start)
        if end: q['createdAt']['$lte'] = datetime.fromisoformat(end)
    sort = [('priority', ASCENDING), ('createdAt', ASCENDING)] if request.args.get('sort') == 'oldest' else [('priority', ASCENDING), ('createdAt', -1)]
    docs = [serialize_report(d) for d in reports_col.find(q).sort(sort)]
    return jsonify({'success': True, 'reports': docs})

@app.route('/api/reports/<rid>/status', methods=['POST'])
@require_admin_api
def api_update_status(rid):
    data = request.json or {}
    new_status = data.get('status')
    note = data.get('note', '')
    if new_status not in ['New', 'Scheduled', 'In Progress', 'Resolved']:
        return jsonify({'success': False, 'error': 'Invalid status'}), 400
    
    report = reports_col.find_one({'_id': ObjectId(rid)})
    if not report:
        return jsonify({'success': False, 'error': 'Not found'}), 404
    
    old_status = report.get('status', 'New')
    res = reports_col.update_one({'_id': ObjectId(rid)}, {'$set': {'status': new_status, 'updatedAt': datetime.utcnow()}})
    
    # Create milestone when status changes
    if old_status != new_status:
        milestone = {
            'reportId': ObjectId(rid),
            'status': new_status,
            'previousStatus': old_status,
            'title': f'Status updated: {old_status} → {new_status}',
            'description': note or f'Road repair status changed to {new_status}',
            'createdAt': datetime.utcnow(),
            'createdBy': session.get('admin', {}).get('email', 'admin')
        }
        milestones_col.insert_one(milestone)
    
    # notification hook (email optional)
    try:
        email = (report.get('reporter') or {}).get('email')
        if email:
            send_email_notification(email, new_status, report)
    except Exception:
        pass
    return jsonify({'success': True})

@app.route('/api/reports/<rid>/assign', methods=['POST'])
@require_admin_api
def api_assign_task(rid):
    data = request.json or {}
    unit = data.get('unit')
    note = data.get('note', f'Assigned to {unit}')
    if not unit:
        return jsonify({'success': False, 'error': 'unit required'}), 400
    
    report = reports_col.find_one({'_id': ObjectId(rid)})
    if not report:
        return jsonify({'success': False, 'error': 'Report not found'}), 404
    
    assignment = {
        'reportId': ObjectId(rid),
        'unit': unit,
        'note': note,
        'createdAt': datetime.utcnow()
    }
    assignments_col.insert_one(assignment)
    
    old_status = report.get('status', 'New')
    reports_col.update_one({'_id': ObjectId(rid)}, {'$set': {'status': 'Scheduled', 'updatedAt': datetime.utcnow()}})
    
    # Create milestone for assignment
    milestone = {
        'reportId': ObjectId(rid),
        'status': 'Scheduled',
        'previousStatus': old_status,
        'title': f'Assigned to Field Unit: {unit}',
        'description': note,
        'createdAt': datetime.utcnow(),
        'createdBy': session.get('admin', {}).get('email', 'admin')
    }
    milestones_col.insert_one(milestone)
    
    return jsonify({'success': True})

@app.route('/api/admin/stats', methods=['GET'])
@require_admin_api
def api_admin_stats():
    """Return summary statistics for the admin dashboard."""
    total = reports_col.count_documents({})
    by_priority = {
        'High': reports_col.count_documents({'priority': 'High'}),
        'Medium': reports_col.count_documents({'priority': 'Medium'}),
        'Low': reports_col.count_documents({'priority': 'Low'}),
    }
    by_status = {
        'New': reports_col.count_documents({'status': 'New'}),
        'Scheduled': reports_col.count_documents({'status': 'Scheduled'}),
        'In Progress': reports_col.count_documents({'status': 'In Progress'}),
        'Resolved': reports_col.count_documents({'status': 'Resolved'}),
    }
    by_severity = {
        'critical': reports_col.count_documents({'severity.level': 'critical'}),
        'poor': reports_col.count_documents({'severity.level': 'poor'}),
        'moderate': reports_col.count_documents({'severity.level': 'moderate'}),
        'good': reports_col.count_documents({'severity.level': 'good'}),
    }
    return jsonify({'success': True, 'total': total, 'by_priority': by_priority, 'by_status': by_status, 'by_severity': by_severity})

# --- User APIs ---
@app.route('/api/user/reports', methods=['GET'])
@require_user_api
def api_user_reports():
    user_email = session.get('user', {}).get('email')
    if not user_email:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    # Get reports by user email
    reports = reports_col.find({'reporter.email': user_email}).sort([('createdAt', -1)])
    docs = [serialize_report(d) for d in reports]
    return jsonify({'success': True, 'reports': docs})

@app.route('/api/user/check', methods=['GET'])
def api_user_check():
    """Check if user is logged in"""
    user = session.get('user')
    if user:
        return jsonify({'success': True, 'user': user})
    return jsonify({'success': False, 'user': None})

@app.route('/api/user/milestones/<rid>', methods=['GET'])
@require_user_api
def api_user_milestones(rid):
    user_email = session.get('user', {}).get('email')
    if not user_email:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    # Verify report belongs to user
    report = reports_col.find_one({'_id': ObjectId(rid), 'reporter.email': user_email})
    if not report:
        return jsonify({'success': False, 'error': 'Report not found'}), 404
    
    # Get milestones for this report
    milestones = milestones_col.find({'reportId': ObjectId(rid)}).sort([('createdAt', -1)])
    docs = []
    for m in milestones:
        doc = dict(m)
        doc['id'] = str(doc.pop('_id'))
        doc['reportId'] = str(doc['reportId'])
        docs.append(doc)
    
    return jsonify({'success': True, 'milestones': docs})

# --- Optional Email Notification ---
import smtplib
from email.mime.text import MIMEText

SMTP_HOST = os.getenv('SMTP_HOST')
SMTP_PORT = int(os.getenv('SMTP_PORT', '0') or 0)
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASS = os.getenv('SMTP_PASS')
SMTP_FROM = os.getenv('SMTP_FROM', 'no-reply@roadsight.local')

def send_email_notification(to_email, status, report):
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS):
        print(f"ℹ️ Email skipped: set SMTP_* env vars to enable emails to {to_email}")
        return
    subject = f"Your RoadSight report status is now: {status}"
    body = f"Hello,\n\nYour reported road issue at {report.get('location', {}).get('address', 'Unknown')} is now '{status}'.\n\nThank you for helping improve our roads.\n"
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SMTP_FROM
    msg['To'] = to_email
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(SMTP_FROM, [to_email], msg.as_string())

@app.route('/generate-description', methods=['POST'])
def generate_description():
    data = request.json
    prompt = f"You are an official at a municipal corporation. A citizen has reported a road with a condition classified as '{data.get('condition', 'damaged')}' at '{data.get('address', 'an unspecified location')}'. Write a concise, formal, and descriptive report (around 40-50 words) for the Public Works Department. The tone should be urgent but professional. Start the description directly, without any preamble."
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        response = requests.post(GEMINI_API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        description = result['candidates'][0]['content']['parts'][0]['text']
        return jsonify({'success': True, 'description': description.strip()})
    except Exception as e:
        print(f"❌ Gemini API error (description): {e}")
        return jsonify({'success': False, 'error': 'Failed to generate description.'}), 500

@app.route('/generate-shareable-image', methods=['POST'])
def generate_shareable_image():
    data = request.json
    original_filename = data.get('original_filename')
    condition = data.get('condition', 'a damaged')
    address = data.get('address', 'our area')
    
    if not original_filename: return jsonify({'success': False, 'error': 'Original filename not provided.'}), 400
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    if not os.path.exists(original_path): return jsonify({'success': False, 'error': 'Original image not found.'}), 404

    city = address.split(',')[-1].strip() if ',' in address else 'OurCity'
    prompt_text = (f"You are a social media manager for a civic activism app. A user reported a road in '{condition}' condition at '{address}'. "
                   f"Generate an inspiring, tweet-length social media post (under 280 characters) to raise awareness. "
                   f"Include #RoadSafety, #{city.replace(' ', '')}, and #CivicAction. The tone should be positive and action-oriented, even for poor conditions (e.g., 'Let's get this fixed!').")
    
    social_post_text = "A local citizen has reported this road. #RoadSafety #CivicAction" # Fallback
    try:
        payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
        response = requests.post(GEMINI_API_URL, headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        social_post_text = result['candidates'][0]['content']['parts'][0]['text'].strip()
    except Exception as e:
        print(f"❌ Gemini API error (social post): {e}")

    try:
        with Image.open(original_path) as base:
            # Standardize image size to 1080x1080, preserving aspect ratio by padding.
            target_size = (1080, 1080)
            base = ImageOps.fit(base, target_size, Image.Resampling.LANCZOS, centering=(0.5, 0.5)).convert("RGBA")
            txt_img = Image.new("RGBA", base.size, (255, 255, 255, 0))
            
            # Dynamic font size based on image width
            font_size = int(base.width / 25)
            try:
                font = ImageFont.truetype("arialbd.ttf", size=font_size)
            except IOError:
                font = ImageFont.load_default()

            draw = ImageDraw.Draw(txt_img)
            
            # Dynamic text wrapping based on font and image width
            avg_char_width = sum(font.getbbox(char)[2] for char in 'abcdefghijklmnopqrstuvwxyz') / 26
            wrap_width = int((base.width * 0.9) / avg_char_width)
            wrapper = textwrap.TextWrapper(width=wrap_width)
            lines = wrapper.wrap(social_post_text)
            
            line_heights = [font.getbbox(line)[3] - font.getbbox(line)[1] for line in lines]
            total_text_height = sum(line_heights) + (len(lines) - 1) * 10
            
            # Draw semi-transparent background banner at the bottom
            banner_height = total_text_height + 60
            draw.rectangle(((0, base.height - banner_height), (base.width, base.height)), fill=(0, 0, 0, 170))
            
            # Draw wrapped text line by line
            y_text = base.height - banner_height + 30
            for i, line in enumerate(lines):
                line_width = font.getbbox(line)[2]
                draw.text(((base.width - line_width) / 2, y_text), line, font=font, fill="white")
                y_text += line_heights[i] + 10

            watermark_font = ImageFont.load_default()
            draw.text((base.width - 160, base.height - 35), "Generated by CivicLens", font=watermark_font, fill=(255, 255, 255, 150))

            combined = Image.alpha_composite(base, txt_img)
            new_filename = f"share_{os.path.splitext(original_filename)[0]}.jpeg"
            save_path = os.path.join(app.config['GENERATED_FOLDER'], new_filename)
            combined.convert("RGB").save(save_path, "JPEG", quality=90)
            
            return jsonify({'success': True, 'shareable_image_url': f'/static/generated/{new_filename}'})
    except Exception as e:
        print(f"❌ Image generation error: {e}")
        return jsonify({'success': False, 'error': 'Failed to generate shareable image.'}), 500

# --- Admin Route to Generate Test Reports ---
@app.route('/admin/generate-reports', methods=['POST'])
@require_admin_api
def admin_generate_reports():
    """Generate random test reports from dataset"""
    try:
        num_reports = int(request.json.get('count', 20)) if request.json else 20
        
        # Import the generation function
        import subprocess
        import sys
        result = subprocess.run([sys.executable, 'generate_random_reports.py', str(num_reports)], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            return jsonify({'success': True, 'message': f'Generated {num_reports} test reports', 'output': result.stdout})
        else:
            return jsonify({'success': False, 'error': result.stderr}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

