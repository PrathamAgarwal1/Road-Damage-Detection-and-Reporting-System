import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw, ImageFont, ImageOps
from datetime import datetime
import secrets
import requests
import json
import textwrap

# --- Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
UPLOAD_FOLDER = 'static/uploads'
GENERATED_FOLDER = 'static/generated'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

# --- Gemini API Configuration ---
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key="

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
    print("✅ Model loaded successfully")
else:
    print(f"❌ MODEL ERROR: The model file '{MODEL_PATH}' was not found.")

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

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files: return jsonify({'success': False, 'error': 'No image provided'}), 400
    file = request.files['image']
    if file.filename == '': return jsonify({'success': False, 'error': 'No file selected'}), 400
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"road_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        condition, confidence = predict_image(filepath)
        if condition is None: return jsonify({'success': False, 'error': 'Failed to analyze image'}), 500
        severity_map = {
            'good': {'level': 'good', 'icon': 'check-circle'},
            'satisfactory': {'level': 'moderate', 'icon': 'exclamation-triangle'},
            'poor': {'level': 'poor', 'icon': 'exclamation-circle'},
            'very_poor': {'level': 'critical', 'icon': 'times-circle'}
        }
        severity = severity_map.get(condition, severity_map['satisfactory'])
        return jsonify({
            'success': True, 'condition': condition.replace('_', ' ').title(),
            'confidence': round(confidence, 2), 'severity': severity,
            'image_url': f'/static/uploads/{filename}', 'original_filename': filename
        })
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit-report', methods=['POST'])
def submit_report():
    data = request.json
    print("\n" + "="*50 + "\n📋 NEW ROAD DAMAGE REPORT\n" + "="*50)
    print(f"📍 Location: {data.get('location', {}).get('address', 'Unknown')}")
    print(f"🗺️  Coordinates: {data.get('location', {}).get('latitude')}, {data.get('location', {}).get('longitude')}")
    print(f"🚧 Condition: {data.get('condition')}")
    print(f"📊 Confidence: {data.get('confidence')}%")
    print(f"💬 Description: {data.get('description', 'N/A')}")
    print(f"📧 Reporter: {data.get('email', 'Anonymous')}")
    print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n" + "="*50 + "\n")
    return jsonify({'success': True, 'message': 'Report submitted successfully!'})

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

if __name__ == '__main__':
    app.run(debug=True)

