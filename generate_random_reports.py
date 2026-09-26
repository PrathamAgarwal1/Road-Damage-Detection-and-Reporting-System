"""
Generate random demo reports from road_damage_dataset.
Only uses images from the 'poor' and 'very_poor' folders.

Reports go through the same duplicate filter, weather-risk, road-health forecast
and priority logic as real citizen submissions (see services.py).

Usage:  python generate_random_reports.py [count]
"""
import os
import random
import shutil
import sys
import uuid
from datetime import datetime, timedelta

import services
from services import reports_col, users_col, UPLOAD_FOLDER, BASE_DIR
from werkzeug.security import generate_password_hash

DATASET_PATH = os.path.join(BASE_DIR, 'road_damage_dataset')
FOLDERS = {'poor': os.path.join(DATASET_PATH, 'poor'), 'very_poor': os.path.join(DATASET_PATH, 'very_poor')}

# Indian cities with coordinates (for realistic locations)
INDIAN_LOCATIONS = [
    {'city': 'Mumbai', 'lat': 19.0760, 'lon': 72.8777, 'address': 'Mumbai, Maharashtra'},
    {'city': 'Delhi', 'lat': 28.6139, 'lon': 77.2090, 'address': 'New Delhi, Delhi'},
    {'city': 'Bangalore', 'lat': 12.9716, 'lon': 77.5946, 'address': 'Bangalore, Karnataka'},
    {'city': 'Hyderabad', 'lat': 17.3850, 'lon': 78.4867, 'address': 'Hyderabad, Telangana'},
    {'city': 'Chennai', 'lat': 13.0827, 'lon': 80.2707, 'address': 'Chennai, Tamil Nadu'},
    {'city': 'Kolkata', 'lat': 22.5726, 'lon': 88.3639, 'address': 'Kolkata, West Bengal'},
    {'city': 'Pune', 'lat': 18.5204, 'lon': 73.8567, 'address': 'Pune, Maharashtra'},
    {'city': 'Ahmedabad', 'lat': 23.0225, 'lon': 72.5714, 'address': 'Ahmedabad, Gujarat'},
    {'city': 'Jaipur', 'lat': 26.9124, 'lon': 75.7873, 'address': 'Jaipur, Rajasthan'},
    {'city': 'Surat', 'lat': 21.1702, 'lon': 72.8311, 'address': 'Surat, Gujarat'},
]

REPORTER_NAMES = [
    'Rajesh Kumar', 'Priya Sharma', 'Amit Patel', 'Sneha Reddy', 'Vikram Singh',
    'Anjali Gupta', 'Rohit Mehta', 'Kavita Nair', 'Suresh Iyer', 'Deepa Joshi',
    'Manoj Desai', 'Sunita Rao', 'Kiran Malhotra', 'Pooja Agarwal', 'Nitin Verma'
]

DESCRIPTIONS = [
    'Large pothole causing vehicle damage. Urgent repair needed.',
    'Severe road damage with multiple cracks. Safety hazard.',
    'Road surface completely deteriorated. Immediate attention required.',
    'Deep potholes causing accidents. Need urgent repair.',
    'Road in very poor condition. Multiple vehicles stuck.',
    'Severe damage to road infrastructure. High priority repair needed.',
    'Road surface broken and dangerous. Public safety concern.',
    'Extensive damage to road. Multiple complaints received.',
    'Road condition critical. Immediate repair required.',
    'Severe deterioration of road surface. Vehicles at risk.'
]

STATUS_FLOW = ['New', 'Scheduled', 'In Progress', 'Resolved']


def ensure_test_user():
    test_email = 'test@roadsight.local'
    if not users_col.find_one({'email': test_email}):
        users_col.insert_one({'email': test_email, 'passwordHash': generate_password_hash('test123'),
                              'name': 'Test User', 'role': 'user', 'createdAt': datetime.utcnow()})
        print(f"✅ Created test user: {test_email}")


def generate_random_reports(num_reports=20):
    all_images = []
    for condition, folder in FOLDERS.items():
        if os.path.isdir(folder):
            all_images += [(condition, os.path.join(folder, f)) for f in os.listdir(folder)
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not all_images:
        print("❌ No images found in poor or very_poor folders!")
        return 0

    services.ensure_indexes()
    ensure_test_user()
    random.shuffle(all_images)
    print(f"✅ Found {len(all_images)} images. Generating {num_reports} reports...\n")

    weather_cache = {}
    generated = skipped = 0
    for condition, image_path in all_images:
        if generated >= num_reports:
            break
        try:
            location = random.choice(INDIAN_LOCATIONS)
            lat = location['lat'] + random.uniform(-0.1, 0.1)
            lon = location['lon'] + random.uniform(-0.1, 0.1)
            address = f"{random.choice(['Main Road', 'Highway', 'Street', 'Avenue'])} near {location['address']}"
            reporter_name = random.choice(REPORTER_NAMES)
            reporter_email = f"{reporter_name.lower().replace(' ', '.')}@example.com"
            created_at = datetime.utcnow() - timedelta(days=random.randint(0, 30), minutes=random.randint(0, 1440))

            image_hash = services.image_dhash(image_path)
            dup, reason = services.find_duplicate(image_hash, lat, lon, reporter_email, created_at)
            if dup:
                skipped += 1
                print(f"⏭️  Skipped duplicate ({reason}): {os.path.basename(image_path)}")
                continue

            filename = f"road_{created_at.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{os.path.basename(image_path)}"
            shutil.copy2(image_path, os.path.join(UPLOAD_FOLDER, filename))
            services.persist_image(os.path.join(UPLOAD_FOLDER, filename))

            severity_info = services.severity_for(condition)
            # Weather is regional: one lookup per city
            if location['city'] not in weather_cache:
                weather_cache[location['city']] = services.compute_weather_risk(location['lat'], location['lon'])
            weather = weather_cache[location['city']]
            risk = weather['riskScore']
            forecast = services.compute_health_forecast(severity_info['score'], risk)
            density = services.compute_density(lat, lon)
            priority, score = services.compute_priority(severity_info['score'], risk, density, 0, severity_info['level'])
            status = random.choice(STATUS_FLOW)

            doc = {
                'imageUrl': f'/static/uploads/{filename}',
                'imageHash': image_hash,
                'location': {'address': address, 'latitude': lat, 'longitude': lon,
                             'geo': {'type': 'Point', 'coordinates': [lon, lat]}},
                'category': 'RoadDamage',
                'condition': condition.replace('_', ' ').title(),
                'confidence': round(random.uniform(75, 95), 2),
                'severity': severity_info,
                'weather': weather,
                'predictiveRisk': risk,
                'forecast': forecast,
                'reportDensity': density,
                'confirmedBy': [],
                'priority': priority,
                'priorityScore': score,
                'status': status,
                'reporter': {'name': reporter_name, 'email': reporter_email},
                'description': random.choice(DESCRIPTIONS),
                'demo': True,
                'createdAt': created_at,
                'updatedAt': created_at,
            }
            rid = reports_col.insert_one(doc).inserted_id

            # Timeline consistent with the status
            services.add_milestone(rid, 'New', 'Report submitted', forecast['summary'] + '.', when=created_at)
            for i in range(1, STATUS_FLOW.index(status) + 1):
                services.add_milestone(rid, STATUS_FLOW[i], f'Status updated: {STATUS_FLOW[i - 1]} → {STATUS_FLOW[i]}',
                                       f'Road repair status changed to {STATUS_FLOW[i]}',
                                       previous_status=STATUS_FLOW[i - 1], created_by='demo',
                                       when=created_at + timedelta(hours=6 * i))
            if status == 'Resolved':
                reports_col.update_one({'_id': rid}, {'$set': {
                    'resolvedAt': created_at + timedelta(days=random.randint(1, 14))}})
            services.refresh_neighbors(lat, lon, exclude_id=rid)

            generated += 1
            print(f"✅ [{generated}/{num_reports}] {condition} at {address} — {priority}")
        except Exception as e:
            print(f"❌ Error generating report: {e}")

    print(f"\n✅ Generated {generated} reports ({skipped} duplicates filtered).")
    return generated


if __name__ == '__main__':
    generate_random_reports(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
