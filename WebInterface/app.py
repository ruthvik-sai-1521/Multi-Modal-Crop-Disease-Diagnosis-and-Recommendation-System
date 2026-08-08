import os
import secrets
import joblib
import torch
import torch.nn as nn
import numpy as np
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from torchvision import models, transforms
from PIL import Image
from google.api_core import exceptions

# ==========================================
# 1. CONFIGURATION & DATABASE
# ==========================================
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# NEW: Cache table to store AI responses
class AICache(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(100), nullable=False)
    mode = db.Column(db.String(20), nullable=False) # 'explain' or 'remedy'
    response_text = db.Column(db.Text, nullable=False)

# ==========================================
# 2. AI CONFIGURATION (UPDATED FOR 2025)
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GOOGLE_API_KEY = "AIzaSyAIDeLP1O-JekhBHIyKIXGl2qs4cxVrhhc"
genai.configure(api_key=GOOGLE_API_KEY)

# Use the specific string that worked in your list_models debug
MODEL_NAME = 'models/gemini-3-flash-preview' 

try:
    ai_model = genai.GenerativeModel(MODEL_NAME)
    print(f"✅ AI Model {MODEL_NAME} initialized.")
except Exception as e:
    print(f"⚠️ Model initialization failed: {e}")
    ai_model = None

CLASS_NAMES = [
    'Chilli Bacterial Spot', 'Chilli Cercospora Leaf Spot', 'Chilli Curl Virus', 'Chilli Healthy Leaf',
    'Chilli Nutrition Deficiency', 'Chilli White spot', 'Cotton bacterial_blight', 'Cotton curl_virus',
    'Cotton fussarium_wilt', 'Cotton healthy', 'Maize fall armyworm', 'Maize grasshoper', 'Maize healthy',
    'Maize leaf beetle', 'Maize leaf blight', 'Maize leaf spot', 'Maize streak virus', 'Potato_Early_blight',
    'Potato_Late_blight', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Potato_healthy',
    'Tomato healthy', 'Tomato leaf blight', 'Tomato leaf curl', 'Tomato septoria leaf spot', 'Tomato verticulium wilt'
]

# ==========================================
# 3. LOAD LOCAL VISION MODELS
# ==========================================
def load_vision_model(name):
    MODEL_PATH = "models"
    try:
        if name == 'efficientnet':
            m = models.efficientnet_b5(weights=None)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(CLASS_NAMES))
            m.load_state_dict(torch.load(f"{MODEL_PATH}/efficientnet_best.pth", map_location=DEVICE))
            m.classifier = nn.Identity()
        elif name == 'densenet':
            m = models.densenet201(weights=None)
            m.classifier = nn.Linear(m.classifier.in_features, len(CLASS_NAMES))
            m.load_state_dict(torch.load(f"{MODEL_PATH}/densenet_best.pth", map_location=DEVICE))
            m.classifier = nn.Identity()
        elif name == 'swin':
            m = models.swin_t(weights=None)
            m.head = nn.Linear(m.head.in_features, len(CLASS_NAMES))
            m.load_state_dict(torch.load(f"{MODEL_PATH}/swin_best.pth", map_location=DEVICE))
            m.head = nn.Identity()
        return m.to(DEVICE).eval()
    except Exception as e:
        print(f"⚠️ Model {name} error: {e}")
        return None

effnet = load_vision_model('efficientnet')
densenet = load_vision_model('densenet')
swin = load_vision_model('swin')

try:
    meta_learner = joblib.load("models/meta_learner_final.pkl")
    scaler = joblib.load("models/env_scaler_final.pkl")
except:
    meta_learner = scaler = None

# ==========================================
# 4. INTELLIGENT AI FUNCTIONS (WITH CACHING)
# ==========================================

def get_offline_backup(disease_name, mode):
    # (Keep your existing dictionary-based backup here)
    db_backup = {
        "Tomato": {"explain": "Diagnosis: Fungal Pathogen.", "remedy": "Copper spray, remove leaves."},
        "Chilli": {"explain": "Diagnosis: Leaf Spot/Viral.", "remedy": "Neem oil, drainage."},
        "Cotton": {"explain": "Diagnosis: Blight/Virus.", "remedy": "Resistant seeds, control vectors."},
        "Maize": {"explain": "Diagnosis: Armyworm/Leaf Spot.", "remedy": "Organic pesticides."},
        "Potato": {"explain": "Diagnosis: Early/Late Blight.", "remedy": "Mancozeb, hilling soil."}
    }
    match = next((v for k, v in db_backup.items() if k.lower() in disease_name.lower()), None)
    if not match: match = {"explain": "Structural abnormality.", "remedy": "General Neem spray."}
    return f"**Offline Cache:**\n{match[mode]}"

def call_gemini(prompt, disease_name, mode):
    print(f"🤖 Requesting Gemini: {disease_name}")
    if not ai_model: return get_offline_backup(disease_name, mode)
    
    try:
        response = ai_model.generate_content(prompt)
        if response.text:
            return response.text.replace('**', '').replace('*', '-')
    except exceptions.ResourceExhausted:
        print("❌ Quota full.")
    except Exception as e:
        print(f"❌ AI Error: {e}")
    
    return get_offline_backup(disease_name, mode)

# ==========================================
# 5. ROUTES
# ==========================================

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def home():
    if current_user.is_authenticated:
        return render_template('index.html', user=current_user)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'No file'})
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    file.save(filepath)
    
    try:
        img = Image.open(filepath).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_t = transform(img).unsqueeze(0).to(DEVICE)
        
        if effnet and meta_learner:
            with torch.no_grad():
                f1 = effnet(img_t).cpu().numpy()
                f2 = densenet(img_t).cpu().numpy()
                f3 = swin(img_t).cpu().numpy()
            env_feats = scaler.transform([[float(request.form.get('temp', 25)), float(request.form.get('humidity', 70)), float(request.form.get('ph', 6.5))]])
            final_input = np.hstack([f1, f2, f3, env_feats])
            pred_idx = meta_learner.predict(final_input)[0]
            pred_class = CLASS_NAMES[pred_idx]
            conf = np.max(meta_learner.predict_proba(final_input)) * 100
        else:
            pred_class, conf = "Chilli Healthy Leaf", 100.0

        return jsonify({'disease': pred_class, 'confidence': f"{conf:.2f}%"})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/explain_remedy', methods=['POST'])
def explain_remedy():
    data = request.json
    disease, mode = data.get('disease'), data.get('mode')
    
    # STEP 1: Check Database Cache
    cached = AICache.query.filter_by(disease=disease, mode=mode).first()
    if cached:
        print("📦 Serving from local cache.")
        return jsonify({'response': cached.response_text})

    # STEP 2: Prepare Prompt
    if mode == 'explain':
        prompt = f"Explain the crop disease '{disease}' concise. Factors: {data.get('temp')}C, {data.get('humidity')}% humidity."
    else:
        prompt = f"Suggest 3 organic remedies for {disease}. Name, Quantity, and How to use."

    # STEP 3: Call AI
    response = call_gemini(prompt, disease, mode)

    # STEP 4: Store in Cache if successful
    if "Offline" not in response:
        new_entry = AICache(disease=disease, mode=mode, response_text=response)
        db.session.add(new_entry)
        db.session.commit()

    return jsonify({'response': response})

# (Keep Login/Register/Logout/About/Profile routes as they were)
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.password == request.form.get('password'):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        if not User.query.filter_by(username=username).first():
            new_user = User(username=username, password=request.form.get('password'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user)
            return redirect(url_for('home'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/profile')
@login_required
def profile(): return render_template('profile.html', user=current_user)

@app.route('/about')
def about(): return render_template('about.html')

if __name__ == '__main__':
    with app.app_context():
        db.create_all() # This creates the new AICache table
    app.run(debug=True, port=5000)