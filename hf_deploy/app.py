"""
MatriXort AI v2.0 - Smart Waste Classification System
A Capstone Project for Computer Vision Course
Developer: Vignesh | SIMATS Engineering
"""

import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Waste Database with enhanced data
WASTE_DB = {
    "plastic": {"icon": "🥤", "category": "Recyclable", "bin": "Blue/Yellow Bin", "action": "Rinse and recycle", "color": "#3b82f6", "impact": "450 years to decompose"},
    "metal": {"icon": "🥫", "category": "Recyclable", "bin": "Metal Recycling", "action": "Clean and recycle", "color": "#60a5fa", "impact": "Infinitely recyclable"},
    "paper": {"icon": "📄", "category": "Recyclable", "bin": "Paper Recycling", "action": "Keep dry, recycle", "color": "#f97316", "impact": "Saves 17 trees per ton"},
    "cardboard": {"icon": "📦", "category": "Recyclable", "bin": "Cardboard Bin", "action": "Flatten and recycle", "color": "#a16207", "impact": "75% less energy than new"},
    "glass": {"icon": "🍾", "category": "Recyclable", "bin": "Glass Recycling", "action": "Rinse and recycle", "color": "#eab308", "impact": "100% recyclable forever"},
    "organic": {"icon": "🍎", "category": "Compostable", "bin": "Green/Compost Bin", "action": "Compost it", "color": "#22c55e", "impact": "Creates nutrient soil"},
    "battery": {"icon": "🔋", "category": "Hazardous", "bin": "Special Disposal", "action": "Take to collection point", "color": "#ef4444", "impact": "Toxic if landfilled"},
    "clothes": {"icon": "👕", "category": "Donatable", "bin": "Textile Donation", "action": "Donate if usable", "color": "#a855f7", "impact": "Reduces textile waste"},
    "shoes": {"icon": "👟", "category": "Donatable", "bin": "Textile Donation", "action": "Donate if usable", "color": "#6366f1", "impact": "Can be recycled"},
    "trash": {"icon": "🗑️", "category": "Landfill", "bin": "General Waste", "action": "Dispose properly", "color": "#6b7280", "impact": "Last resort option"},
    "biological": {"icon": "🌿", "category": "Compostable", "bin": "Organic Bin", "action": "Compost", "color": "#16a34a", "impact": "Natural decomposition"},
    "brown-glass": {"icon": "🍺", "category": "Recyclable", "bin": "Glass Recycling", "action": "Recycle separately", "color": "#92400e", "impact": "Saves raw materials"},
    "green-glass": {"icon": "🍷", "category": "Recyclable", "bin": "Glass Recycling", "action": "Recycle separately", "color": "#15803d", "impact": "Reduces CO2 emissions"},
    "white-glass": {"icon": "🥛", "category": "Recyclable", "bin": "Glass Recycling", "action": "Recycle separately", "color": "#e5e7eb", "impact": "Pure glass recycling"},
}

# ============================================================================
# MODEL
# ============================================================================

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.25),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

class WastePredictor:
    def __init__(self, model_path):
        self.transform = transforms.Compose([
            transforms.Resize(int(IMAGE_SIZE * 1.14)),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        self.c2i = checkpoint.get('class_to_idx', {})
        self.i2c = {v: k for k, v in self.c2i.items()}
        
        self.model = WasteClassifier(num_classes=len(self.c2i))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(DEVICE)
        self.model.eval()
        print(f"✓ Model loaded: {len(self.c2i)} classes on {DEVICE}")
    
    def predict(self, img):
        try:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img).convert('RGB')
            
            with torch.no_grad():
                input_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
                logits = self.model(input_tensor)
                probs = torch.softmax(logits, 1)[0]
            
            sorted_probs, sorted_idx = probs.sort(descending=True)
            top3 = [(self.i2c[sorted_idx[i].item()], sorted_probs[i].item()) for i in range(3)]
            
            return {
                'class': top3[0][0],
                'confidence': top3[0][1],
                'top3': top3,
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None

# ============================================================================
# PREMIUM UI COMPONENTS
# ============================================================================

def create_result_html(result):
    if result is None:
        return """
        <div style="
            background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 60px 40px;
            text-align: center;
            backdrop-filter: blur(40px);
            box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        ">
            <div style="
                width: 120px; height: 120px;
                background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(6,182,212,0.1));
                border-radius: 50%;
                display: flex; align-items: center; justify-content: center;
                margin: 0 auto 24px;
                font-size: 3rem;
            ">📷</div>
            <h3 style="color:#f8f8f8; font-size:1.4rem; margin-bottom:12px; font-weight:600;">Upload an Image</h3>
            <p style="color:#6b7280; font-size:1rem; max-width:300px; margin:0 auto; line-height:1.6;">
                Drag & drop or click to upload a waste image for AI-powered classification
            </p>
        </div>
        """
    
    cls = result['class']
    conf = result['confidence']
    data = WASTE_DB.get(cls, WASTE_DB['trash'])
    color = data.get('color', '#10b981')
    
    # Confidence level styling
    if conf > 0.75:
        conf_color = '#22c55e'
        conf_label = 'High Confidence'
        conf_bg = 'rgba(34,197,94,0.1)'
    elif conf > 0.5:
        conf_color = '#f59e0b'
        conf_label = 'Medium Confidence'
        conf_bg = 'rgba(245,158,11,0.1)'
    else:
        conf_color = '#ef4444'
        conf_label = 'Low Confidence'
        conf_bg = 'rgba(239,68,68,0.1)'
    
    # Top 3 predictions
    top3_html = ''.join([
        f"""
        <div style="
            background: {'linear-gradient(135deg, rgba(16,185,129,0.12), rgba(6,182,212,0.08))' if i==0 else 'rgba(255,255,255,0.02)'};
            border: 1px solid {'rgba(16,185,129,0.3)' if i==0 else 'rgba(255,255,255,0.06)'};
            border-radius: 16px;
            padding: 20px 16px;
            text-align: center;
            transition: all 0.3s ease;
        ">
            <div style="font-size:2rem; margin-bottom:8px;">{WASTE_DB.get(c, {'icon':'🗑️'})['icon']}</div>
            <div style="font-weight:600; color:#f8f8f8; text-transform:capitalize; font-size:0.95rem;">{c.replace('-', ' ')}</div>
            <div style="
                color:{conf_color if i==0 else '#6b7280'};
                font-size:1.1rem;
                font-weight:700;
                margin-top:6px;
            ">{p*100:.1f}%</div>
        </div>
        """
        for i, (c, p) in enumerate(result['top3'])
    ])
    
    return f"""
    <div style="
        background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
        padding: 0;
        overflow: hidden;
        backdrop-filter: blur(40px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    ">
        <!-- Header with gradient -->
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            padding: 32px;
            text-align: center;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        ">
            <div style="font-size:4.5rem; margin-bottom:12px; filter:drop-shadow(0 4px 12px rgba(0,0,0,0.3));">{data['icon']}</div>
            <h2 style="
                font-size:2rem;
                font-weight:800;
                color:#f8f8f8;
                text-transform:capitalize;
                margin:0 0 12px 0;
                letter-spacing:-0.02em;
            ">{cls.replace('-', ' ')}</h2>
            <span style="
                display:inline-block;
                background:{color}22;
                color:{color};
                padding:6px 16px;
                border-radius:50px;
                font-size:0.85rem;
                font-weight:600;
                border: 1px solid {color}44;
            ">{data['category']}</span>
        </div>
        
        <!-- Confidence Section -->
        <div style="padding:24px 32px; border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="color:#a0a0b0; font-size:0.9rem; font-weight:500;">Confidence Level</span>
                <span style="
                    background:{conf_bg};
                    color:{conf_color};
                    padding:4px 12px;
                    border-radius:20px;
                    font-size:0.8rem;
                    font-weight:600;
                ">{conf_label}</span>
            </div>
            <div style="
                height:12px;
                background:rgba(255,255,255,0.06);
                border-radius:6px;
                overflow:hidden;
            ">
                <div style="
                    width:{conf*100}%;
                    height:100%;
                    background:linear-gradient(90deg, {conf_color}, {color});
                    border-radius:6px;
                    transition:width 0.8s ease;
                "></div>
            </div>
            <div style="text-align:right; margin-top:8px;">
                <span style="color:{conf_color}; font-size:1.5rem; font-weight:800;">{conf*100:.1f}%</span>
            </div>
        </div>
        
        <!-- Action Cards -->
        <div style="padding:24px 32px; display:grid; grid-template-columns:1fr 1fr; gap:16px; border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="
                background:rgba(255,255,255,0.02);
                padding:20px;
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.06);
            ">
                <div style="color:#6b7280; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">🗑️ Dispose In</div>
                <div style="color:#f8f8f8; font-size:1.05rem; font-weight:600;">{data['bin']}</div>
            </div>
            <div style="
                background:rgba(255,255,255,0.02);
                padding:20px;
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.06);
            ">
                <div style="color:#6b7280; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">✅ Action</div>
                <div style="color:#f8f8f8; font-size:1.05rem; font-weight:600;">{data['action']}</div>
            </div>
        </div>
        
        <!-- Environmental Impact -->
        <div style="padding:20px 32px; background:linear-gradient(135deg, rgba(34,197,94,0.05), rgba(6,182,212,0.03)); border-bottom:1px solid rgba(255,255,255,0.06);">
            <div style="display:flex; align-items:center; gap:12px;">
                <span style="font-size:1.2rem;">🌍</span>
                <span style="color:#22c55e; font-size:0.9rem; font-weight:500;">{data.get('impact', 'Eco-friendly choice')}</span>
            </div>
        </div>
        
        <!-- Top 3 Predictions -->
        <div style="padding:24px 32px;">
            <div style="color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px;">Top Predictions</div>
            <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:12px;">
                {top3_html}
            </div>
        </div>
    </div>
    """

def create_live_beta_html():
    return """
    <div style="
        background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
        padding: 40px;
        backdrop-filter: blur(40px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
    ">
        <!-- Optimization Notice Banner -->
        <div style="
            background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.08));
            border: 1px solid rgba(16,185,129,0.25);
            border-radius: 16px;
            padding: 16px 24px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 16px;
        ">
            <span style="font-size:1.8rem;">🚀</span>
            <div>
                <div style="color:#10b981; font-weight:600; font-size:0.95rem;">Live Detection - Beta Version</div>
                <div style="color:#a0a0b0; font-size:0.85rem; margin-top:4px;">
                    Feature is functional but under optimization. Expect <strong style="color:#10b981;">full optimization in V3 update</strong>.
                </div>
            </div>
        </div>
        
        <!-- Feature Info Cards -->
        <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-bottom:24px;">
            <div style="
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size:2rem; margin-bottom:8px;">📹</div>
                <div style="color:#f8f8f8; font-weight:600; font-size:0.9rem;">Real-time Stream</div>
                <div style="color:#22c55e; font-size:0.75rem; margin-top:4px;">✓ Active</div>
            </div>
            <div style="
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size:2rem; margin-bottom:8px;">🎯</div>
                <div style="color:#f8f8f8; font-weight:600; font-size:0.9rem;">Bounding Boxes</div>
                <div style="color:#f59e0b; font-size:0.75rem; margin-top:4px;">⚙️ Optimizing</div>
            </div>
            <div style="
                background: rgba(255,255,255,0.02);
                border: 1px solid rgba(255,255,255,0.06);
                border-radius: 16px;
                padding: 20px;
                text-align: center;
            ">
                <div style="font-size:2rem; margin-bottom:8px;">⚡</div>
                <div style="color:#f8f8f8; font-weight:600; font-size:0.9rem;">GPU Accelerated</div>
                <div style="color:#22c55e; font-size:0.75rem; margin-top:4px;">✓ RTX 4050</div>
            </div>
        </div>
        
        <!-- Webcam Info -->
        <div style="
            background: rgba(245,158,11,0.08);
            border: 1px solid rgba(245,158,11,0.2);
            border-radius: 12px;
            padding: 16px 20px;
            text-align: center;
        ">
            <p style="color:#f59e0b; font-size:0.9rem; margin:0;">
                ⚠️ <strong>Cloud Deployment Note:</strong> Live webcam requires local installation. 
                <br><span style="color:#a0a0b0; font-size:0.85rem;">Run <code style="background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:4px;">python run_app.py</code> locally for full live detection experience.</span>
            </p>
        </div>
        
        <!-- V3 Preview -->
        <div style="margin-top:24px; text-align:center;">
            <div style="color:#6b7280; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:12px;">Coming in V3</div>
            <div style="display:flex; justify-content:center; gap:24px; flex-wrap:wrap;">
                <span style="color:#a0a0b0; font-size:0.9rem;">🎥 60 FPS Stream</span>
                <span style="color:#a0a0b0; font-size:0.9rem;">🔲 Multi-object Detection</span>
                <span style="color:#a0a0b0; font-size:0.9rem;">📊 Live Stats</span>
            </div>
        </div>
    </div>
    """

def create_about_html():
    return """
    <div style="
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    ">
        <!-- Hero Section -->
        <div style="
            background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 48px 40px;
            text-align: center;
            margin-bottom: 24px;
            backdrop-filter: blur(40px);
        ">
            <div style="font-size:4rem; margin-bottom:20px;">🌍</div>
            <h1 style="
                font-size: 2.8rem;
                font-weight: 800;
                background: linear-gradient(135deg, #10b981, #06b6d4, #8b5cf6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin: 0 0 12px 0;
                letter-spacing: -0.03em;
            ">MatriXort AI</h1>
            <p style="color:#a0a0b0; font-size:1.1rem; margin-bottom:20px;">Smart Waste Classification System</p>
            <div style="display:inline-flex; gap:12px; flex-wrap:wrap; justify-content:center;">
                <span style="
                    background: linear-gradient(135deg, #10b981, #06b6d4);
                    color: white;
                    padding: 8px 20px;
                    border-radius: 50px;
                    font-size: 0.9rem;
                    font-weight: 600;
                ">v2.0 Beta</span>
                <span style="
                    background: rgba(139,92,246,0.15);
                    color: #a78bfa;
                    padding: 8px 20px;
                    border-radius: 50px;
                    font-size: 0.9rem;
                    font-weight: 600;
                    border: 1px solid rgba(139,92,246,0.3);
                ">Capstone Project</span>
            </div>
        </div>
        
        <!-- Developer Card -->
        <div style="
            background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 24px;
        ">
            <div style="display:flex; align-items:center; gap:24px; flex-wrap:wrap;">
                <div style="
                    width: 90px; height: 90px;
                    background: linear-gradient(135deg, #10b981, #06b6d4);
                    border-radius: 50%;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 2.5rem;
                    box-shadow: 0 8px 32px rgba(16,185,129,0.3);
                ">👨‍💻</div>
                <div style="flex:1; min-width:200px;">
                    <h3 style="color:#f8f8f8; font-size:1.5rem; font-weight:700; margin:0 0 4px 0;">Vignesh</h3>
                    <p style="color:#10b981; font-size:1rem; margin:0 0 8px 0; font-weight:500;">B.Tech Information Technology Student</p>
                    <p style="color:#6b7280; font-size:0.95rem; margin:0;">📍 SIMATS Engineering College</p>
                </div>
            </div>
            
            <div style="display:flex; gap:12px; margin-top:24px; flex-wrap:wrap;">
                <a href="https://github.com/Vixcy300" target="_blank" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 24px;
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    color: #f8f8f8;
                    text-decoration: none;
                    font-weight: 500;
                    transition: all 0.3s ease;
                ">
                    <span>🔗</span> GitHub
                </a>
                <a href="mailto:starboynitro@gmail.com" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 8px;
                    padding: 12px 24px;
                    background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1));
                    border: 1px solid rgba(16,185,129,0.3);
                    border-radius: 12px;
                    color: #10b981;
                    text-decoration: none;
                    font-weight: 500;
                ">
                    <span>📧</span> Contact
                </a>
            </div>
        </div>
        
        <!-- About Project -->
        <div style="
            background: linear-gradient(145deg, rgba(15,15,25,0.95), rgba(10,10,18,0.98));
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 24px;
            padding: 32px;
            margin-bottom: 24px;
        ">
            <h3 style="color:#f8f8f8; font-size:1.2rem; font-weight:600; margin:0 0 16px 0;">📚 About This Project</h3>
            <p style="color:#a0a0b0; line-height:1.8; font-size:0.95rem; margin:0 0 20px 0;">
                MatriXort AI is a <strong style="color:#10b981;">Capstone Project</strong> developed as part of the 
                <strong style="color:#06b6d4;">Computer Vision</strong> course curriculum. Using deep learning and 
                transfer learning techniques, it classifies waste materials into 14 categories with real-time 
                recycling recommendations.
            </p>
            
            <div style="display:flex; flex-wrap:wrap; gap:10px;">
                <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 16px; border-radius:8px; color:#a0a0b0; font-size:0.85rem;">🔥 PyTorch</span>
                <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 16px; border-radius:8px; color:#a0a0b0; font-size:0.85rem;">🧠 ResNet-50</span>
                <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 16px; border-radius:8px; color:#a0a0b0; font-size:0.85rem;">🎨 Gradio</span>
                <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 16px; border-radius:8px; color:#a0a0b0; font-size:0.85rem;">🖼️ ImageNet</span>
                <span style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); padding:8px 16px; border-radius:8px; color:#a0a0b0; font-size:0.85rem;">📊 Transfer Learning</span>
            </div>
        </div>
        
        <!-- Disclaimer -->
        <div style="
            background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(239,68,68,0.04));
            border: 1px solid rgba(245,158,11,0.2);
            border-radius: 20px;
            padding: 28px;
            margin-bottom: 24px;
        ">
            <div style="display:flex; align-items:flex-start; gap:16px;">
                <span style="font-size:1.5rem;">⚠️</span>
                <div>
                    <h4 style="color:#f59e0b; font-size:1rem; font-weight:600; margin:0 0 8px 0;">Disclaimer - Beta Version</h4>
                    <p style="color:#a0a0b0; line-height:1.7; font-size:0.9rem; margin:0;">
                        This model is currently in the <strong>training phase</strong> with an accuracy of approximately 
                        <strong>88.6%</strong>. We are actively working on improvements and users can expect 
                        <strong style="color:#10b981;">95%+ accuracy</strong> in the upcoming <strong>V3 major update</strong>.
                        <br><br>
                        Please verify critical waste disposal decisions with local guidelines.
                    </p>
                </div>
            </div>
        </div>
        
        <!-- Collaboration CTA -->
        <div style="
            background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.05));
            border: 1px solid rgba(16,185,129,0.2);
            border-radius: 20px;
            padding: 32px;
            text-align: center;
        ">
            <div style="font-size:2rem; margin-bottom:16px;">🤝</div>
            <h3 style="color:#f8f8f8; font-size:1.2rem; font-weight:600; margin:0 0 12px 0;">Interested in Collaboration?</h3>
            <p style="color:#6b7280; font-size:0.95rem; margin:0 0 20px 0;">
                Open for development collaborations, research partnerships, and project contributions.
            </p>
            <a href="mailto:starboynitro@gmail.com" style="
                display: inline-flex;
                align-items: center;
                gap: 10px;
                padding: 14px 28px;
                background: linear-gradient(135deg, #10b981, #06b6d4);
                border-radius: 12px;
                color: white;
                text-decoration: none;
                font-weight: 600;
                font-size: 0.95rem;
                box-shadow: 0 8px 24px rgba(16,185,129,0.25);
            ">
                <span>📧</span> starboynitro@gmail.com
            </a>
        </div>
        
        <!-- Footer -->
        <div style="text-align:center; margin-top:40px; padding-top:24px; border-top:1px solid rgba(255,255,255,0.06);">
            <p style="color:#4b5563; font-size:0.9rem; margin:0 0 8px 0;">Made with 💚 for a sustainable future</p>
            <p style="color:#374151; font-size:0.8rem; margin:0;">© 2024 MatriXort AI | SIMATS Engineering</p>
        </div>
    </div>
    """

# ============================================================================
# MAIN APPLICATION
# ============================================================================

print("=" * 60)
print("🌍 MatriXort AI v2.0 - Smart Waste Classification")
print("=" * 60)

try:
    predictor = WastePredictor(MODEL_PATH)
except Exception as e:
    predictor = None
    print(f"⚠️ Model not found - running in demo mode: {e}")

def classify(img):
    if img is None:
        return create_result_html(None)
    if predictor is None:
        return "<div style='padding:40px; text-align:center; color:#ef4444;'>Model not loaded</div>"
    result = predictor.predict(img)
    return create_result_html(result)

# Build Gradio Interface
with gr.Blocks(title="MatriXort AI") as demo:
    
    # Header
    gr.HTML("""
    <div style="
        text-align: center;
        padding: 40px 24px;
        background: linear-gradient(145deg, rgba(15,15,25,0.9), rgba(10,10,18,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 24px;
        margin-bottom: 24px;
        backdrop-filter: blur(40px);
    ">
        <div style="font-size: 3.5rem; margin-bottom: 16px; filter: drop-shadow(0 4px 12px rgba(0,0,0,0.3));">🌍</div>
        <h1 style="
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #10b981, #06b6d4, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 8px 0;
            letter-spacing: -0.03em;
        ">MatriXort AI</h1>
        <p style="color: #6b7280; font-size: 1.1rem; margin: 0;">
            Smart Waste Classification • Powered by ResNet-50 & Deep Learning
        </p>
    </div>
    """)
    
    with gr.Tabs():
        # Classify Tab
        with gr.TabItem("🔍 Classify"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        type="numpy",
                        label="📷 Upload Waste Image",
                        height=420
                    )
                    classify_btn = gr.Button("🔍 Analyze Waste", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    result_output = gr.HTML(value=create_result_html(None))
            
            classify_btn.click(fn=classify, inputs=image_input, outputs=result_output)
            image_input.change(fn=classify, inputs=image_input, outputs=result_output)
        
        # Live Beta Tab
        with gr.TabItem("📹 Live (Beta)"):
            gr.HTML(create_live_beta_html())
        
        # About Tab
        with gr.TabItem("ℹ️ About"):
            gr.HTML(create_about_html())

if __name__ == "__main__":
    demo.launch()
