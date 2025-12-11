# -*- coding: utf-8 -*-
"""
MatriXort AI - Smart Waste Classification System v2.0
Capstone Project for Computer Vision Course
Developer: Vignesh | SIMATS Engineering
"""

import sys
from pathlib import Path
import json
import gradio as gr
import numpy as np
from PIL import Image, ImageStat, ImageFilter
import torch
import torch.nn as nn
from torchvision import models, transforms
from datetime import datetime
import random
import csv
import io
import base64
import cv2  # For bounding box overlays

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
STATS_FILE = MODEL_DIR / "user_stats.json"
HISTORY_FILE = MODEL_DIR / "prediction_history.json"

MODEL_VERSION = "v2.0 - Dec 2025"
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Thresholds
HIGH_CONFIDENCE = 0.80
LOW_CONFIDENCE = 0.50
MAX_HISTORY = 20

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    try:
        GPU_NAME = torch.cuda.get_device_name(0)
    except:
        GPU_NAME = "NVIDIA GeForce RTX 4050"
    print(f"Detected GPU: {GPU_NAME}")
else:
    DEVICE = torch.device("cpu")
    GPU_NAME = "CPU"

# Bounding box colors for each waste category (BGR format for OpenCV)
BBOX_COLORS = {
    "plastic": (216, 180, 0),      # Blue #00B4D8
    "metal": (228, 202, 72),       # Cyan #48CAE4
    "paper": (0, 127, 247),        # Orange #F77F00
    "cardboard": (67, 107, 155),   # Brown #9B6B43
    "glass": (73, 191, 252),       # Yellow #FCBF49
    "organic": (160, 214, 6),      # Green #06D6A0
    "battery": (68, 68, 239),      # Red for hazardous
    "clothes": (247, 85, 168),     # Purple #A855F7
    "shoes": (241, 102, 99),       # Indigo #6366F1
    "trash": (128, 128, 128),      # Gray
    "biological": (106, 166, 22),  # Green
    "brown-glass": (46, 94, 139),  # Brown
    "green-glass": (84, 101, 22),  # Dark green
    "white-glass": (235, 231, 224) # Light gray
}

# ============================================================================
# WASTE DATABASE
# ============================================================================

WASTE_DB = {
    "plastic": {
        "icon": "🥤", "category": "Recyclable", "bin": "Plastic Recycling (Blue/Yellow)",
        "color": "#3b82f6", "decomp": 450, "co2": 1.5, "water": 100,
        "prep": ["Empty and rinse", "Remove caps", "Check recycling number", "Flatten"],
        "dont": ["No plastic bags", "No styrofoam", "No food residue"],
        "contamination": ["Remove food residue", "Rinse container thoroughly"],
        "examples": ["Water bottles", "Detergent bottles", "Food containers"]
    },
    "metal": {
        "icon": "🥫", "category": "Recyclable", "bin": "Metal Recycling (Blue/Yellow)",
        "color": "#6b7280", "decomp": 200, "co2": 9.0, "water": 40,
        "prep": ["Rinse cans", "Crush to save space", "Include clean foil"],
        "dont": ["No aerosol cans", "No paint cans", "No electronics"],
        "contamination": ["Empty completely", "Rinse food residue"],
        "examples": ["Aluminum cans", "Food tins", "Metal bottle caps"]
    },
    "glass": {
        "icon": "🍾", "category": "Recyclable", "bin": "Glass Recycling (Green)",
        "color": "#06b6d4", "decomp": 1000000, "co2": 0.3, "water": 50,
        "prep": ["Rinse bottles", "Remove caps", "Sort by color if needed"],
        "dont": ["No window glass", "No mirrors", "No ceramics"],
        "contamination": ["Remove food residue", "No broken glass mixed with other waste"],
        "examples": ["Glass bottles", "Jars", "Glass containers"]
    },
    "paper": {
        "icon": "📰", "category": "Recyclable", "bin": "Paper Recycling (Blue)",
        "color": "#f59e0b", "decomp": 2, "co2": 1.0, "water": 26,
        "prep": ["Keep clean and dry", "Remove staples", "Flatten"],
        "dont": ["No wet paper", "No waxed paper", "No tissues"],
        "contamination": ["Must be dry", "No food stains"],
        "examples": ["Newspapers", "Magazines", "Office paper"]
    },
    "cardboard": {
        "icon": "📦", "category": "Recyclable", "bin": "Cardboard Recycling (Brown)",
        "color": "#92400e", "decomp": 2, "co2": 1.2, "water": 31,
        "prep": ["Flatten boxes", "Remove tape", "Remove plastic inserts"],
        "dont": ["No greasy boxes", "No wax-coated", "No wet cardboard"],
        "contamination": ["Pizza boxes with grease = trash", "Remove food residue"],
        "examples": ["Shipping boxes", "Cereal boxes", "Egg cartons"]
    },
    "organic": {
        "icon": "🍌", "category": "Compostable", "bin": "Compost / Green Bin",
        "color": "#22c55e", "decomp": 0.5, "co2": 0.5, "water": 5,
        "prep": ["Collect in bin", "Drain liquids", "Remove stickers"],
        "dont": ["No meat/dairy (usually)", "No oils", "No diseased plants"],
        "contamination": [],
        "examples": ["Fruit peels", "Vegetable scraps", "Coffee grounds"]
    },
    "battery": {
        "icon": "🔋", "category": "Hazardous", "bin": "⚠️ Hazardous Waste Center",
        "color": "#ef4444", "decomp": 100, "co2": 5.0, "water": 200,
        "prep": ["Tape terminals", "Store safely", "Take to collection point"],
        "dont": ["⚠️ NEVER regular trash!", "Don't puncture", "Don't burn"],
        "contamination": [],
        "examples": ["AA/AAA batteries", "Phone batteries", "Car batteries"]
    },
    "clothes": {
        "icon": "👕", "category": "Donatable", "bin": "Textile Donation",
        "color": "#a855f7", "decomp": 40, "co2": 3.6, "water": 2700,
        "prep": ["Wash before donating", "Repair if possible", "Bag for textile bin"],
        "dont": ["Don't trash wearable items", "No wet/moldy", "No heavily stained"],
        "contamination": [],
        "examples": ["T-shirts", "Pants", "Jackets", "Socks"]
    },
    "shoes": {
        "icon": "👟", "category": "Donatable", "bin": "Textile Donation / Nike Grind",
        "color": "#6366f1", "decomp": 50, "co2": 2.5, "water": 1800,
        "prep": ["Pair together", "Clean", "Check brand programs"],
        "dont": ["No single shoes", "Don't regular trash"],
        "contamination": [],
        "examples": ["Sneakers", "Boots", "Sandals"]
    },
    "trash": {
        "icon": "🗑️", "category": "Landfill", "bin": "General Waste (Black/Gray)",
        "color": "#374151", "decomp": 500, "co2": 0.1, "water": 1,
        "prep": ["Confirm not recyclable", "Bag securely"],
        "dont": ["Don't include recyclables", "No hazardous materials"],
        "contamination": [],
        "examples": ["Chip bags", "Styrofoam", "Mixed plastics"]
    },
    "biological": {
        "icon": "🌱", "category": "Compostable", "bin": "Compost / Organic Bin",
        "color": "#16a34a", "decomp": 0.5, "co2": 0.5, "water": 5,
        "prep": ["Collect yard waste", "Chop large branches"],
        "dont": ["No treated wood", "No invasive plants"],
        "contamination": [],
        "examples": ["Leaves", "Grass clippings", "Plant trimmings"]
    },
    "brown-glass": {
        "icon": "🍺", "category": "Recyclable", "bin": "Glass Recycling (Brown)",
        "color": "#78350f", "decomp": 1000000, "co2": 0.3, "water": 50,
        "prep": ["Rinse", "Remove caps"],
        "dont": ["No ceramics", "No drinking glasses"],
        "contamination": ["Rinse beer/medicine bottles"],
        "examples": ["Beer bottles", "Medicine bottles"]
    },
    "green-glass": {
        "icon": "🍷", "category": "Recyclable", "bin": "Glass Recycling (Green)",
        "color": "#166534", "decomp": 1000000, "co2": 0.3, "water": 50,
        "prep": ["Rinse", "Remove corks"],
        "dont": ["No window glass", "No light bulbs"],
        "contamination": [],
        "examples": ["Wine bottles", "Olive oil bottles"]
    },
    "white-glass": {
        "icon": "🫙", "category": "Recyclable", "bin": "Glass Recycling (Clear)",
        "color": "#e5e7eb", "decomp": 1000000, "co2": 0.35, "water": 55,
        "prep": ["Rinse jars", "Remove lids"],
        "dont": ["No Pyrex", "No mirrors"],
        "contamination": ["Rinse food jars"],
        "examples": ["Jam jars", "Sauce jars", "Clear bottles"]
    }
}

TIPS = [
    "💡 Rinse containers before recycling!",
    "💡 Flatten cardboard to save space.",
    "💡 Remove caps - they're different plastic.",
    "💡 When in doubt, check local guidelines.",
    "💡 Reduce > Reuse > Recycle!",
    "💡 Batteries can cause landfill fires!",
    "💡 Glass is infinitely recyclable.",
    "💡 Composting diverts 30% of waste.",
]

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
    def __init__(self, path):
        print(f"Loading model: {path}")
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
        self.c2i = ckpt['class_to_idx']
        self.i2c = {v: k for k, v in self.c2i.items()}
        
        self.model = WasteClassifier(len(self.c2i))
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(DEVICE)
        self.use_half = False
        if DEVICE.type == 'cuda':
            self.model.half()
            self.use_half = True
            try:
                # Enable cuDNN benchmark for optimized kernel selection
                torch.backends.cudnn.benchmark = True
            except:
                pass
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
        print(f"Model ready: {len(self.c2i)} classes (FP16: {self.use_half})")
    
    def predict(self, img):
        try:
            if isinstance(img, np.ndarray):
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
                img = Image.fromarray(img).convert('RGB')
            
            with torch.no_grad():
                input_tensor = self.transform(img).unsqueeze(0).to(DEVICE)
                if self.use_half:
                    input_tensor = input_tensor.half()
                
                # Get raw logits from model
                logits = self.model(input_tensor)
                
                # Apply temperature scaling for better confidence calibration
                # Higher temperature = less overconfident predictions
                calibrated_logits = logits / CALIBRATION_TEMPERATURE
                probs = torch.softmax(calibrated_logits, 1)[0]

            
            sorted_probs, sorted_idx = probs.sort(descending=True)
            top3 = [(self.i2c[sorted_idx[i].item()], sorted_probs[i].item()) for i in range(3)]
            
            return {
                'class': top3[0][0],
                'confidence': top3[0][1],
                'top3': top3,
                'all_probs': {self.i2c[i]: probs[i].item() for i in range(len(self.c2i))}
            }
        except Exception as e:
            print(f"Prediction error: {e}")
            return None


# ============================================================================
# IMAGE QUALITY CHECKER
# ============================================================================

def check_image_quality(img):
    """Check image quality and return warnings."""
    warnings = []
    
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img.astype(np.uint8))
    
    # Check brightness
    stat = ImageStat.Stat(img.convert('L'))
    brightness = stat.mean[0]
    if brightness < 40:
        warnings.append("⚠️ Image is too dark - consider better lighting")
    elif brightness > 220:
        warnings.append("⚠️ Image is overexposed - reduce lighting")
    
    # Check blur using Laplacian variance
    gray = np.array(img.convert('L'))
    laplacian_var = np.var(np.array(Image.fromarray(gray).filter(ImageFilter.FIND_EDGES)))
    if laplacian_var < 100:
        warnings.append("⚠️ Image may be blurry - try refocusing")
    
    # Check size
    if img.size[0] < 100 or img.size[1] < 100:
        warnings.append("⚠️ Image resolution is low - use higher quality")
    
    return warnings


# ============================================================================
# FEATURE: IMAGE QUALITY AUTO-ENHANCEMENT
# ============================================================================

def enhance_image_quality(img):
    """
    Automatically enhance image quality for better classification.
    Applies: CLAHE contrast, brightness correction, slight sharpening.
    Returns: (enhanced_image, enhancement_info_dict)
    """
    if img is None:
        return None, {}
    
    # Convert to numpy if needed
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    original = img.copy()
    enhanced = img.copy()
    enhancements_applied = []
    
    # 1. Check brightness and apply correction if needed
    gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    if mean_brightness < 80:  # Dark image
        # Apply gamma correction to brighten
        gamma = 1.5
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        enhancements_applied.append(f"Brightness +{int((gamma-1)*100)}%")
    
    # 2. Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    # Convert to LAB color space for better results
    lab = cv2.cvtColor(enhanced, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(l_channel)
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    enhancements_applied.append("CLAHE Contrast")
    
    # 3. Apply slight sharpening for blurry images
    # Check blur level first
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 500:  # Blurry image
        # Unsharp mask sharpening
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.3, gaussian, -0.3, 0)
        enhancements_applied.append("Sharpening")
    
    # 4. Slight denoise
    enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 5, 5, 7, 21)
    enhancements_applied.append("Denoise")
    
    enhancement_info = {
        "original_brightness": mean_brightness,
        "enhancements": enhancements_applied,
        "enhanced": len(enhancements_applied) > 1
    }
    
    return enhanced, enhancement_info


# ============================================================================
# FEATURE: UNCERTAINTY QUANTIFICATION
# ============================================================================

def calculate_uncertainty(probs_dict):
    """
    Calculate model uncertainty metrics from prediction probabilities.
    Returns: uncertainty info dict with entropy, top-2 gap, etc.
    """
    # Convert dict values to numpy array
    probs = np.array(list(probs_dict.values()))
    classes = list(probs_dict.keys())
    
    # Ensure probs sum to 1
    probs = probs / (probs.sum() + 1e-10)
    
    # 1. Calculate prediction entropy (higher = more uncertain)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(len(probs))  # Maximum possible entropy
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # 2. Calculate top-2 gap (larger gap = more confident)
    sorted_indices = np.argsort(probs)[::-1]
    top1_prob = probs[sorted_indices[0]]
    top2_prob = probs[sorted_indices[1]] if len(probs) > 1 else 0
    top2_gap = top1_prob - top2_prob
    
    top1_class = classes[sorted_indices[0]]
    top2_class = classes[sorted_indices[1]] if len(probs) > 1 else None
    
    # 3. Determine confidence level
    certainty_percent = (1 - normalized_entropy) * 100
    
    if certainty_percent >= 80:
        confidence_level = "high"
        confidence_icon = "✅"
        confidence_text = "High Confidence"
    elif certainty_percent >= 50:
        confidence_level = "medium"
        confidence_icon = "⚠️"
        confidence_text = "Moderate Confidence"
    else:
        confidence_level = "low"
        confidence_icon = "❌"
        confidence_text = "Low Confidence - Manual Review Suggested"
    
    # 4. Check for confusing categories (similar predictions)
    confusion_warning = None
    if top2_gap < 0.3 and top2_class:
        confusion_warning = f"{top1_class.title()} and {top2_class.title()} look similar"
    
    return {
        "entropy": entropy,
        "certainty_percent": certainty_percent,
        "top2_gap": top2_gap,
        "top1_class": top1_class,
        "top2_class": top2_class,
        "confidence_level": confidence_level,
        "confidence_icon": confidence_icon,
        "confidence_text": confidence_text,
        "confusion_warning": confusion_warning,
        "needs_review": certainty_percent < 50
    }


# ============================================================================
# FEATURE: RECYCLING LOCATION MAPS
# ============================================================================

RECYCLING_SEARCH_TERMS = {
    "plastic": "plastic recycling center",
    "metal": "metal recycling center",
    "glass": "glass recycling center",
    "paper": "paper recycling center",
    "cardboard": "cardboard recycling center",
    "organic": "composting facility",
    "battery": "battery recycling hazardous waste",
    "clothes": "clothing donation center",
    "shoes": "shoe recycling donation",
    "trash": "waste disposal facility",
    "brown-glass": "glass recycling center",
    "green-glass": "glass recycling center",
    "white-glass": "glass recycling center",
    "biological": "composting facility",
}

def get_recycling_location_link(waste_class):
    """
    Generate a Google Maps search link for nearby recycling facilities.
    """
    search_term = RECYCLING_SEARCH_TERMS.get(waste_class.lower(), "recycling center")
    # URL encode the search term
    encoded_term = search_term.replace(" ", "+")
    
    # Google Maps search URL (will use user's location)
    maps_url = f"https://www.google.com/maps/search/{encoded_term}+near+me"
    
    return {
        "url": maps_url,
        "search_term": search_term,
        "button_text": f"🗺️ Find {search_term.title()} Near Me"
    }


# ============================================================================
# FEATURE: TEMPERATURE SCALING FOR CONFIDENCE CALIBRATION
# ============================================================================

# Temperature > 1 makes model less overconfident (more realistic)
# Temperature < 1 makes model more confident
# Typical values: 1.2 - 2.0 for better calibration
CALIBRATION_TEMPERATURE = 1.1  # Reduced from 1.5 - allows proper confidence display


# ============================================================================
# BOUNDING BOX & EDGE DETECTION FOR LIVE MODE
# ============================================================================

def find_object_bbox(frame):
    """
    Find bounding box of main object using edge detection.
    Since we have a classification model (not detection), we use saliency/edge detection.
    """
    try:
        h, w = frame.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect broken lines
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Only use if contour is significant (at least 5% of frame)
            min_area = w * h * 0.05
            if area > min_area:
                x, y, bw, bh = cv2.boundingRect(largest)
                # Add some padding
                pad = 15
                x = max(0, x - pad)
                y = max(0, y - pad)
                bw = min(w - x, bw + 2 * pad)
                bh = min(h - y, bh + 2 * pad)
                return (x, y, bw, bh)
        
        # Fallback: center crop (assume object is in middle 60% of frame)
        margin_x = int(w * 0.2)
        margin_y = int(h * 0.2)
        return (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    
    except Exception as e:
        print(f"BBox detection error: {e}")
        # Return center 60% as fallback
        h, w = frame.shape[:2]
        return (int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6))


def draw_annotated_frame(frame, pred_class, confidence, show_crosshair=True):
    """
    Draw bounding box, label, and crosshair on frame.
    Returns annotated frame as numpy array.
    """
    try:
        # Make a copy to avoid modifying original
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        
        # Get bounding box
        x, y, bw, bh = find_object_bbox(frame)
        
        # Get color for this class (BGR)
        color = BBOX_COLORS.get(pred_class, (0, 255, 0))
        
        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 3)
        
        # Draw corner accents for style
        corner_len = 20
        thick = 4
        # Top-left
        cv2.line(annotated, (x, y), (x + corner_len, y), color, thick)
        cv2.line(annotated, (x, y), (x, y + corner_len), color, thick)
        # Top-right
        cv2.line(annotated, (x + bw, y), (x + bw - corner_len, y), color, thick)
        cv2.line(annotated, (x + bw, y), (x + bw, y + corner_len), color, thick)
        # Bottom-left
        cv2.line(annotated, (x, y + bh), (x + corner_len, y + bh), color, thick)
        cv2.line(annotated, (x, y + bh), (x, y + bh - corner_len), color, thick)
        # Bottom-right
        cv2.line(annotated, (x + bw, y + bh), (x + bw - corner_len, y + bh), color, thick)
        cv2.line(annotated, (x + bw, y + bh), (x + bw, y + bh - corner_len), color, thick)
        
        # Prepare label text
        label = f"{pred_class.upper().replace('-', ' ')} {confidence*100:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background (semi-transparent)
        label_bg_y1 = max(0, y - text_h - 12)
        label_bg_y2 = y
        cv2.rectangle(annotated, (x, label_bg_y1), (x + text_w + 10, label_bg_y2), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x + 5, y - 5), font, font_scale, (255, 255, 255), font_thickness)
        
        # Draw center crosshair if enabled
        if show_crosshair:
            cx, cy = w // 2, h // 2
            cross_size = 20
            cross_color = (200, 200, 200)
            cv2.line(annotated, (cx - cross_size, cy), (cx + cross_size, cy), cross_color, 1)
            cv2.line(annotated, (cx, cy - cross_size), (cx, cy + cross_size), cross_color, 1)
            # Small circle at center
            cv2.circle(annotated, (cx, cy), 4, cross_color, 1)
        
        return annotated
    
    except Exception as e:
        print(f"Frame annotation error: {e}")
        return frame


# ============================================================================
# STATS & HISTORY
# ============================================================================

def load_stats():
    try:
        return json.load(open(STATS_FILE)) if STATS_FILE.exists() else {}
    except:
        return {}

def save_stats(s):
    json.dump(s, open(STATS_FILE, 'w'), indent=2)

def load_history():
    try:
        return json.load(open(HISTORY_FILE)) if HISTORY_FILE.exists() else []
    except:
        return []

def save_history(h):
    json.dump(h[-MAX_HISTORY:], open(HISTORY_FILE, 'w'), indent=2)

def export_history_csv(history):
    """Export history to CSV string."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Timestamp', 'Predicted Class', 'Confidence', 'Recycling Bin'])
    for item in history:
        data = WASTE_DB.get(item['class'], {})
        writer.writerow([
            item.get('timestamp', 'N/A'),
            item['class'].replace('-', ' ').title(),
            f"{item['confidence']*100:.1f}%",
            data.get('bin', 'N/A')
        ])
    return output.getvalue()


# ============================================================================
# CSS - DARK & LIGHT MODE
# ============================================================================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ============ MINIMAL GSAP-STYLE ANIMATIONS (CSS Only) ============ */

/* 1. PAGE LOAD - Staggered Fade In */
@keyframes fadeInUp {
    from { 
        opacity: 0; 
        transform: translateY(20px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

/* 3. PREDICTION RESULTS - Slide Up */
@keyframes slideUp {
    from { 
        opacity: 0; 
        transform: translateY(30px); 
    }
    to { 
        opacity: 1; 
        transform: translateY(0); 
    }
}

/* 5. IMAGE UPLOAD - Subtle Scale */
@keyframes scaleIn {
    from { 
        opacity: 0; 
        transform: scale(0.95); 
    }
    to { 
        opacity: 1; 
        transform: scale(1); 
    }
}

/* 4. CONFIDENCE BARS - Animated Fill (handled by transition) */
@keyframes fillWidth {
    from { width: 0%; }
}

/* Smooth easing curve for all animations */
:root {
    --ease-smooth: cubic-bezier(0.4, 0, 0.2, 1);
    
    /* Premium Dark Theme */
    --bg-base: #0a0a0c;
    --bg-card: rgba(18, 18, 24, 0.85);
    --bg-elevated: rgba(28, 28, 38, 0.9);
    --bg-glass: rgba(255, 255, 255, 0.02);
    --bg-hover: rgba(255, 255, 255, 0.06);
    --text-primary: #f8f8f8;
    --text-secondary: #a0a0b0;
    --text-muted: #5a5a6a;
    --border: rgba(255, 255, 255, 0.06);
    --border-glow: rgba(16, 185, 129, 0.25);
    --accent: #10b981;
    --accent-glow: rgba(16, 185, 129, 0.35);
    --accent-hover: #059669;
    --warning: #f59e0b;
    --danger: #ef4444;
    --info: #3b82f6;
    --success: #22c55e;
    --purple: #8b5cf6;
    --cyan: #06b6d4;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
    --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.4);
    --shadow-lg: 0 16px 48px rgba(0, 0, 0, 0.5);
    --shadow-glow: 0 0 30px rgba(16, 185, 129, 0.12);
}

* { 
    box-sizing: border-box; 
    margin: 0;
    padding: 0;
    /* Smooth all transitions by default */
    transition-timing-function: var(--ease-smooth);
}

body, .gradio-container {
    background: var(--bg-base) !important;
    background-image: 
        radial-gradient(ellipse 100% 60% at 50% -30%, rgba(16, 185, 129, 0.08), transparent),
        radial-gradient(ellipse 50% 30% at 100% 0%, rgba(59, 130, 246, 0.05), transparent) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
    min-height: 100vh;
    -webkit-font-smoothing: antialiased;
    letter-spacing: -0.01em;
}

.gradio-container { 
    max-width: 1300px !important; 
    margin: 0 auto !important; 
    padding: 24px !important;
    animation: fadeInUp 0.8s var(--ease-out-expo);
}

/* ============ PREMIUM HEADER ============ */
.app-header {
    background: linear-gradient(145deg, rgba(15, 15, 20, 0.98) 0%, rgba(10, 10, 15, 0.98) 50%, rgba(8, 8, 12, 0.98) 100%);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    padding: 48px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4), 0 0 40px rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.06);
}

.app-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.08'/%3E%3C/svg%3E");
    pointer-events: none;
}

.header-content { position: relative; z-index: 1; }
.header-top { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 16px; }
.header-main { text-align: center; }

.header-title {
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #10b981, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
    filter: drop-shadow(0 4px 20px rgba(16, 185, 129, 0.3));
}

.header-subtitle {
    color: rgba(160, 160, 176, 0.95);
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
    letter-spacing: 0.02em;
}

.header-badge {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 182, 212, 0.15));
    backdrop-filter: blur(10px);
    color: #10b981;
    padding: 8px 16px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.3); }
    50% { box-shadow: 0 0 0 8px rgba(255, 255, 255, 0); }
}

.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    backdrop-filter: blur(10px);
}

.status-online { 
    background: rgba(34, 197, 94, 0.2); 
    color: #4ade80;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

/* ============ GLASSMORPHISM CARDS ============ */
.result-panel {
    background: var(--bg-card);
    backdrop-filter: blur(40px);
    -webkit-backdrop-filter: blur(40px);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 28px;
    min-height: 600px;
    box-shadow: var(--shadow-md);
    /* 2. HOVER - Smooth 0.3s transition */
    transition: all 0.3s var(--ease-smooth);
    /* 3. PREDICTION RESULTS - Slide up when appears */
    animation: slideUp 0.5s ease-out;
}

.result-panel:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow-lg);
}

/* Stats Bar */
.stats-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 24px;
}

.stat-card {
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    /* 2. HOVER - Smooth 0.3s transition */
    transition: all 0.3s var(--ease-smooth);
    position: relative;
    overflow: hidden;
    /* 1. PAGE LOAD - Staggered fade in */
    animation: fadeInUp 0.6s ease-out backwards;
}

/* Staggered entrance: 0.1s between each */
.stat-card:nth-child(1) { animation-delay: 0.0s; }
.stat-card:nth-child(2) { animation-delay: 0.1s; }
.stat-card:nth-child(3) { animation-delay: 0.2s; }
.stat-card:nth-child(4) { animation-delay: 0.3s; }

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--cyan));
    opacity: 0;
    transition: opacity 0.3s var(--ease-smooth);
}

/* 2. HOVER - Smooth lift (NOT bouncy) */
.stat-card:hover {
    border-color: var(--border-glow);
    transform: scale(1.02);
    box-shadow: var(--shadow-md);
}

.stat-card:hover::before { opacity: 1; }

.stat-value { 
    font-size: 1.8rem; 
    font-weight: 800; 
    background: linear-gradient(135deg, var(--accent), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.stat-label { 
    font-size: 0.75rem; 
    color: var(--text-muted); 
    margin-top: 6px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Classification Header */
.class-header {
    display: flex;
    align-items: center;
    gap: 20px;
    padding: 28px;
    background: linear-gradient(135deg, var(--bg-elevated), var(--bg-glass));
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.class-header::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, var(--accent-glow), transparent 70%);
    opacity: 0.3;
    pointer-events: none;
}

.class-icon { 
    font-size: 4rem; 
    filter: drop-shadow(0 4px 12px rgba(0, 0, 0, 0.3));
}

.class-details h2 { 
    font-size: 1.8rem; 
    font-weight: 800; 
    margin: 0 0 10px 0; 
    text-transform: capitalize;
    letter-spacing: -0.01em;
}

.class-meta { 
    display: flex; 
    gap: 14px; 
    align-items: center; 
    flex-wrap: wrap; 
}

/* Premium Badges */
.badge {
    padding: 6px 14px;
    border-radius: 50px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    border: 1px solid transparent;
}

.badge-success { 
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.1)); 
    color: #4ade80;
    border-color: rgba(34, 197, 94, 0.3);
}

.badge-warning { 
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2), rgba(245, 158, 11, 0.1)); 
    color: #fbbf24;
    border-color: rgba(245, 158, 11, 0.3);
}

.badge-danger { 
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(239, 68, 68, 0.1)); 
    color: #f87171;
    border-color: rgba(239, 68, 68, 0.3);
}

.badge-info { 
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.2), rgba(59, 130, 246, 0.1)); 
    color: #60a5fa;
    border-color: rgba(59, 130, 246, 0.3);
}

/* Confidence Bar */
.conf-bar-container { 
    width: 120px; 
    height: 8px; 
    background: var(--bg-hover); 
    border-radius: 10px;
    overflow: hidden;
}

.conf-bar { 
    height: 100%; 
    border-radius: 10px; 
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 0 10px currentColor;
}

/* Alert Boxes */
.warning-box {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-left: 4px solid var(--warning);
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 24px;
    backdrop-filter: blur(10px);
}

.quality-warning {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 16px;
    padding: 14px 20px;
    margin-bottom: 20px;
    font-size: 0.9rem;
    color: #f87171;
    backdrop-filter: blur(10px);
}

.contamination-warning {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-radius: 16px;
    padding: 16px 20px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
}

.contamination-title { 
    color: #fbbf24; 
    font-weight: 700; 
    margin-bottom: 8px;
    font-size: 0.95rem;
}

/* Unable Box */
.unable-box {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(239, 68, 68, 0.05));
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: 24px;
    padding: 48px;
    text-align: center;
    margin-bottom: 24px;
    backdrop-filter: blur(20px);
}

.unable-icon { font-size: 5rem; margin-bottom: 16px; opacity: 0.8; }
.unable-title { font-size: 1.5rem; font-weight: 800; color: #f87171; margin-bottom: 10px; }
.unable-text { color: var(--text-secondary); max-width: 450px; margin: 0 auto; line-height: 1.6; }

/* Bin Info */
.bin-info {
    display: flex;
    align-items: center;
    gap: 16px;
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 24px;
    transition: all 0.3s;
}

.bin-info:hover {
    border-color: var(--border-glow);
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
}

.bin-color { 
    width: 36px; 
    height: 36px; 
    border-radius: 10px; 
    border: 2px solid rgba(255, 255, 255, 0.1);
    box-shadow: var(--shadow-sm);
}

.bin-text { 
    font-weight: 700; 
    font-size: 1rem;
}

/* Section */
.section { margin-bottom: 20px; }

.section-header { 
    display: flex; 
    align-items: center; 
    gap: 10px; 
    font-weight: 700; 
    margin-bottom: 14px; 
    font-size: 1rem;
    color: var(--text-primary);
}

.section-content { 
    background: var(--bg-glass); 
    backdrop-filter: blur(20px);
    border: 1px solid var(--border); 
    border-radius: 16px; 
    padding: 18px 24px;
    transition: all 0.3s;
}

.section-content:hover {
    border-color: rgba(255, 255, 255, 0.12);
}

/* Lists */
.info-list { list-style: none; padding: 0; margin: 0; }

.info-list li { 
    display: flex; 
    align-items: flex-start; 
    gap: 12px; 
    padding: 12px 0; 
    color: var(--text-secondary); 
    border-bottom: 1px solid var(--border); 
    font-size: 0.95rem;
    line-height: 1.5;
}

.info-list li:last-child { border-bottom: none; }

.check { color: #4ade80; font-weight: bold; }
.cross { color: #f87171; font-weight: bold; }

/* Impact Grid */
.impact-grid { 
    display: grid; 
    grid-template-columns: repeat(3, 1fr); 
    gap: 14px; 
    margin-bottom: 20px; 
}

.impact-card { 
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border); 
    border-radius: 16px; 
    padding: 20px; 
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.impact-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-glow);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.impact-value { 
    font-size: 1.6rem; 
    font-weight: 800;
    background: linear-gradient(135deg, var(--info), var(--purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.impact-label { 
    font-size: 0.75rem; 
    color: var(--text-muted); 
    margin-top: 4px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Top 3 Predictions */
.top3-section { margin-bottom: 24px; }

.top3-grid { 
    display: grid; 
    grid-template-columns: repeat(3, 1fr); 
    gap: 14px; 
}

.top3-card { 
    background: var(--bg-glass);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border); 
    border-radius: 16px; 
    padding: 18px; 
    text-align: center;
    /* 2. HOVER - Smooth 0.3s transition */
    transition: all 0.3s var(--ease-smooth);
    cursor: default;
    /* 1. PAGE LOAD - Staggered fade in */
    animation: fadeInUp 0.6s ease-out backwards;
}

/* Staggered entrance: 0.1s between each */
.top3-card:nth-child(1) { animation-delay: 0.2s; }
.top3-card:nth-child(2) { animation-delay: 0.3s; }
.top3-card:nth-child(3) { animation-delay: 0.4s; }

/* 2. HOVER - Smooth lift (NOT bouncy) */
.top3-card:hover {
    transform: scale(1.02);
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: var(--shadow-md);
}

.top3-card.active { 
    border-color: var(--accent);
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(16, 185, 129, 0.05));
    box-shadow: var(--shadow-glow);
}

.top3-icon { font-size: 2rem; filter: drop-shadow(0 2px 8px rgba(0, 0, 0, 0.2)); }
.top3-class { font-size: 0.9rem; font-weight: 700; margin-top: 8px; text-transform: capitalize; }
.top3-conf { font-size: 0.8rem; color: var(--text-muted); margin-top: 4px; font-weight: 500; }

/* Tip Box */
.tip-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(139, 92, 246, 0.1));
    border: 1px solid rgba(59, 130, 246, 0.2);
    border-radius: 16px;
    padding: 18px 24px;
    margin-top: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    font-size: 0.95rem;
    color: var(--text-secondary);
    line-height: 1.5;
    backdrop-filter: blur(10px);
}

/* History Section */
.history-section {
    margin-top: 28px;
    background: var(--bg-card);
    backdrop-filter: blur(40px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: var(--shadow-sm);
}

.history-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.history-title { 
    font-weight: 700; 
    font-size: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.history-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.9rem;
}

.history-table th {
    text-align: left;
    padding: 14px 16px;
    background: var(--bg-elevated);
    color: var(--text-muted);
    font-weight: 600;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-bottom: 1px solid var(--border);
}

.history-table th:first-child { border-radius: 12px 0 0 0; }
.history-table th:last-child { border-radius: 0 12px 0 0; }

.history-table td {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    color: var(--text-secondary);
}

.history-table tr:last-child td { border-bottom: none; }

.history-table tr:hover td { 
    background: var(--bg-hover);
}

/* Empty State */
.empty-state { 
    text-align: center; 
    padding: 80px 32px;
}

.empty-icon { 
    font-size: 5rem; 
    margin-bottom: 20px; 
    opacity: 0.3;
    filter: grayscale(50%);
}

.empty-title { 
    font-size: 1.4rem; 
    font-weight: 700; 
    margin-bottom: 10px;
}

.empty-text { 
    color: var(--text-muted); 
    max-width: 320px; 
    margin: 0 auto;
    line-height: 1.6;
}

/* Examples Gallery */
.examples-section {
    background: var(--bg-card);
    backdrop-filter: blur(40px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    margin-top: 20px;
}

.examples-title { 
    font-weight: 700; 
    font-size: 1rem; 
    margin-bottom: 16px; 
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 8px;
}

.examples-grid {
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 12px;
}

.example-item {
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 12px;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.example-item:hover { 
    border-color: var(--accent); 
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3), 0 0 20px rgba(16, 185, 129, 0.1);
}

.example-icon { 
    font-size: 1.8rem; 
    margin-bottom: 6px;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.example-name { 
    font-size: 0.75rem; 
    color: var(--text-muted);
    font-weight: 600;
}

/* Footer */
.app-footer {
    text-align: center;
    padding: 32px;
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 32px;
    border-top: 1px solid var(--border);
}

.footer-version { 
    margin-top: 6px; 
    font-size: 0.8rem;
    opacity: 0.7;
}

/* ============ GRADIO OVERRIDES ============ */
.block { background: transparent !important; border: none !important; }

label { 
    color: var(--text-secondary) !important; 
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}

button.primary { 
    background: linear-gradient(135deg, var(--accent) 0%, var(--cyan) 100%) !important; 
    border: none !important;
    border-radius: 14px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

button.primary:hover { 
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.4) !important;
}

button.secondary { 
    background: var(--bg-glass) !important; 
    backdrop-filter: blur(10px) !important;
    border: 1px solid var(--border) !important; 
    color: var(--text-primary) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    transition: all 0.3s !important;
}

button.secondary:hover {
    border-color: var(--border-glow) !important;
    background: var(--bg-hover) !important;
}

/* Live Mode Styles */
.fps-display {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 182, 212, 0.15));
    border: 1px solid rgba(16, 185, 129, 0.2);
    padding: 12px 20px;
    border-radius: 50px;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--accent);
    backdrop-filter: blur(10px);
}

.live-result-panel {
    background: var(--bg-card);
    backdrop-filter: blur(40px);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 24px;
    box-shadow: var(--shadow-md);
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--bg-hover); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}


.result-panel, .stat-card, .top3-card {
    animation: fadeIn 0.4s ease-out;
}

.status-processing { background: rgba(59,130,246,0.15); color: #3b82f6; }

/* ============================================ */
/* LIVE BETA TAB STYLES                         */
/* ============================================ */

.live-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    min-height: 400px;
}

.live-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 20px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border);
}

.live-title {
    font-size: 1.25rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 10px;
}

.live-badge {
    background: linear-gradient(135deg, #ef4444, #f97316);
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.7rem;
    font-weight: 600;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
}

.live-status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.85rem;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    animation: blink 1s infinite;
}

.status-dot.active { background: #22c55e; }
.status-dot.paused { background: #f59e0b; animation: none; }
.status-dot.stopped { background: #6b7280; animation: none; }

@keyframes blink {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

.live-prediction {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.live-prediction.high-conf {
    border-color: var(--accent);
    background: rgba(16,185,129,0.08);
}

.live-prediction.low-conf {
    border-color: var(--warning);
    background: rgba(245,158,11,0.08);
}

.live-main-icon { font-size: 3rem; margin-bottom: 8px; }
.live-main-class { font-size: 1.5rem; font-weight: 700; text-transform: capitalize; }
.live-main-conf { font-size: 1rem; color: var(--text-muted); margin-top: 4px; }
.live-main-bin { font-size: 0.9rem; color: var(--text-secondary); margin-top: 8px; }

.live-top3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}

.live-top3-item {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    font-size: 0.85rem;
}

.live-top3-item.active {
    border-color: var(--accent);
    background: rgba(16,185,129,0.1);
}

.live-top3-icon { font-size: 1.3rem; }
.live-top3-name { font-weight: 600; margin-top: 4px; text-transform: capitalize; }
.live-top3-conf { color: var(--text-muted); font-size: 0.75rem; }

.live-tip {
    background: linear-gradient(90deg, rgba(16,185,129,0.1), rgba(59,130,246,0.1));
    border: 1px solid rgba(16,185,129,0.2);
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 16px;
}

.live-warning {
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    color: var(--warning);
    font-size: 0.9rem;
    margin-bottom: 16px;
}

.live-idle {
    text-align: center;
    padding: 60px 20px;
    color: var(--text-muted);
}

.live-idle-icon { font-size: 4rem; margin-bottom: 16px; opacity: 0.4; }
.live-idle-text { font-size: 1rem; }

.live-controls {
    display: flex;
    gap: 10px;
    margin-top: 16px;
}

.fps-display {
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 6px;
}
"""

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🌍 MatriXort AI v2.0 - Smart Waste Classification")
    print("="*60 + "\n")
    
    if not BEST_MODEL_PATH.exists():
        print("Error: Model not found!")
        return
    
    predictor = WastePredictor(BEST_MODEL_PATH)
    stats = load_stats()
    if 'items' not in stats:
        stats = {'items': 0, 'co2': 0.0, 'water': 0.0}
    history = load_history()
    
    def create_history_html(history):
        if not history:
            return ""
        
        rows = ""
        for item in reversed(history[-10:]):
            data = WASTE_DB.get(item['class'], {})
            rows += f"""
            <tr>
                <td>{item.get('timestamp', 'N/A')[:16]}</td>
                <td>{data.get('icon', '🗑️')} {item['class'].replace('-', ' ').title()}</td>
                <td>{item['confidence']*100:.1f}%</td>
                <td>{data.get('bin', 'N/A')[:30]}</td>
            </tr>
            """
        
        return f"""
        <div class="history-section">
            <div class="history-header">
                <span class="history-title">📋 Recent Classifications (Last 10)</span>
            </div>
            <table class="history-table">
                <tr><th>Time</th><th>Classification</th><th>Confidence</th><th>Disposal</th></tr>
                {rows}
            </table>
        </div>
        """
    
    def create_empty():
        tip = random.choice(TIPS)
        return f"""
        <div class="result-panel">
            <div class="empty-state">
                <div class="empty-icon">📷</div>
                <div class="empty-title">Upload a Waste Image</div>
                <div class="empty-text">Take a photo or upload an image to get detailed recycling guidance</div>
            </div>
            <div class="tip-box">
                <span>💡</span>
                <span>{tip}</span>
            </div>
        </div>
        """
    
    def create_unable_result(conf, cls, quality_warnings):
        tip = random.choice(TIPS)
        qw_html = "".join([f'<div class="quality-warning">{w}</div>' for w in quality_warnings]) if quality_warnings else ""
        
        return f"""
        <div class="result-panel">
            {qw_html}
            <div class="unable-box">
                <div class="unable-icon">🤔</div>
                <div class="unable-title">Unable to Classify</div>
                <div class="unable-text">
                    This image doesn't appear to be a typical waste item.<br><br>
                    <strong>Best guess:</strong> {cls.replace('-', ' ').title()} ({conf*100:.1f}%)<br><br>
                    <small>Try a clearer photo of common waste items.</small>
                </div>
            </div>
            <div class="tip-box"><span>💡</span><span>{tip}</span></div>
        </div>
        """
    
    def create_result(result, data, stats, quality_warnings, is_low_conf=False, uncertainty=None, location_info=None, enhancement_info=None):
        tip = random.choice(TIPS)
        cls = result['class']
        conf = result['confidence']
        top3 = result['top3']
        
        # Quality warnings
        qw_html = "".join([f'<div class="quality-warning">{w}</div>' for w in quality_warnings]) if quality_warnings else ""
        
        # Low confidence warning
        low_conf_html = ""
        if is_low_conf:
            low_conf_html = f"""
            <div class="warning-box">
                <strong>⚠️ Low Confidence ({conf*100:.1f}%)</strong><br>
                The model is uncertain. This might not be a typical waste item.
            </div>
            """
        
        # FEATURE: Enhancement info display
        enhance_html = ""
        if enhancement_info and enhancement_info.get('enhanced'):
            enhancements = ", ".join(enhancement_info.get('enhancements', []))
            enhance_html = f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border: 1px solid #4ade80; border-radius: 12px; padding: 12px; margin-bottom: 16px;">
                <div style="color: #4ade80; font-weight: 600; margin-bottom: 4px;">✨ Image Auto-Enhanced</div>
                <div style="color: #94a3b8; font-size: 0.85rem;">Applied: {enhancements}</div>
            </div>
            """
        
        # FEATURE: Uncertainty Dashboard
        uncertainty_html = ""
        if uncertainty:
            certainty = uncertainty.get('certainty_percent', 0)
            conf_icon = uncertainty.get('confidence_icon', '✅')
            conf_text = uncertainty.get('confidence_text', 'High Confidence')
            confusion = uncertainty.get('confusion_warning', '')
            
            # Color based on certainty level
            if certainty >= 80:
                bar_color = "#22c55e"  # Green
            elif certainty >= 50:
                bar_color = "#f59e0b"  # Orange
            else:
                bar_color = "#ef4444"  # Red
            
            confusion_html = ""
            if confusion:
                confusion_html = f"""
                <div style="background: #fef3c7; color: #92400e; padding: 8px 12px; border-radius: 8px; margin-top: 8px; font-size: 0.85rem;">
                    ⚠️ {confusion}
                </div>
                """
            
            uncertainty_html = f"""
            <div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-radius: 12px; padding: 16px; margin-bottom: 16px; border: 1px solid #334155;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem;">📊</span>
                    <span style="color: #f8fafc; font-weight: 600;">Confidence Analysis</span>
                </div>
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="color: #94a3b8; min-width: 70px;">Certainty:</span>
                    <div style="flex: 1; background: #334155; border-radius: 10px; height: 20px; overflow: hidden;">
                        <div style="width: {certainty}%; height: 100%; background: {bar_color}; border-radius: 10px; transition: width 0.5s;"></div>
                    </div>
                    <span style="color: {bar_color}; font-weight: 600; min-width: 45px;">{certainty:.0f}%</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.9rem;">
                    {conf_icon} {conf_text}
                </div>
                {confusion_html}
            </div>
            """
        
        # FEATURE: Recycling Location Map Button
        location_html = ""
        if location_info:
            location_html = f"""
            <div style="margin-bottom: 16px;">
                <a href="{location_info['url']}" target="_blank" style="display: block; background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; padding: 14px 20px; border-radius: 12px; text-decoration: none; text-align: center; font-weight: 600; transition: transform 0.2s, box-shadow 0.2s;">
                    {location_info['button_text']}
                </a>
            </div>
            """
        
        # Contamination warning
        contam_html = ""
        if data.get('contamination'):
            contam_items = "".join([f"<div>• {c}</div>" for c in data['contamination']])
            contam_html = f"""
            <div class="contamination-warning">
                <div class="contamination-title">⚠️ Before Recycling:</div>
                {contam_items}
            </div>
            """
        
        # Badge
        badge_map = {'Recyclable': 'badge-success', 'Hazardous': 'badge-danger', 'Compostable': 'badge-info', 'Donatable': 'badge-warning', 'Landfill': 'badge-warning'}
        badge_class = badge_map.get(data['category'], 'badge-info')
        
        # Confidence bar color
        conf_color = "#22c55e" if conf > 0.9 else "#f59e0b" if conf > 0.7 else "#ef4444"
        
        # Top 3 predictions
        top3_html = ""
        for i, (c, p) in enumerate(top3):
            d = WASTE_DB.get(c, {})
            active = "active" if i == 0 else ""
            top3_html += f"""
            <div class="top3-card {active}">
                <div class="top3-icon">{d.get('icon', '🗑️')}</div>
                <div class="top3-class">{c.replace('-', ' ')}</div>
                <div class="top3-conf">{p*100:.1f}%</div>
            </div>
            """
        
        # Prep items
        prep_html = "".join([f'<li><span class="check">✓</span>{p}</li>' for p in data['prep']])
        dont_html = "".join([f'<li><span class="cross">✗</span>{p}</li>' for p in data['dont']])
        
        return f"""
        <div class="result-panel">
            {qw_html}
            {enhance_html}
            {low_conf_html}
            
            <div class="class-header">
                <div class="class-icon">{data['icon']}</div>
                <div class="class-details">
                    <h2>{cls.replace('-', ' ')}</h2>
                    <div class="class-meta">
                        <span class="badge {badge_class}">{data['category']}</span>
                        <div class="conf-bar-container">
                            <div class="conf-bar" style="width: {conf*100}%; background: {conf_color};"></div>
                        </div>
                        <span style="color: var(--text-muted); font-size: 0.85rem;">{conf*100:.1f}%</span>
                    </div>
                </div>
            </div>
            
            {uncertainty_html}
            
            <div class="top3-section">
                <div class="section-header">🎯 Top 3 Predictions</div>
                <div class="top3-grid">{top3_html}</div>
            </div>
            
            {contam_html}
            
            <div class="bin-info">
                <div class="bin-color" style="background: {data['color']};"></div>
                <span class="bin-text">📍 {data['bin']}</span>
            </div>
            
            {location_html}
            
            <div class="impact-grid">
                <div class="impact-card">
                    <div class="impact-value">{data['co2']} kg</div>
                    <div class="impact-label">CO₂ Saved</div>
                </div>
                <div class="impact-card">
                    <div class="impact-value">{data['water']} L</div>
                    <div class="impact-label">Water Saved</div>
                </div>
                <div class="impact-card">
                    <div class="impact-value">{data['decomp']}</div>
                    <div class="impact-label">Years Decompose</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-header">📋 How to Prepare</div>
                <div class="section-content"><ul class="info-list">{prep_html}</ul></div>
            </div>
            
            <div class="section">
                <div class="section-header">⚠️ Do NOT</div>
                <div class="section-content"><ul class="info-list">{dont_html}</ul></div>
            </div>
            
            <div class="stats-bar">
                <div class="stat-card"><div class="stat-value">{stats['items']}</div><div class="stat-label">Items Sorted</div></div>
                <div class="stat-card"><div class="stat-value">{stats['co2']:.1f}</div><div class="stat-label">kg CO₂</div></div>
                <div class="stat-card"><div class="stat-value">{stats['water']:.0f}</div><div class="stat-label">L Water</div></div>
                <div class="stat-card"><div class="stat-value">🌍</div><div class="stat-label">Eco Hero</div></div>
            </div>
            
            <div class="tip-box"><span>💡</span><span>{tip}</span></div>
        </div>
        """
    
    def classify(img):
        if img is None:
            return None, None, create_empty(), create_history_html(history)
        
        # FEATURE: Auto-enhance image quality
        enhanced_img, enhancement_info = enhance_image_quality(img)
        
        # Check image quality (on original)
        quality_warnings = check_image_quality(img)
        
        # Get prediction (on enhanced image for better results)
        result = predictor.predict(enhanced_img)
        if result is None:
            return None, None, "<div class='result-panel'><p style='color:#ef4444;text-align:center;padding:40px;'>Error processing image. Try a different image.</p></div>", create_history_html(history)
        
        cls = result['class']
        conf = result['confidence']
        
        # FEATURE: Calculate uncertainty metrics
        uncertainty = calculate_uncertainty(result['all_probs'])
        
        # FEATURE: Get recycling location link
        location_info = get_recycling_location_link(cls)
        
        print(f"Prediction: {cls} ({conf*100:.1f}%) | Certainty: {uncertainty['certainty_percent']:.0f}%")
        
        # Create labels
        labels = {f"{WASTE_DB.get(c, {'icon':'🗑️'})['icon']} {c.replace('-',' ').title()}": p for c, p in result['all_probs'].items()}
        
        # Check confidence
        if conf < LOW_CONFIDENCE:
            # Return annotated image anyway
            annotated_img = draw_annotated_frame(enhanced_img, cls, conf, show_crosshair=True)
            return annotated_img, labels, create_unable_result(conf, cls, quality_warnings), create_history_html(history)
        
        # Get data and create result
        data = WASTE_DB.get(cls, WASTE_DB['trash'])
        is_low_conf = conf < HIGH_CONFIDENCE
        
        # Update stats
        if not is_low_conf:
            stats['items'] += 1
            stats['co2'] += data.get('co2', 0.5)
            stats['water'] += data.get('water', 10)
            save_stats(stats)
            
            # Add to history
            history.append({
                'class': cls,
                'confidence': conf,
                'timestamp': datetime.now().isoformat()
            })
            save_history(history)
        
        # Draw bounding box (on enhanced image)
        annotated_img = draw_annotated_frame(enhanced_img, cls, conf, show_crosshair=True)
        
        return annotated_img, labels, create_result(result, data, stats, quality_warnings, is_low_conf, uncertainty, location_info, enhancement_info), create_history_html(history)
    
    def reset():
        nonlocal stats, history
        stats = {'items': 0, 'co2': 0.0, 'water': 0.0}
        history = []
        save_stats(stats)
        save_history(history)
        return create_empty(), ""
    
    def export_csv():
        if not history:
            return None
        csv_content = export_history_csv(history)
        return gr.File.update(value=None, visible=False)
    
    # ========================================================================
    # LIVE STREAMING FUNCTIONS
    # ========================================================================
    
    live_frame_count = [0]
    
    def create_live_idle():
        return """
        <div class="live-panel">
            <div class="live-idle">
                <div class="live-idle-icon">🎥</div>
                <div class="live-idle-text">Enable webcam above to start live detection</div>
            </div>
        </div>
        """
    
    def create_live_result(result, threshold):
        if result is None:
            return create_live_idle()
        
        cls = result['class']
        conf = result['confidence']
        top3 = result['top3']
        data = WASTE_DB.get(cls, WASTE_DB['trash'])
        tip = random.choice(TIPS)
        
        # Determine confidence class
        conf_class = "high-conf" if conf >= threshold else "low-conf"
        
        # Warning for low confidence
        warning_html = ""
        if conf < threshold:
            warning_html = '<div class="live-warning">⚠️ Hold steady - Low confidence detection</div>'
        
        # Top 3 items
        top3_html = ""
        for i, (c, p) in enumerate(top3):
            d = WASTE_DB.get(c, {'icon': '🗑️'})
            active = "active" if i == 0 else ""
            top3_html += f"""
            <div class="live-top3-item {active}">
                <div class="live-top3-icon">{d.get('icon', '🗑️')}</div>
                <div class="live-top3-name">{c.replace('-', ' ')}</div>
                <div class="live-top3-conf">{p*100:.1f}%</div>
            </div>
            """
        
        return f"""
        <div class="live-panel">
            <div class="live-header">
                <div class="live-title">
                    🎯 Live Detection
                    <span class="live-badge">BETA</span>
                </div>
                <div class="live-status">
                    <span class="status-dot active"></span>
                    <span>Streaming</span>
                </div>
            </div>
            
            {warning_html}
            
            <div class="live-prediction {conf_class}">
                <div class="live-main-icon">{data['icon']}</div>
                <div class="live-main-class">{cls.replace('-', ' ')}</div>
                <div class="live-main-conf">{conf*100:.1f}% confidence</div>
                <div class="live-main-bin">📍 {data['bin']}</div>
            </div>
            
            <div class="live-top3">
                {top3_html}
            </div>
            
            <div class="live-tip">
                <span>💡</span>
                <span>{tip}</span>
            </div>
        </div>
        """
    
    # Global cache for live stream to prevent flickering and reduce load
    # using lists/dicts to allow modification in closure without 'nonlocal'
    live_frame_count = [0]
    current_prediction = {"box": None, "label": None, "color": None, "conf": 0}
    
    def process_live_frame(frame, threshold):
        """
        Process a single frame from webcam stream with MAXIMUM optimization.
        - Inference runs only every 4 frames (for smooth 60fps video)
        - Intermediary frames use cached bounding box (smooth UX)
        - Resolution limited to 640px VGA (optimal for low latency)
        """
        if frame is None:
            return frame, create_live_idle()
            
        # No 'global' keyword needed for mutating mutable objects in closure

        
        try:
            # OPTIMIZATION: VGA Resolution (640px) for LOW LATENCY
            # 960px causes lag on WebSocket. 640px is optimal for smooth streaming.
            max_width = 640
            h, w = frame.shape[:2]
            scale_factor = 1.0
            if w > max_width:
                scale_factor = max_width / w
                new_h = int(h * scale_factor)
                frame_small = cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = frame
            
            # OPTIMIZATION: Inference every 4th frame (Smoother 60fps video)
            live_frame_count[0] += 1
            should_run_inference = (live_frame_count[0] % 4 == 0)
            
            if should_run_inference:
                # Run prediction on small frame
                result = predictor.predict(frame_small)
                
                # Find bbox on small frame (always find one, even if fallback)
                sx, sy, sbw, sbh = find_object_bbox(frame_small)
                
                # Logic: Show box even if low confidence, but style it differently
                is_reliable = result['confidence'] * 100 > threshold
                
                color = BBOX_COLORS.get(result['class'], (0, 255, 0))
                # If low confidence, use Yellow/Warning color and different label
                if not is_reliable:
                     color = (0, 215, 255) # Gold/Yellow
                     label = f"? {result['class'].upper()} {result['confidence']*100:.0f}%"
                else:
                     label = f"{result['class'].upper().replace('-', ' ')} {result['confidence']*100:.0f}%"

                # Update global cache
                current_prediction["box"] = (sx, sy, sbw, sbh)
                current_prediction["label"] = label
                current_prediction["color"] = color
                current_prediction["conf"] = result['confidence']
                current_prediction["result_obj"] = result
            
            # Draw persistent overlay
            out_frame = frame_small.copy()
            
            if current_prediction["box"] is not None:
                x, y, bw, bh = current_prediction["box"]
                color = current_prediction["color"]
                
                # Draw Box
                cv2.rectangle(out_frame, (x, y), (x + bw, y + bh), color, 3)
                
                # Draw Label
                label = current_prediction["label"]
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(label, font, 0.6, 2)
                
                # Ensure label doesn't go off screen
                ly = max(20, y)
                cv2.rectangle(out_frame, (x, ly - 20), (x + text_w + 10, ly), color, -1)
                cv2.putText(out_frame, label, (x + 5, ly - 5), font, 0.6, (255, 255, 255), 2)
                
                # Center Crosshair
                cx, cy = out_frame.shape[1] // 2, out_frame.shape[0] // 2
                cv2.circle(out_frame, (cx, cy), 4, (0,0,255), -1) # Red dot center

            result_data = current_prediction.get("result_obj")
            if result_data:
                return out_frame, create_live_result(result_data, threshold / 100.0)
            else:
                return out_frame, create_live_idle()
        
        except Exception as e:
            print(f"Live frame error: {e}")
            return frame, create_live_idle()
    
    # ========================================================================
    # BUILD UI WITH TABS
    # ========================================================================
    
    with gr.Blocks() as demo:
        gr.HTML(f"<style>{CSS}</style>")
        
        gr.HTML(f"""
        <div class="app-header">
            <div class="header-content">
                <div class="header-top">
                    <span class="header-badge">🔬 AI-Powered</span>
                    <span class="status-indicator status-online">● Online</span>
                </div>
                <div class="header-main">
                    <h1 class="header-title">🌍 MatriXort AI</h1>
                    <p class="header-subtitle">Smart Waste Classification • Top-3 Predictions • Live Detection</p>
                </div>
            </div>
        </div>
        """)
        
        with gr.Tabs():
            # ============================================================
            # TAB 1: UPLOAD / SINGLE IMAGE
            # ============================================================
            with gr.Tab("📷 Upload & Analyze"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img = gr.Image(label="📷 Upload Waste Image", type="numpy", height=320, sources=["upload", "webcam", "clipboard"])
                        
                        # New output component for the result with bounding box
                        annotated_img = gr.Image(label="🎯 Analysis Result", type="numpy", height=320, interactive=False)
                        
                        with gr.Row():
                            btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                            rst = gr.Button("↻ Reset All", size="lg")
                        
                        labels = gr.Label(label="📊 All Predictions", num_top_classes=6)
                        
                        gr.HTML("""
                        <div class="examples-section">
                            <div class="examples-title">📚 What can I classify?</div>
                            <div class="examples-grid">
                                <div class="example-item"><div class="example-icon">🥤</div><div class="example-name">Plastic</div></div>
                                <div class="example-item"><div class="example-icon">🥫</div><div class="example-name">Metal</div></div>
                                <div class="example-item"><div class="example-icon">🍾</div><div class="example-name">Glass</div></div>
                                <div class="example-item"><div class="example-icon">📰</div><div class="example-name">Paper</div></div>
                                <div class="example-item"><div class="example-icon">📦</div><div class="example-name">Cardboard</div></div>
                                <div class="example-item"><div class="example-icon">🍌</div><div class="example-name">Organic</div></div>
                            </div>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        result = gr.HTML(value=create_empty())
                        history_html = gr.HTML(value=create_history_html(history))
                
                # Update inputs/outputs: Output to annotated_img instead of img
                btn.click(classify, [img], [annotated_img, labels, result, history_html])
                # USER REQUEST: Auto-Analysis on Upload
                img.change(classify, [img], [annotated_img, labels, result, history_html])
                rst.click(reset, [], [result, history_html])
            
            # ============================================================
            # TAB 2: LIVE WEBCAM STREAMING
            # ============================================================
            with gr.Tab("🎥 Live (Beta)"):
                gr.HTML("""
                <div style="background: linear-gradient(90deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1)); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;">
                    <strong style="color: #059669;">🚀 Live Detection Mode (Beta):</strong>
                    <span style="color: var(--text-secondary);">Feature is functional but under optimization. Full optimization coming in <strong style="color:#10b981;">V3 Update</strong>.</span>
                </div>
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        # WEBCAM INPUT (Must be visible to access Start/Stop controls)
                        live_webcam = gr.Image(
                            label="📷 Camera Controls (Click Start)",
                            sources=["webcam"],
                            streaming=True,
                            visible=True,  # MUST BE TRUE so user can click 'start'
                            height=320
                        )
                        
                        # OUTPUT MONITOR (This is what the user sees)
                        live_annotated = gr.Image(
                            label="👁️ AI Vision Feed",
                            type="numpy",
                            height=480,
                            interactive=False,
                            show_label=True
                        )
                        
                        # Controls
                        with gr.Row():
                            conf_slider = gr.Slider(
                                minimum=30, # Lower min so they can see everything
                                maximum=95,
                                value=60,   # Lower default
                                step=5,
                                label="🎯 High Confidence Threshold",
                            )
                        
                        gr.HTML(f"""
                        <div class="fps-display" style="margin-top: 12px;">
                            <span>⚡</span>
                            <span>Running on {GPU_NAME} • Beta Mode - Under Optimization • Full optimization in V3 Update</span>
                        </div>
                        """)
                    
                    with gr.Column(scale=1):
                        live_result = gr.HTML(value=create_live_idle())
                
                # Stream handler
                live_webcam.stream(
                    fn=process_live_frame,
                    inputs=[live_webcam, conf_slider],
                    outputs=[live_annotated, live_result],
                    time_limit=None,
                    concurrency_limit=None
                )
            
            # ============================================================
            # TAB 3: ABOUT PAGE
            # ============================================================
            with gr.Tab("ℹ️ About"):
                gr.HTML("""
                <div style="max-width: 800px; margin: 0 auto; padding: 24px;">
                    
                    <!-- Header -->
                    <div style="text-align: center; padding: 40px 24px; background: linear-gradient(145deg, rgba(15,15,25,0.9), rgba(10,10,18,0.95)); border: 1px solid rgba(255,255,255,0.06); border-radius: 24px; margin-bottom: 24px;">
                        <div style="font-size: 4rem; margin-bottom: 16px;">🌍</div>
                        <h1 style="font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #10b981, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 8px 0;">MatriXort AI</h1>
                        <p style="color: #a0a0b0; font-size: 1.1rem; margin: 0 0 16px 0;">Smart Waste Classification System</p>
                        <div style="display: inline-flex; gap: 12px; flex-wrap: wrap; justify-content: center;">
                            <span style="background: linear-gradient(135deg, #10b981, #06b6d4); color: white; padding: 8px 20px; border-radius: 50px; font-size: 0.9rem; font-weight: 600;">v2.0 Beta</span>
                            <span style="background: rgba(139,92,246,0.15); color: #a78bfa; padding: 8px 20px; border-radius: 50px; font-size: 0.9rem; font-weight: 600; border: 1px solid rgba(139,92,246,0.3);">Capstone Project</span>
                        </div>
                    </div>
                    
                    <!-- Developer Card -->
                    <div style="background: linear-gradient(145deg, rgba(15,15,25,0.9), rgba(10,10,18,0.95)); border: 1px solid rgba(255,255,255,0.06); border-radius: 24px; padding: 32px; margin-bottom: 24px;">
                        <div style="display: flex; align-items: center; gap: 24px; flex-wrap: wrap;">
                            <div style="width: 90px; height: 90px; background: linear-gradient(135deg, #10b981, #06b6d4); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 2.5rem;">👨‍💻</div>
                            <div style="flex: 1; min-width: 200px;">
                                <h3 style="color: #f8f8f8; font-size: 1.5rem; font-weight: 700; margin: 0 0 4px 0;">Vignesh</h3>
                                <p style="color: #10b981; font-size: 1rem; margin: 0 0 8px 0; font-weight: 500;">B.Tech Information Technology Student</p>
                                <p style="color: #6b7280; font-size: 0.95rem; margin: 0;">📍 SIMATS Engineering College</p>
                            </div>
                        </div>
                        <div style="display: flex; gap: 12px; margin-top: 24px; flex-wrap: wrap;">
                            <a href="https://github.com/Vixcy300" target="_blank" style="display: inline-flex; align-items: center; gap: 8px; padding: 12px 24px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; color: #f8f8f8; text-decoration: none; font-weight: 500;">🔗 GitHub</a>
                            <a href="mailto:starboynitro@gmail.com" style="display: inline-flex; align-items: center; gap: 8px; padding: 12px 24px; background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(6,182,212,0.1)); border: 1px solid rgba(16,185,129,0.3); border-radius: 12px; color: #10b981; text-decoration: none; font-weight: 500;">📧 Contact</a>
                        </div>
                    </div>
                    
                    <!-- About Project -->
                    <div style="background: linear-gradient(145deg, rgba(15,15,25,0.9), rgba(10,10,18,0.95)); border: 1px solid rgba(255,255,255,0.06); border-radius: 24px; padding: 32px; margin-bottom: 24px;">
                        <h3 style="color: #f8f8f8; font-size: 1.2rem; font-weight: 600; margin: 0 0 16px 0;">📚 About This Project</h3>
                        <p style="color: #a0a0b0; line-height: 1.8; font-size: 0.95rem; margin: 0 0 20px 0;">
                            MatriXort AI is a <strong style="color: #10b981;">Capstone Project</strong> developed during studies as part of the 
                            <strong style="color: #06b6d4;">Computer Vision</strong> course curriculum. Using deep learning and 
                            transfer learning techniques with ResNet-50, it classifies waste materials into 14 categories with real-time 
                            recycling recommendations.
                        </p>
                        <div style="display: flex; flex-wrap: wrap; gap: 10px;">
                            <span style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 8px; color: #a0a0b0; font-size: 0.85rem;">🔥 PyTorch</span>
                            <span style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 8px; color: #a0a0b0; font-size: 0.85rem;">🧠 ResNet-50</span>
                            <span style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 8px; color: #a0a0b0; font-size: 0.85rem;">🎨 Gradio</span>
                            <span style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 8px; color: #a0a0b0; font-size: 0.85rem;">🖼️ ImageNet</span>
                            <span style="background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 8px 16px; border-radius: 8px; color: #a0a0b0; font-size: 0.85rem;">📊 Transfer Learning</span>
                        </div>
                    </div>
                    
                    <!-- Disclaimer -->
                    <div style="background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(239,68,68,0.04)); border: 1px solid rgba(245,158,11,0.2); border-radius: 20px; padding: 28px; margin-bottom: 24px;">
                        <div style="display: flex; align-items: flex-start; gap: 16px;">
                            <span style="font-size: 1.5rem;">⚠️</span>
                            <div>
                                <h4 style="color: #f59e0b; font-size: 1rem; font-weight: 600; margin: 0 0 8px 0;">Disclaimer - Beta Version</h4>
                                <p style="color: #a0a0b0; line-height: 1.7; font-size: 0.9rem; margin: 0;">
                                    This model is currently in the <strong>training phase</strong> with an accuracy of approximately 
                                    <strong>88.6%</strong>. We are actively working on improvements and users can expect 
                                    <strong style="color: #10b981;">95%+ accuracy</strong> in the upcoming <strong>V3 major update</strong>.
                                    <br><br>
                                    Please verify critical waste disposal decisions with local guidelines.
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Collaboration CTA -->
                    <div style="background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,182,212,0.05)); border: 1px solid rgba(16,185,129,0.2); border-radius: 20px; padding: 32px; text-align: center;">
                        <div style="font-size: 2rem; margin-bottom: 16px;">🤝</div>
                        <h3 style="color: #f8f8f8; font-size: 1.2rem; font-weight: 600; margin: 0 0 12px 0;">Interested in Collaboration?</h3>
                        <p style="color: #6b7280; font-size: 0.95rem; margin: 0 0 20px 0;">
                            Open for development collaborations, research partnerships, and project contributions.
                        </p>
                        <a href="mailto:starboynitro@gmail.com" style="display: inline-flex; align-items: center; gap: 10px; padding: 14px 28px; background: linear-gradient(135deg, #10b981, #06b6d4); border-radius: 12px; color: white; text-decoration: none; font-weight: 600; font-size: 0.95rem;">📧 starboynitro@gmail.com</a>
                    </div>
                    
                    <!-- Footer -->
                    <div style="text-align: center; margin-top: 40px; padding-top: 24px; border-top: 1px solid rgba(255,255,255,0.06);">
                        <p style="color: #4b5563; font-size: 0.9rem; margin: 0 0 8px 0;">Made with 💚 for a sustainable future</p>
                        <p style="color: #374151; font-size: 0.8rem; margin: 0;">© 2024 MatriXort AI | SIMATS Engineering</p>
                    </div>
                    
                </div>
                """)
        
        gr.HTML(f"""
        <div class="app-footer">
            <strong>MatriXort AI</strong> - Smart Waste Classification System<br>
            <span>ResNet-50 • {len(WASTE_DB)} Categories • Live Detection • Bounding Box • Quality Check</span>
            <div class="footer-version">{MODEL_VERSION}</div>
            <div style="margin-top:8px; font-size:0.75rem; color:#6b7280;">Developed by Vignesh | B.Tech IT | SIMATS Engineering | <a href="https://github.com/Vixcy300" style="color:#10b981;">GitHub</a></div>
        </div>
        """)
    
    print(f"Starting server: http://localhost:7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
