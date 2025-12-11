<p align="center">
  <img src="https://img.shields.io/badge/🌍-MatriXort_AI-10b981?style=for-the-badge&labelColor=0a0a0f" alt="MatriXort AI"/>
</p>

<h1 align="center">
  🌍 MatriXort AI
</h1>

<p align="center">
  <strong>Smart Waste Classification System powered by Deep Learning</strong>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/vignesh2007/MatriXort-AI">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Hugging_Face-ff9d00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Live Demo"/>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Gradio-4.0+-ff7c00?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/ResNet--50-ImageNet-10b981?style=flat-square" alt="ResNet-50"/>
  <img src="https://img.shields.io/badge/Accuracy-88.6%25-22c55e?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta_v2.0-f59e0b?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/Categories-14-10b981?style=flat-square" alt="Categories"/>
</p>

---

## 🚀 Live Demo

<p align="center">
  <a href="https://huggingface.co/spaces/vignesh2007/MatriXort-AI">
    <img src="https://img.shields.io/badge/Try_MatriXort_AI_Now!-Click_Here-10b981?style=for-the-badge" alt="Try Now"/>
  </a>
</p>

**🌐 Demo Link:** [https://huggingface.co/spaces/vignesh2007/MatriXort-AI](https://huggingface.co/spaces/vignesh2007/MatriXort-AI)

> Upload any waste image and get instant AI-powered classification with recycling recommendations!

---

## 📝 Project Description

**MatriXort AI** is an intelligent waste classification system that uses deep learning to automatically identify and categorize waste materials. Built as a **Capstone Project** for the **Computer Vision** course at SIMATS Engineering College, this application aims to promote sustainable waste management practices.

### 🎯 What It Does

- **Classifies Waste Images** into 14 different categories (Plastic, Metal, Glass, Paper, Cardboard, Organic, Battery, Clothes, Shoes, Trash, Biological, Brown Glass, Green Glass, White Glass)
- **Provides Recycling Guidance** with specific bin recommendations and preparation instructions
- **Calculates Environmental Impact** showing CO₂ saved, water saved, and decomposition time
- **Real-time Detection** with bounding box visualization and confidence scores
- **Tracks Classification History** for session-based analytics

### 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **AI-Powered** | ResNet-50 deep learning model with 88.6% accuracy |
| 🎯 **Top-3 Predictions** | Shows alternative classifications with confidence % |
| 📍 **Find Recycling Centers** | Google Maps integration to locate nearby facilities |
| 📊 **Environmental Impact** | CO₂ & water savings calculations |
| ✅ **Preparation Tips** | Step-by-step recycling preparation guide |
| ⚠️ **Do NOT Guide** | Warnings for common recycling mistakes |
| 📹 **Live Detection** | Real-time webcam classification (local only) |
| 🔲 **Bounding Box** | Visual object detection overlay |

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Core Features
- **14 Waste Categories** - Plastic, Metal, Glass, Paper, Organic & more
- **Real-time Classification** - Instant AI-powered predictions
- **Top-3 Predictions** - See alternative classifications
- **Confidence Scoring** - Know how sure the AI is

</td>
<td width="50%">

### 🚀 Advanced Features
- **Live Webcam Detection** - Real-time video analysis
- **Bounding Box Overlay** - Visual object detection
- **Image Quality Check** - Warnings for blur/dark images
- **Environmental Impact** - CO₂ & water savings display

</td>
</tr>
</table>

---

## 🖼️ Screenshots

<table>
<tr>
<td align="center">
<strong>📷 Upload & Analyze</strong><br/>
<em>Drag & drop or upload images for instant classification</em>
</td>
<td align="center">
<strong>🎯 Results Panel</strong><br/>
<em>Detailed recycling guidance with impact stats</em>
</td>
<td align="center">
<strong>🎥 Live Detection</strong><br/>
<em>Real-time webcam analysis (local only)</em>
</td>
</tr>
</table>

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="20%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" width="40" height="40"/><br/>
<strong>Python</strong>
</td>
<td align="center" width="20%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/pytorch/pytorch-original.svg" width="40" height="40"/><br/>
<strong>PyTorch</strong>
</td>
<td align="center" width="20%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" width="40" height="40"/><br/>
<strong>OpenCV</strong>
</td>
<td align="center" width="20%">
<img src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" width="40" height="40"/><br/>
<strong>NumPy</strong>
</td>
<td align="center" width="20%">
<img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" width="40" height="40"/><br/>
<strong>Gradio</strong>
</td>
</tr>
</table>

---

## 📊 Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     MatriXort AI Model                       │
├─────────────────────────────────────────────────────────────┤
│  Input Image (224x224)                                       │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            ResNet-50 (Pre-trained ImageNet)          │    │
│  │            Feature Extraction: 2048 features         │    │
│  └─────────────────────────────────────────────────────┘    │
│        ↓                                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Custom Classifier Head                  │    │
│  │   Linear(2048→512) → ReLU → BatchNorm → Dropout     │    │
│  │   Linear(512→256)  → ReLU → BatchNorm → Dropout     │    │
│  │   Linear(256→14)   → Output                          │    │
│  └─────────────────────────────────────────────────────┘    │
│        ↓                                                     │
│  14 Waste Categories                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Try Online (No Installation!)

👉 **[Open MatriXort AI Demo](https://huggingface.co/spaces/vignesh2007/MatriXort-AI)**

### Run Locally

#### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

#### Installation

```bash
# Clone the repository
git clone https://github.com/Vixcy300/MatriXort-AI-Deep-Learning.git
cd MatriXort-AI-Deep-Learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Run the Application

```bash
python run_app.py
```

Then open **http://localhost:7860** in your browser.

---

## 📁 Project Structure

```
matrixort-ai/
├── 📂 data/
│   └── 📂 waste_classification/    # Training dataset
├── 📂 models/
│   ├── 🧠 best_model.pth           # Trained model weights
│   └── 📊 training_history.json    # Training metrics
├── 📂 hf_deploy/                   # Hugging Face deployment files
│   ├── app.py
│   ├── requirements.txt
│   └── README.md
├── 🐍 run_app.py                   # Main application
├── 🐍 train_model.py               # Model training script
├── 📋 requirements.txt             # Dependencies
└── 📖 README.md                    # This file
```

---

## 🎯 Waste Categories

| Category | Icon | Disposal | Recyclable |
|----------|------|----------|------------|
| Plastic | 🥤 | Blue/Yellow Bin | ✅ Yes |
| Metal | 🥫 | Metal Recycling | ✅ Yes |
| Paper | 📄 | Paper Bin | ✅ Yes |
| Cardboard | 📦 | Cardboard Bin | ✅ Yes |
| Glass | 🍾 | Glass Recycling | ✅ Yes |
| Organic | 🍎 | Compost Bin | 🌱 Compost |
| Battery | 🔋 | Hazardous Waste | ⚠️ Special |
| Clothes | 👕 | Donation | ♻️ Donate |
| Shoes | 👟 | Donation | ♻️ Donate |
| Trash | 🗑️ | General Waste | ❌ No |
| Biological | 🌿 | Organic Bin | 🌱 Compost |
| Brown Glass | 🍺 | Glass (Brown) | ✅ Yes |
| Green Glass | 🍷 | Glass (Green) | ✅ Yes |
| White Glass | 🥛 | Glass (Clear) | ✅ Yes |

---

## 📈 Training Results

| Metric | Value |
|--------|-------|
| **Best Validation Accuracy** | 88.59% |
| **Training Epochs** | 13 (Early Stopping) |
| **Learning Rate** | 0.0003 |
| **Batch Size** | 32 |
| **Optimizer** | AdamW |
| **Label Smoothing** | 0.1 |

---

## ⚠️ Disclaimer

> **Beta Version Notice**
> 
> This model is currently in the training phase with an accuracy of approximately **88.6%**. 
> We are actively working on improvements and users can expect **95%+ accuracy** in the upcoming **V3 major update**.
> 
> Please verify critical waste disposal decisions with local guidelines.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📬 Contact

<p align="center">
  <strong>Vignesh</strong><br/>
  B.Tech Information Technology<br/>
  SIMATS Engineering College
</p>

<p align="center">
  <a href="https://github.com/Vixcy300">
    <img src="https://img.shields.io/badge/GitHub-Vixcy300-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
  &nbsp;
  <a href="mailto:starboynitro@gmail.com">
    <img src="https://img.shields.io/badge/Email-starboynitro@gmail.com-ea4335?style=for-the-badge&logo=gmail&logoColor=white" alt="Email"/>
  </a>
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/vignesh2007/MatriXort-AI">
    <img src="https://img.shields.io/badge/🚀_Try_Demo-Hugging_Face-ff9d00?style=for-the-badge" alt="Demo"/>
  </a>
</p>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <strong>Made with 💚 for a sustainable future</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/🌍-Save_The_Planet-10b981?style=for-the-badge&labelColor=0a0a0f" alt="Save The Planet"/>
</p>

---

<p align="center">
  <sub>© 2024 MatriXort AI | SIMATS Engineering | Computer Vision Capstone Project</sub>
</p>
