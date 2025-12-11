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
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/Gradio-4.0+-ff7c00?style=flat-square&logo=gradio&logoColor=white" alt="Gradio"/>
  <img src="https://img.shields.io/badge/ResNet--50-ImageNet-10b981?style=flat-square" alt="ResNet-50"/>
  <img src="https://img.shields.io/badge/Accuracy-88.6%25-22c55e?style=flat-square" alt="Accuracy"/>
  <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta-f59e0b?style=flat-square" alt="Status"/>
  <img src="https://img.shields.io/badge/GPU-RTX_4050-76b900?style=flat-square&logo=nvidia&logoColor=white" alt="GPU"/>
</p>

---

<p align="center">
  <em>A Capstone Project for Computer Vision Course</em><br/>
  <strong>Developer:</strong> Vignesh | B.Tech IT | SIMATS Engineering
</p>

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
<em>Drag & drop or upload images</em>
</td>
<td align="center">
<strong>🎥 Live Detection</strong><br/>
<em>Real-time webcam analysis</em>
</td>
<td align="center">
<strong>ℹ️ About Page</strong><br/>
<em>Project information</em>
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

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 4GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/Vixcy300/matrixort-ai.git
cd matrixort-ai

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

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
| Paper | 📰 | Paper Bin | ✅ Yes |
| Cardboard | 📦 | Cardboard Bin | ✅ Yes |
| Glass | 🍾 | Glass Recycling | ✅ Yes |
| Organic | 🍌 | Compost Bin | 🌱 Compost |
| Battery | 🔋 | Hazardous Waste | ⚠️ Special |
| Clothes | 👕 | Donation | ♻️ Donate |
| Shoes | 👟 | Donation | ♻️ Donate |
| Trash | 🗑️ | General Waste | ❌ No |

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
