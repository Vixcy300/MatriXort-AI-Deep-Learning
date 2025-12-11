# 🌍 EcoSort AI - Hugging Face Spaces Deployment Guide

## Quick Deploy (5 Minutes)

### Step 1: Create Hugging Face Account
1. Go to [huggingface.co](https://huggingface.co)
2. Click **Sign Up** → Create free account

### Step 2: Create New Space
1. Click your profile → **New Space**
2. Fill in:
   - **Space name**: `ecosort-ai`
   - **License**: MIT
   - **SDK**: Select **Gradio**
   - **Hardware**: CPU (Free)
3. Click **Create Space**

### Step 3: Upload Files
Upload these 3 files from the `hf_deploy` folder:
```
hf_deploy/
├── app.py              ← Main application
├── requirements.txt    ← Dependencies  
└── best_model.pth      ← Trained model (important!)
```

**Upload via:**
- Drag & drop files to the Space page, OR
- Use Git (see below)

### Step 4: Wait for Build
- Hugging Face will automatically install dependencies
- Build takes 2-5 minutes
- When status shows "Running" → Your app is live!

### Your Live URL
```
https://huggingface.co/spaces/YOUR_USERNAME/ecosort-ai
```

---

## Alternative: Git Upload

```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/ecosort-ai

# Copy files
cp hf_deploy/* ecosort-ai/

# Push to deploy
cd ecosort-ai
git add .
git commit -m "Deploy EcoSort AI"
git push
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Build fails | Check `requirements.txt` for typos |
| Out of memory | Model too large for free tier - contact HF |
| Slow loading | Normal for first load (model loads ~30s) |

---

## Share Your App! 🎉

Once deployed, share:
- Direct link: `https://YOUR_USERNAME-ecosort-ai.hf.space`
- Embed: Can be embedded in websites
- API: Gradio provides automatic API endpoint
