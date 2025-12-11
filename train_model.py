# -*- coding: utf-8 -*-
# Standalone training script

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Tuple
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# Training params - IMPROVED for better accuracy
BATCH_SIZE = 16  # Smaller batch = more gradient updates
NUM_WORKERS = 4
LEARNING_RATE = 0.0005  # Lower LR for more stable training
WEIGHT_DECAY = 1e-4
EPOCHS = 30  # More epochs for better convergence
LR_SCHEDULER_PATIENCE = 4  # Patience before reducing LR
LR_SCHEDULER_FACTOR = 0.5  # Gentler LR reduction
EARLY_STOPPING_PATIENCE = 8  # More patience to find best model
FREEZE_BACKBONE_EPOCHS = 3  # Less frozen = faster fine-tuning
FINE_TUNE_LR = 5e-5  # Lower fine-tune LR for stability
VAL_SPLIT = 0.15  # More training data
RANDOM_SEED = 42
LABEL_SMOOTHING = 0.1  # Prevents overconfidence

# Image settings
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Get device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

# ============================================================================
# DATASET
# ============================================================================

from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image


class WasteDataset(Dataset):
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        if class_to_idx is None:
            self.classes = sorted([d.name for d in self.root_dir.iterdir() 
                                   if d.is_dir() and not d.name.startswith('.')])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
            self.classes = sorted(class_to_idx.keys(), key=lambda x: class_to_idx[x])
        
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.samples.append((img_path, class_idx))
        
        print(f"Loaded {len(self.samples)} images from {len(self.classes)} classes")
        for cls, idx in self.class_to_idx.items():
            count = sum(1 for _, label in self.samples if label == idx)
            print(f"  - {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transforms():
    """Enhanced augmentation for better generalization"""
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),  # More aggressive crop
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),  # Added vertical flip
        transforms.RandomRotation(degrees=20),  # More rotation
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Added affine
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Added perspective
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.15),  # Added random erasing
    ])


def get_val_transforms():
    return transforms.Compose([
        transforms.Resize(int(IMAGE_SIZE * 1.14)),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def create_dataloaders(train_dir, batch_size=BATCH_SIZE):
    full_train_dataset = WasteDataset(root_dir=train_dir, transform=get_train_transforms())
    class_to_idx = full_train_dataset.class_to_idx
    
    torch.manual_seed(RANDOM_SEED)
    total_size = len(full_train_dataset)
    val_size = int(total_size * VAL_SPLIT)
    train_size = total_size - val_size
    
    train_dataset, val_indices_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    val_dataset = WasteDataset(root_dir=train_dir, transform=get_val_transforms(), class_to_idx=class_to_idx)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices_dataset.indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    
    print(f"Training samples: {len(train_dataset)}, Validation: {len(val_dataset)}")
    return train_loader, val_loader, class_to_idx


# ============================================================================
# MODEL
# ============================================================================

from torchvision import models


class WasteClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = models.resnet50(weights=weights)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.25),
            nn.Linear(256, num_classes)
        )
        
        if freeze_backbone:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for fine-tuning")
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================================
# TRAINER
# ============================================================================

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_accuracy):
        if self.best_score is None:
            self.best_score = val_accuracy
        elif val_accuracy < self.best_score + 0.001:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_accuracy
            self.counter = 0
        return self.early_stop


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100.*correct/total:.1f}%"})
    
    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    
    for images, labels in tqdm(val_loader, desc="Validation", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return running_loss / total, correct / total


def train(train_loader, val_loader, class_to_idx, epochs=EPOCHS):
    model = WasteClassifier(num_classes=len(class_to_idx), pretrained=True, freeze_backbone=True)
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)  # Prevents overconfidence
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR)
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    best_epoch = 0
    
    print(f"\n{'='*60}\nStarting Training\n{'='*60}")
    print(f"Device: {DEVICE}, Epochs: {epochs}, Classes: {len(class_to_idx)}\n")
    
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 40)
        
        if epoch == FREEZE_BACKBONE_EPOCHS + 1:
            print("\n>>> Unfreezing backbone <<<")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(model.parameters(), lr=FINE_TUNE_LR, weight_decay=WEIGHT_DECAY)
            scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=LR_SCHEDULER_PATIENCE, factor=LR_SCHEDULER_FACTOR)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")
        
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'class_to_idx': class_to_idx,
            }, BEST_MODEL_PATH)
            print(f"* New best model! Accuracy: {val_acc*100:.2f}%")
        
        if early_stopping(val_acc):
            print(f"\nEarly stopping at epoch {epoch}")
            break
        print()
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best: {best_val_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"Model saved to: {BEST_MODEL_PATH}")
    print(f"{'='*60}")
    
    # Save history
    with open(MODEL_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train Waste Classification Model')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    
    torch.manual_seed(RANDOM_SEED)
    
    print("Loading datasets...")
    train_loader, val_loader, class_to_idx = create_dataloaders(TRAIN_DIR, args.batch_size)
    
    train(train_loader, val_loader, class_to_idx, args.epochs)


if __name__ == '__main__':
    main()
