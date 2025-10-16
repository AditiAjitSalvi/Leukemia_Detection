import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Install efficientnet if needed
# pip install efficientnet-pytorch

try:
    from efficientnet_pytorch import EfficientNet
    EFFICIENTNET_AVAILABLE = True
except ImportError:
    print("⚠️ EfficientNet not installed. Install with: pip install efficientnet-pytorch")
    EFFICIENTNET_AVAILABLE = False

# -----------------------
# Configuration
# -----------------------
DATA_DIR = r'F:\BoneMarrowSamples\Data'
BATCH_SIZE = 8
VALID_RATIO = 0.15
NUM_EPOCHS = 25
LR = 2e-4
EARLY_STOP_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_efficientnet_leukemia.pth'

# -----------------------
# EfficientNet Model
# -----------------------
class EfficientNetLeukemiaClassifier(nn.Module):
    def __init__(self, num_classes, model_name='efficientnet-b3', dropout=0.3):
        super().__init__()
        print(f"🏗️ Creating {model_name} for {num_classes} classes...")
        if EFFICIENTNET_AVAILABLE:
            self.backbone = EfficientNet.from_pretrained(model_name)
            feature_dim = self.backbone._fc.in_features
            self.backbone._fc = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(feature_dim // 2),
                nn.Dropout(dropout * 0.7),
                nn.Linear(feature_dim // 2, feature_dim // 4),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(feature_dim // 4),
                nn.Dropout(dropout * 0.5),
                nn.Linear(feature_dim // 4, num_classes)
            )
        else:
            from torchvision import models
            print("⚠️ Using ResNet50 as fallback...")
            self.backbone = models.resnet50(pretrained=True)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.BatchNorm1d(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(feature_dim // 2),
                nn.Dropout(dropout * 0.7),
                nn.Linear(feature_dim // 2, num_classes)
            )
        self.feature_dim = feature_dim
        
    def forward(self, x):
        return self.backbone(x)

# -----------------------
# Dataset Creation
# -----------------------
def create_datasets():
    print("📊 Creating datasets...")
    
    train_transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.33))
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset_full = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    val_dataset_full = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    
    total_len = len(train_dataset_full)
    val_size = int(VALID_RATIO * total_len)
    train_size = total_len - val_size
    
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(total_len, generator=generator)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_subset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    print(f"✅ Split: {len(train_subset)} train, {len(val_subset)} validation")
    return train_subset, val_subset

# -----------------------
# Training Functions
# -----------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).long()
        
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
    
    return running_loss / total, correct / total

def validate_with_metrics(model, loader, criterion, device, NUM_CLASSES):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    return running_loss / len(all_labels), accuracy, all_preds, all_labels

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, EARLY_STOP_PATIENCE, MODEL_PATH, DEVICE, CLASSES):
    print("\n🎯 Starting EfficientNet training...")
    
    best_acc = 0.0
    patience_counter = 0
    training_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCHS):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\n📅 Epoch {epoch+1}/{NUM_EPOCHS} (LR: {current_lr:.6f})")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc, val_preds, val_labels = validate_with_metrics(model, val_loader, criterion, DEVICE, len(CLASSES))
        
        scheduler.step()
        
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        print(f"🏃 Train | Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"🎯 Val   | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"⏱️ Time: {epoch_time:.1f}s")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_acc,
                'classes': CLASSES,
                'model_type': 'EfficientNet',
                'val_predictions': (val_preds, val_labels)
            }, MODEL_PATH)
            print(f"✅ Best model saved! (Acc: {best_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes!")
    print(f"🏆 Best accuracy: {best_acc:.4f}")
    
    return training_history, best_acc

# -----------------------
# Main Execution
# -----------------------
if __name__ == "__main__":
    print("🧬 EFFICIENTNET FOR LEUKEMIA DETECTION")
    print(f"📱 Device: {DEVICE}")
    print(f"🐍 PyTorch: {torch.__version__}")
    
    # Dataset info
    temp_dataset = datasets.ImageFolder(DATA_DIR)
    CLASSES = temp_dataset.classes
    NUM_CLASSES = len(CLASSES)
    print(f"📊 Classes ({NUM_CLASSES}): {CLASSES}")
    
    # Create datasets
    train_dataset, val_dataset = create_datasets()
    
    # Detect if GPU is available for pin_memory and workers
    use_cuda = torch.cuda.is_available()
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0 if not use_cuda else 4,
        pin_memory=use_cuda,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0 if not use_cuda else 4,
        pin_memory=use_cuda
    )
    
    # Model
    model = EfficientNetLeukemiaClassifier(num_classes=NUM_CLASSES, model_name='efficientnet-b3').to(DEVICE)
    
    # Loss, optimizer, scheduler
    class_counts = [378, 364, 277, 35, 29, 50, 159, 239, 289, 219, 289, 289, 389]
    if len(class_counts) == NUM_CLASSES:
        total = sum(class_counts)
        class_weights = torch.tensor([total/c for c in class_counts], dtype=torch.float32).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    
    # Train
    history, final_acc = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        NUM_EPOCHS, EARLY_STOP_PATIENCE, MODEL_PATH, DEVICE, CLASSES
    )
    
    print("\n✅ EfficientNet model training complete!")
