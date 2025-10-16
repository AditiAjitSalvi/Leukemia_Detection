"""import os, torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from timm import create_model
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt

# Dataset
DATA_DIR = r'F:\BoneMarrowSamples\Data'
CLASSES = sorted(os.listdir(DATA_DIR))

# Transforms
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# Model
class SwinClassifier(nn.Module):
    def __init__(self, embed_dim=768, num_classes=13):
        super().__init__()
        self.swin = create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.swin.head = nn.Identity()
        self.fc = nn.Linear(self.swin.num_features, num_classes)
    
    def forward(self, x):
        features = self.swin(x)
        return self.fc(features)

# Train function
def train(model, loader, optimizer, criterion, device):
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Predict
def predict(model, image_path, device):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = CLASSES[output.argmax().item()]
        print(f"Predicted Leukemia Type: {pred}")
        plt.imshow(img)
        plt.title(f"Prediction: {pred}")
        plt.axis('off')
        plt.show()

# Main
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = SwinClassifier(num_classes=len(CLASSES)).to(device)
    class_counts = [378, 364, 277, 35, 29, 50, 159, 239, 289, 219, 289, 289, 389]
    total = sum(class_counts)
    class_weights = torch.tensor([total / c for c in class_counts], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(5):
        train(model, loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1} complete")

    # Predict example
    predict(model, r"F:\BoneMarrowSamples\Training\BAS_00202.jpg", device)
"""
import torch
print(torch.cuda.is_available())
#pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
