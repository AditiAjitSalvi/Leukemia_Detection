import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_geometric.data import Data
from PIL import Image
from sklearn.model_selection import train_test_split

# 🔍 Class labels (13 leukemia types)
LEUKEMIA_CLASSES = sorted([
    'BAS', 'BLA', 'EOS', 'FGC', 'KSC', 'LYI', 'LYT',
    'MMZ', 'MYB', 'NGB', 'NGS', 'PEB', 'PMO'
])
CLASS_TO_IDX = {label: idx for idx, label in enumerate(LEUKEMIA_CLASSES)}

# 📁 Custom Dataset
class LeukemiaTypeDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# 📂 Load dataset from folder structure
def load_leukemia_data(data_dir):
    image_paths = []
    labels = []
    for label in LEUKEMIA_CLASSES:
        class_dir = os.path.join(data_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(CLASS_TO_IDX[label])
    return train_test_split(image_paths, labels, test_size=0.2, stratify=labels, random_state=42)

# 🧠 Swin + GNN Model
class SwinGNN(nn.Module):
    def __init__(self, num_classes=13):
        super(SwinGNN, self).__init__()
        self.swin = models.swin_t(weights='IMAGENET1K_V1')
        self.swin.head = nn.Identity()
        self.gnn1 = GCNConv(768, 256)
        self.gnn2 = GCNConv(256, 128)
        self.fc = nn.Linear(128, num_classes)
        self.edge_index, _ = grid(7, 7)  # 7x7 patches from Swin

    def forward(self, x):
        features = self.swin(x)  # [B, 768]
        features = features.view(features.size(0), 768, 7, 7)
        features = features.view(features.size(0), 768, -1).permute(0, 2, 1)  # [B, 49, 768]

        outputs = []
        for i in range(features.size(0)):
            graph_data = Data(x=features[i], edge_index=self.edge_index.to(x.device))
            g = torch.relu(self.gnn1(graph_data.x, graph_data.edge_index))
            g = torch.relu(self.gnn2(g, graph_data.edge_index))
            g = g.mean(dim=0)  # Global mean pooling
            outputs.append(g)
        return self.fc(torch.stack(outputs))

# 🧪 Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 🏋️ Training Function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_correct += (outputs.argmax(1) == labels).sum().item()

        print(f"Epoch {epoch+1}/{epochs} | "f"Train Loss: {train_loss/len(train_loader):.4f}, "      f"Train Acc: {train_correct/len(train_loader.dataset):.4f}, "   f"Val Loss: {val_loss/len(val_loader):.4f}, "              f"Val Acc: {val_correct/len(val_loader.dataset):.4f}")

# 🔍 Inference
def predict_type(model, image_path, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    return LEUKEMIA_CLASSES[pred]

# 🚀 Main Execution
if __name__ == '__main__':
    data_dir = r'F:\BoneMarrowSamples\Data'
    train_paths, train_labels, val_paths, val_labels = load_leukemia_data(data_dir)

    train_dataset = LeukemiaTypeDataset(train_paths, train_labels, transform=transform)
    val_dataset = LeukemiaTypeDataset(val_paths, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = SwinGNN(num_classes=len(LEUKEMIA_CLASSES))
    train_model(model, train_loader, val_loader, epochs=10)

    torch.save(model.state_dict(), 'leukemia_type_swin_gnn.pth')
    print("Model saved.")

    # 🔍 Predict on a test image
    test_image = r'F:\BoneMarrowSamples\Training\BAS_00202.jpg'
    prediction = predict_type(model, test_image)
    print(f"Predicted Leukemia Type: {prediction}")
