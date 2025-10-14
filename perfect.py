import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# ---- Custom Dataset ----
class LeukemiaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = []
        self.labels_stage = []
        self.transform = transform

        stages = ['Pro', 'Pre', 'Early', 'Benign'] # 0: Pro, 1: Pre, 2: Early, 3: Benign
        self.stage_to_idx = {stage: idx for idx, stage in enumerate(stages)}

        for stage in stages:
            stage_folder = os.path.join(data_dir, stage)
            if os.path.isdir(stage_folder):
                for img_name in os.listdir(stage_folder):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(stage_folder, img_name))
                        self.labels_stage.append(self.stage_to_idx[stage])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels_stage[idx]
        return image, label

# ---- Swin Transformer (Toy Sample, not real Swin) ----
class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        return torch.flatten(x, 1)

# ---- Graph Neural Network ----
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=4):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_features, swin_features):
        h = F.relu(self.fc1(graph_features + swin_features))
        return self.fc2(h)

# ---- Calculate Class Weights ----
def calculate_class_weights(labels, num_classes):
    counts = np.bincount(labels, minlength=num_classes)
    total = float(sum(counts))
    weights = [total/(num_classes*c) if c > 0 else 0.0 for c in counts]
    return torch.tensor(weights, dtype=torch.float)

# ---- Weighted Sampler Creation ----
def create_weighted_sampler(labels):
    class_sample_count = np.bincount(labels)
    weight = 1. / (class_sample_count + 1e-6)
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler

# ---- Model Training ----
def train_model():
    data_dir = r'C://Users//aditi//Downloads//archive//Original'
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    dataset = LeukemiaDataset(data_dir, transform=transform)
    num_classes = len(dataset.stage_to_idx)

    class_weights = calculate_class_weights(dataset.labels_stage, num_classes)
    sampler = create_weighted_sampler(dataset.labels_stage)
    loader = DataLoader(dataset, batch_size=8, sampler=sampler)

    swin = SwinTransformer()
    gnn_stage = GraphNeuralNetwork(output_dim=num_classes)
    optimizer = torch.optim.Adam(list(swin.parameters()) + list(gnn_stage.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    for epoch in range(30):  # more epochs for better convergence
        swin.train()
        gnn_stage.train()
        running_loss = 0
        correct = 0
        total = 0

        for images, labels_stage in loader:
            # REPLACE this with realistic graph features for highest accuracy!
            graph_features = torch.randn(images.size(0), 64)

            swin_feats = swin(images)
            outputs_stage = gnn_stage(graph_features, swin_feats)
            loss_stage = criterion(outputs_stage, labels_stage.long())

            optimizer.zero_grad()
            loss_stage.backward()
            optimizer.step()

            running_loss += loss_stage.item() * images.size(0)
            preds = torch.argmax(outputs_stage, 1)
            correct += (preds == labels_stage).sum().item()
            total += images.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch + 1} -- Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    torch.save(swin.state_dict(), 'swin_transformer_model.pth')
    torch.save(gnn_stage.state_dict(), 'gnn_stage_model.pth')
    print("Models saved.")

# ---- Model Loading ----
def load_models():
    swin = SwinTransformer()
    gnn_stage = GraphNeuralNetwork(output_dim=4)
    swin.load_state_dict(torch.load('swin_transformer_model.pth'))
    gnn_stage.load_state_dict(torch.load('gnn_stage_model.pth'))
    swin.eval()
    gnn_stage.eval()
    return swin, gnn_stage

# ---- Prediction Function ----
def predict_image(image_path, swin, gnn_stage):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        # REPLACE this with proper graph-based features for actual performance!
        graph_features = torch.randn(1, 64)
        swin_feats = swin(image)
        output = gnn_stage(graph_features, swin_feats)
        predicted_class = torch.argmax(output, dim=1).item()
    idx_to_stage = {0: 'Pro', 1: 'Pre', 2: 'Early', 3: 'Benign'}
    return idx_to_stage[predicted_class]

# ---- Script Entrypoint ----
if __name__ == "__main__":
    train_model()
    swin_model, gnn_model = load_models()
#"C:Users\aditi\Downloads\archive\traning\WBC-Malignant-Pro-793.jpg" C:\Users\aditi\Downloads\archive\traning\WBC-Benign-490.jpg
    test_image_path = r"C:\Users\aditi\Downloads\archive\traning\WBC-Malignant-Early-971.jpg"
    prediction = predict_image(test_image_path, swin_model, gnn_model)
    print(f"Predicted Leukemia Stage: {prediction}")























"""import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from torch_geometric.data import Data
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

# Custom Dataset for Kaggle Leukemia Data
class LeukemiaDataset(Dataset):
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

# Load and preprocess Kaggle dataset
def load_kaggle_data(data_dir='C:/Users/aditi/Downloads/archive/Original'):
    image_paths = []
    labels = []
    class_map = {
        'Original/Benign': 0,  # Healthy
        'Original/Early': 1,  # Early Stage
        'Original/Pre': 2,  # Intermediate Stage
        'Original/Pro': 3  # Advanced Stage
    }
    
    for class_name in class_map:
        class_dir = os.path.join(data_dir, class_name.replace('/', os.sep))
        if not os.path.exists(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(class_map[class_name])
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return train_paths, train_labels, val_paths, val_labels

# Swin + GNN Model
class SwinGNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SwinGNN, self).__init__()
        # Load pre-trained Swin Transformer (tiny)
        self.swin = models.swin_t(weights='IMAGENET1K_V1')
        self.swin.head = nn.Identity()  # Remove classification head
        
        # GNN layers
        self.gnn1 = GCNConv(768, 256)  # Swin output dim = 768
        self.gnn2 = GCNConv(256, 128)
        self.fc = nn.Linear(128, num_classes)
        
        # Grid graph for 7x7 patches
        self.edge_index, _ = grid(7, 7)

    def forward(self, x):
        # Extract features with Swin
        features = self.swin(x)  # B, 768
        features = features.view(features.size(0), 768, 7, 7)
        features = features.view(features.size(0), 768, -1).permute(0, 2, 1)  # B, 49, 768
        
        # GNN processing
        batch_size = features.size(0)
        outputs = []
        for i in range(batch_size):
            graph_data = Data(x=features[i], edge_index=self.edge_index.to(features.device))
            g = self.gnn1(graph_data.x, graph_data.edge_index)
            g = torch.relu(g)
            g = self.gnn2(g, graph_data.edge_index)
            g = torch.relu(g)
            g = g.mean(dim=0)  # Global mean pooling
            outputs.append(g)
        outputs = torch.stack(outputs)
        
        return self.fc(outputs)

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Training function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
        
        train_acc = train_correct / len(train_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Val Acc: {val_acc:.4f}')

# Main execution
if __name__ == '__main__':
    # Load data
    train_paths, train_labels, val_paths, val_labels = load_kaggle_data(data_dir='leukemia/Original')
    
    # Create datasets and loaders
    train_dataset = LeukemiaDataset(train_paths, train_labels, transform=transform)
    val_dataset = LeukemiaDataset(val_paths, val_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialize and train model
    model = SwinGNN(num_classes=4)
    train_model(model, train_loader, val_loader, epochs=10)
    
    # Save model
    torch.save(model.state_dict(), 'leukemia_swin_gnn.pth')

# Inference function
def predict_stage(model, image_path, device='cuda'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    stages = ['Healthy', 'Early Stage', 'Intermediate Stage', 'Advanced Stage']
    return stages[pred]
"""
