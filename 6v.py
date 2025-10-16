import os
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
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# ----- Dataset Class -----
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

# ----- Load Dataset -----
def load_kaggle_data(data_dir='C:/Users/aditi/Downloads/archive/'):
    image_paths = []
    labels = []
    class_map = {
        'Original/Benign': 0,
        'Original/Early': 1,
        'Original/Pre': 2,
        'Original/Pro': 3
    }

    for class_name in class_map:
        class_dir = os.path.join(data_dir, class_name.replace('/', os.sep))
        if not os.path.exists(class_dir):
            print(f"⚠️ Warning: Folder not found: {class_dir}")
            continue

        images_in_class = 0
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_name))
                labels.append(class_map[class_name])
                images_in_class += 1
        print(f"✅ Found {images_in_class} images in '{class_name}'")

    if len(image_paths) == 0:
        raise ValueError("🚫 No images found! Check your dataset path and structure.")

    print(f"📦 Total images found: {len(image_paths)}")

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, stratify=labels, random_state=42
    )
    return train_paths, train_labels, val_paths, val_labels

# ----- Model with Feature Extractor + GNN -----
class SwinGNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SwinGNN, self).__init__()

        swin = models.swin_t(weights='IMAGENET1K_V1')

        # Print available nodes so you can choose a valid one
        train_nodes, eval_nodes = get_graph_node_names(swin)
        print(">>> Available nodes in Swin model (train):", train_nodes)
        print(">>> Available nodes in Swin model (eval):", eval_nodes)

        # ** You need to pick a valid node from the printed list **
        # For example, suppose you saw "layers.2" or "features.norm1" etc.
        # Here is a placeholder. Replace 'your_node_here' with the real name.
        return_nodes = {'your_node_here': 'feat_map'}

        self.backbone = create_feature_extractor(swin, return_nodes=return_nodes)

        # We don't know C, H, W until runtime, but often patch features are 768 × 7 × 7, etc.
        self.gnn1 = GCNConv(768, 256)
        self.gnn2 = GCNConv(256, 128)
        self.fc = nn.Linear(128, num_classes)

        # If your feature map is H×W = 7x7, grid(7,7). But if it's different, adjust accordingly.
        self.edge_index, _ = grid(7, 7)

    def forward(self, x):
        out = self.backbone(x)
        # 'feat_map' is the key you used in return_nodes
        features = out['feat_map']
        # features shape = [B, C, H, W]
        B, C, H, W = features.shape

        # Flatten patch dimension: [B, H*W, C]
        features = features.reshape(B, C, H * W).permute(0, 2, 1)

        outputs = []
        for i in range(B):
            graph_data = Data(x=features[i], edge_index=self.edge_index.to(features.device))
            g = self.gnn1(graph_data.x, graph_data.edge_index)
            g = torch.relu(g)
            g = self.gnn2(g, graph_data.edge_index)
            g = torch.relu(g)
            g = g.mean(dim=0)
            outputs.append(g)

        stacked = torch.stack(outputs)  # shape [B, 128]
        return self.fc(stacked)

# ----- Transforms -----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

# ----- Training Function -----
def train_model(model, train_loader, val_loader, device, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(outputs, 1) == labels).sum().item()

        train_acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}, Accuracy = {train_acc*100:.2f}%")

    torch.save(model.state_dict(), "leukemia_swin_gnn.pth")
    print("✅ Model trained and saved as 'leukemia_swin_gnn.pth'")

# ----- Prediction Function -----
def predict_stage(model, image_path, device='cpu'):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()
    stages = ['Healthy', 'Early Stage', 'Intermediate Stage', 'Advanced Stage']
    return stages[pred]

# ----- Main Switch -----
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Select Option:")
    print("1: Train the model")
    print("2: Predict from an image")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == '1':
        try:
            train_paths, train_labels, val_paths, val_labels = load_kaggle_data()
        except ValueError as e:
            print(e)
            return

        train_dataset = LeukemiaDataset(train_paths, train_labels, transform=transform)
        val_dataset = LeukemiaDataset(val_paths, val_labels, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        model = SwinGNN(num_classes=4)
        train_model(model, train_loader, val_loader, device=device)

    elif choice == '2':
        model = SwinGNN(num_classes=4)
        if not os.path.exists("leukemia_swin_gnn.pth"):
            print("❌ Model file not found. Train first.")
            return
        model.load_state_dict(torch.load("leukemia_swin_gnn.pth", map_location=device))
        model.to(device)

        image_path = input("Enter the full path to image for prediction: ").strip('"')
        if not os.path.isfile(image_path):
            print("❌ Invalid image path")
            return

        stage = predict_stage(model, image_path, device=device)
        print(f"🔍 Predicted Leukemia Stage: {stage}")

    else:
        print("❌ Invalid choice. Use 1 or 2.")

if __name__ == "__main__":
    main()
