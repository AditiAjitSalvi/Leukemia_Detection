import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Import PyTorch Geometric modules
from torch_geometric.data import Data as GeoData
from torch_geometric.nn import GCNConv
from torchvision import transforms

class LeukemiaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_paths = []
        self.labels_stage = []
        self.transform = transform

        stages = ['Pro', 'Pre', 'Early', 'Benign']  # Your stage folders
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

class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        return torch.flatten(x, 1)

class GNN(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=4):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def build_graph(features):
    """
    Build a simple fully connected graph (for demonstration)
    features: [num_nodes, feature_dim]
    Returns edge_index tensor with shape [2, num_edges]
    """
    num_nodes = features.size(0)
    # Connect every node to every other node (complete graph)
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i)
                col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    return edge_index

def train_model():
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    data_dir = r'C://Users//aditi//Downloads//archive//Original'
    dataset = LeukemiaDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    swin = SwinTransformer()
    gnn = GNN(output_dim=len(dataset.stage_to_idx))

    optimizer = torch.optim.Adam(list(swin.parameters()) + list(gnn.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        swin.train()
        gnn.train()
        total_loss = 0

        for images, labels in loader:
            # Extract features for batch images
            swin_feats = swin(images)  # shape: [batch_size, feature_dim]

            # Build graph edges for this batch (simple fully connected graph)
            edge_index = build_graph(swin_feats)

            # Forward pass through GNN
            outputs = gnn(swin_feats, edge_index)

            loss = criterion(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} - Loss: {total_loss / len(loader):.4f}")

    # Save models
    torch.save(swin.state_dict(), 'swin_transformer_model.pth')
    torch.save(gnn.state_dict(), 'gnn_model.pth')
    print("Models saved.")

def load_models():
    swin = SwinTransformer()
    gnn = GNN(output_dim=4)

    swin.load_state_dict(torch.load('swin_transformer_model.pth'))
    gnn.load_state_dict(torch.load('gnn_model.pth'))

    swin.eval()
    gnn.eval()
    return swin, gnn

def predict_image(image_path, swin, gnn):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # batch size 1

    with torch.no_grad():
        swin_feats = swin(image)  # [1, feature_dim]
        # For prediction, we need at least one node's edge_index.
        # A single node has no edges; you can create a self-loop to avoid errors:
        edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        output = gnn(swin_feats, edge_index)
        predicted_class = torch.argmax(output, dim=1).item()

    idx_to_stage = {0: 'Pro', 1: 'Pre', 2: 'Early', 3: 'Benign'}
    return idx_to_stage[predicted_class]

if __name__ == "__main__":
    train_model()
    swin_model, gnn_model = load_models()
    test_image_path = r"C:\Users\aditi\Downloads\archive\traning\WBC-Malignant-Pro-792.jpg"
    prediction = predict_image(test_image_path, swin_model, gnn_model)
    print(f"Predicted Leukemia Stage: {prediction}")
