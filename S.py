import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# ----- Dataset Example -----
class LeukemiaDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels_stage = []  # For leukemia stage label (blood)
        self.labels_type = []   # For leukemia type label (bone marrow)
        
        # Dummy example: assume data_dir has 'stage' and 'type' subfolders with images
        for stage_label in ['stage1', 'stage2']:  # example stages
            stage_folder = os.path.join(data_dir, 'stage', stage_label)
            if os.path.exists(stage_folder):
                for img_name in os.listdir(stage_folder):
                    self.image_paths.append(os.path.join(stage_folder, img_name))
                    self.labels_stage.append(int(stage_label[-1]))  # simple numeric label
                    self.labels_type.append(-1)  # no type info
                
        for type_label in ['ALL', 'AML']:  # example types
            type_folder = os.path.join(data_dir, 'type', type_label)
            if os.path.exists(type_folder):
                for img_name in os.listdir(type_folder):
                    self.image_paths.append(os.path.join(type_folder, img_name))
                    self.labels_stage.append(-1)  # no stage info
                    self.labels_type.append(type_label == 'ALL')  # dummy binary class
                
        # Add transform if any
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.labels_stage[idx], self.labels_type[idx]


# ----- Simple Swin Transformer (Dummy) -----
class SwinTransformer(nn.Module):
    def __init__(self):
        super(SwinTransformer, self).__init__()
        # Simplified feature extractor
        self.conv = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        return torch.flatten(x, 1)


# ----- Simple GNN -----
class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=2):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_features, swin_features):
        # For simplicity, just combine graph_features with swin_features
        h = F.relu(self.fc1(graph_features + swin_features))
        return self.fc2(h)


# ----- Training Function -----
def train_model():
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    dataset = LeukemiaDataset('C:/Users/aditi/Downloads/archive/Segmented/Pro', transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    swin = SwinTransformer()
    gnn_stage = GraphNeuralNetwork()
    gnn_type = GraphNeuralNetwork()

    optimizer = torch.optim.Adam(list(swin.parameters()) +
                                 list(gnn_stage.parameters()) +
                                 list(gnn_type.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):  # example epoch count
        swin.train()
        gnn_stage.train()
        gnn_type.train()

        for images, labels_stage, labels_type in loader:
            # Simulate graph features as dummy tensor
            graph_features = torch.randn(images.size(0), 64)  # dummy graph feats
            swin_feats = swin(images)
            
            outputs_stage = gnn_stage(graph_features, swin_feats)
            outputs_type = gnn_type(graph_features, swin_feats)

            loss_stage = criterion(outputs_stage, labels_stage.clamp(min=0).long())
            loss_type = criterion(outputs_type, labels_type.clamp(min=0).long())
            loss = loss_stage + loss_type

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item()}")

    print("Training complete")


if __name__ == "__main__":
    train_model()
