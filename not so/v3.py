import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


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


class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=32, output_dim=4):
        super(GraphNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, graph_features, swin_features):
        h = F.relu(self.fc1(graph_features + swin_features))
        return self.fc2(h)


def train_model():
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    data_dir = r'C:\Users\aditi\Downloads\archive\Segmented'
    dataset = LeukemiaDataset(data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    swin = SwinTransformer()
    gnn_stage = GraphNeuralNetwork(output_dim=len(dataset.stage_to_idx))

    optimizer = torch.optim.Adam(list(swin.parameters()) + list(gnn_stage.parameters()), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        swin.train()
        gnn_stage.train()

        for images, labels_stage in loader:
            graph_features = torch.randn(images.size(0), 64)  # Dummy graph features
            swin_feats = swin(images)

            outputs_stage = gnn_stage(graph_features, swin_feats)
            loss_stage = criterion(outputs_stage, labels_stage.long())

            optimizer.zero_grad()
            loss_stage.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}: Loss = {loss_stage.item()}")
 # Save models
    torch.save(swin.state_dict(), 'swin_transformer_model.pth')
    torch.save(gnn_stage.state_dict(), 'gnn_stage_model.pth')
    print("Models saved.")
    print("Training complete")


if __name__ == "__main__":
    train_model()
