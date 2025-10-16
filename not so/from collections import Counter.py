from collections import Counter
from V4 import LeukemiaDataset# Replace with actual filename without .py extension
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

data_dir = r'C://Users//aditi//Downloads//archive//Original'
dataset = LeukemiaDataset(data_dir, transform=transform)

print("Dataset size:", len(dataset))
print("Class distribution:", Counter(dataset.labels_stage))
