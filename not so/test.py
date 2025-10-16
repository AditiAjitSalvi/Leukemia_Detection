import torch
from PIL import Image
from torchvision import transforms

def load_models():
    swin = SwinTransformer()
    gnn_stage = GraphNeuralNetwork(output_dim=4)  # Number of classes

    # Load saved weights
    swin.load_state_dict(torch.load('swin_transformer_model.pth'))
    gnn_stage.load_state_dict(torch.load('gnn_stage_model.pth'))

    swin.eval()
    gnn_stage.eval()
    return swin, gnn_stage

def predict_image(image_path, swin, gnn_stage):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        graph_features = torch.randn(1, 64)  # Dummy graph features for demo
        swin_feats = swin(image)
        output = gnn_stage(graph_features, swin_feats)
        predicted_class = torch.argmax(output, 1).item()

    # Map predicted class back to label name
    class_names = ['Pro', 'Pre', 'Early', 'Benign']
    return class_names[predicted_class]

if __name__ == "__main__":
    swin_model, gnn_model = load_models()
    img_path = 'path_to_one_image.jpg'  # Replace with your test image path
    prediction = predict_image(img_path, swin_model, gnn_model)
    print(f"Predicted Leukemia Stage: {prediction}")
