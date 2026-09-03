import torch
import torch.nn as nn  # neural network module
import timm  # PyTorch Image Models library for pre-trained models
from torch.utils.data import (
    DataLoader,
)  # load data efficiently in batches during training
from torchvision import datasets, transforms  #

# Define Transforms for Images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)  # chains many image transformeations into a single pipeline


# Build the Model Architecture
class AIArtDetector(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
        super(AIArtDetector, self).__init__()
        # Load pre-trained ViT and modify classifier head for binary output (Human=0, AI=1)
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        # Outputs a raw logit; application of sigmoid later for probability
        return self.model(x)


# Example instantiation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AIArtDetector().to(device)
print(f"Model loaded successfully on {device}")
