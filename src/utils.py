import torch
from torchvision import transforms

def get_transforms():
    """Returns standard transforms for training/inference."""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def get_device():
    """Returns available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
