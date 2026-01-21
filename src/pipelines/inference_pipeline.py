import torch
from torchvision import models
from PIL import Image
import io
from src.utils import get_transforms, get_device

def load_prediction_model(checkpoint_path):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint['classes']

    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    
    return model, classes

def predict_image(model, classes, image_bytes):
    device = get_device()
    transform = get_transforms()
    
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        _, idx = torch.max(outputs, 1)
        probs = torch.softmax(outputs, 1)
        
    return classes[idx.item()], probs[0][idx].item()
