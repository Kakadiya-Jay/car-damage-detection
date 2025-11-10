import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from torchvision.models import ResNet50_Weights

# Choose device (works on both CPU/GPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)  # adjust for your CPU

class_names = [
    "Front Breakage",
    "Front Crushed",
    "Front Normal",
    "Rear Breakage",
    "Rear Crushed",
    "Rear Normal",
]

# Use a portable path (avoid "\s" escape issues)
MODEL_PATH = Path("model") / "saved_model.pth"

# Optional: correct weights enum for recent torchvision (still accepts 'DEFAULT' in many versions)
# from torchvision.models import ResNet50_Weights


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # weights=ResNet50_Weights.DEFAULT  # if on recent torchvision
        self.model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.layer4.parameters():
            p.requires_grad = True
        self.model.fc = nn.Sequential(
            nn.Dropout(0.2), nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# --- Streamlit-friendly cache so model loads only once ---
try:
    import streamlit as st

    @st.cache_resource(show_spinner=False)
    def _load_model():
        model = CarClassifierResNet().to(DEVICE)
        # Force-load on CPU to avoid CUDA deserialization errors,
        # then move to DEVICE (CPU or GPU).
        state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

except Exception:
    # Fallback if not running inside Streamlit context (e.g., unit tests)
    _model_instance = None

    def _load_model():
        global _model_instance
        if _model_instance is None:
            model = CarClassifierResNet().to(DEVICE)
            state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
            model.load_state_dict(state, strict=True)
            model.eval()
            _model_instance = model
        return _model_instance


# Preprocessing pipeline
_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_tensor = _transform(image).unsqueeze(0).to(DEVICE)

    model = _load_model()

    with torch.no_grad():
        output = model(image_tensor)
        _, pred_idx = torch.max(output, 1)
        return class_names[pred_idx.item()]
