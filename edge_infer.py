import torch
import torchvision.transforms as T
from PIL import Image
import json

# -------- Settings for Spyder ----------
MODEL_PATH = "outputs/model_ts.pt"
CLASSES_PATH = "outputs/classes.json"
IMAGE_PATH = "test.jpg"
# ---------------------------------------

# Load model
model = torch.jit.load(MODEL_PATH)
model.eval()

# Load classes
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# Image transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Load image
img = Image.open(IMAGE_PATH).convert("RGB")
tensor = transform(img).unsqueeze(0)   # add batch dimension

# Predict
with torch.no_grad():
    outputs = model(tensor)
    pred = outputs.argmax(1).item()

print("Predicted:", classes[pred])
