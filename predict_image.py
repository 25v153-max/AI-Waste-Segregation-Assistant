import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "/home/ecoassist/ecoassist/models/best_mobilenet_wrapper_final.pth"
IMAGE_PATH = "/home/ecoassist/ecoassist/ecoassist_test.jpg"

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]

model = models.mobilenet_v3_small(weights=None)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    len(classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

with torch.inference_mode():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)

print("\n========== ECOASSIST IMAGE PREDICTION ==========")
print("Predicted waste:", classes[prediction.item()])
print(f"Confidence: {confidence.item() * 100:.2f}%")
print("================================================")
