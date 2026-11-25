import torch
import torch.nn as nn
import librosa
import numpy as np

MODEL_PATH = "/home/ecoassist/ecoassist/models/audio_cnn_best.pth"
AUDIO_PATH = "/home/ecoassist/ecoassist_test.wav"

class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


print("Loading audio model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

classes = checkpoint["classes"]

model = AudioCNN(len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Audio model loaded successfully!")

print("Processing audio...")

audio, sr = librosa.load(
    AUDIO_PATH,
    sr=16000,
    mono=True
)

target_length = 5 * 16000

if len(audio) > target_length:
    start = (len(audio) - target_length) // 2
    audio = audio[start:start + target_length]
elif len(audio) < target_length:
    pad = target_length - len(audio)
    left = pad // 2
    audio = np.pad(audio, (left, pad - left))

mel = librosa.feature.melspectrogram(
    y=audio,
    sr=16000,
    n_mels=64,
    n_fft=1024,
    hop_length=256,
    power=2.0
)

mel = librosa.power_to_db(mel, ref=np.max)
mel = (mel - mel.mean()) / (mel.std() + 1e-6)

input_tensor = torch.tensor(
    mel,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

print("Running audio prediction...")

with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.softmax(output, dim=1)
    confidence, predicted = torch.max(probabilities, dim=1)

predicted_class = classes[predicted.item()]
confidence_value = confidence.item() * 100

print("\n========== ECOASSIST AUDIO PREDICTION ==========")
print("Audio file:", AUDIO_PATH)
print("Predicted category:", predicted_class)
print(f"Confidence: {confidence_value:.2f}%")
print("================================================")
