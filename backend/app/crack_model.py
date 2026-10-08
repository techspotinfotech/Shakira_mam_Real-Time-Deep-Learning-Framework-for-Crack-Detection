from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

IMAGE_SIZE = 224
CLASS_NAMES = {0: "no_crack", 1: "crack"}


class CrackClassifier(nn.Module):
    """Small CNN for binary concrete crack classification."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(3, 16),
            self._block(16, 32),
            self._block(32, 64),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.25), nn.Linear(64, 1))

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images)).squeeze(1)


def preprocess_image(image: Image.Image) -> torch.Tensor:
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def predict_crack_image(image_bytes: bytes, checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Trained model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = CrackClassifier().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with Image.open(BytesIO(image_bytes)) as image:
        input_tensor = preprocess_image(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probability_crack = torch.sigmoid(model(input_tensor)).item()

    label = "crack" if probability_crack >= 0.5 else "no_crack"
    confidence = probability_crack if label == "crack" else 1.0 - probability_crack
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "probability_crack": round(probability_crack, 4),
    }
