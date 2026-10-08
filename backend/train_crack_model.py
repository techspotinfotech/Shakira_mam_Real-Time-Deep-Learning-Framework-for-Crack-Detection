from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.crack_model import IMAGE_SIZE, CrackClassifier, preprocess_image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "Dataset" / "Concrete Crack Images for Classification"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "backend" / "models" / "crack_classifier.pth"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ConcreteCrackDataset(Dataset):
    def __init__(self, samples: Sequence[tuple[Path, int]], augment: bool = False) -> None:
        self.samples = list(samples)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image_tensor = preprocess_image(image)

        if self.augment:
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[2])
            if random.random() < 0.5:
                image_tensor = torch.flip(image_tensor, dims=[1])

        return image_tensor, torch.tensor(label, dtype=torch.float32)


def _normalized_name(name: str) -> str:
    return "".join(character for character in name.lower() if character.isalnum())


def discover_class_directories(data_dir: Path) -> dict[int, Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")

    directories = { _normalized_name(path.name): path for path in data_dir.iterdir() if path.is_dir() }
    positive_names = {"positive", "crack", "cracked", "withcrack", "crackedconcrete"}
    negative_names = {"negative", "nocrack", "withoutcrack", "noncrack", "uncracked"}
    positive_dir = next((path for name, path in directories.items() if name in positive_names), None)
    negative_dir = next((path for name, path in directories.items() if name in negative_names), None)

    if positive_dir is None or negative_dir is None:
        found = sorted(path.name for path in data_dir.iterdir() if path.is_dir())
        raise ValueError(
            "Expected one crack-positive folder (Positive/crack) and one no-crack "
            f"folder (Negative/no crack) inside {data_dir}. Found: {found}"
        )

    return {0: negative_dir, 1: positive_dir}


def collect_samples(data_dir: Path) -> list[tuple[Path, int]]:
    class_dirs = discover_class_directories(data_dir)
    samples: list[tuple[Path, int]] = []
    counts: dict[int, int] = {}

    for label, class_dir in class_dirs.items():
        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        counts[label] = len(images)
        samples.extend((path, label) for path in images)

    if not counts.get(0) or not counts.get(1):
        raise ValueError(f"Both classes need images. Found no_crack={counts.get(0, 0)}, crack={counts.get(1, 0)}")

    print(f"Dataset images: no_crack={counts[0]}, crack={counts[1]}, total={len(samples)}")
    return samples


def stratified_split(
    samples: Sequence[tuple[Path, int]], seed: int = 42
) -> tuple[list[tuple[Path, int]], list[tuple[Path, int]], list[tuple[Path, int]]]:
    rng = random.Random(seed)
    splits: list[list[tuple[Path, int]]] = [[], [], []]

    for label in (0, 1):
        class_samples = [sample for sample in samples if sample[1] == label]
        rng.shuffle(class_samples)
        train_end = int(len(class_samples) * 0.70)
        validation_end = int(len(class_samples) * 0.85)
        splits[0].extend(class_samples[:train_end])
        splits[1].extend(class_samples[train_end:validation_end])
        splits[2].extend(class_samples[validation_end:])

    for split in splits:
        rng.shuffle(split)
    return splits[0], splits[1], splits[2]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += ((torch.sigmoid(logits) >= 0.5) == (labels >= 0.5)).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def train_model(
    data_dir: Path = DEFAULT_DATA_DIR,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    epochs: int = 15,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    samples = collect_samples(data_dir)
    train_samples, validation_samples, test_samples = stratified_split(samples, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; split sizes: train={len(train_samples)}, validation={len(validation_samples)}, test={len(test_samples)}")

    train_loader = DataLoader(ConcreteCrackDataset(train_samples, augment=True), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(ConcreteCrackDataset(validation_samples), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    test_loader = DataLoader(ConcreteCrackDataset(test_samples), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = CrackClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_validation_accuracy = -1.0
    best_epoch = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        validation_loss, validation_accuracy = evaluate(model, validation_loader, criterion, device)
        train_loss = running_loss / len(train_samples)
        print(
            f"Epoch {epoch:02d}/{epochs} - train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} val_accuracy={validation_accuracy:.4f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": {"no_crack": 0, "crack": 1},
                    "image_size": IMAGE_SIZE,
                    "best_epoch": best_epoch,
                    "validation_accuracy": best_validation_accuracy,
                },
                checkpoint_path,
            )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    metrics = {
        "best_epoch": float(best_epoch),
        "validation_accuracy": best_validation_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    print(f"Test: loss={test_loss:.4f} accuracy={test_accuracy:.4f}")
    print(f"Saved model: {checkpoint_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CNN to classify concrete images as crack/no-crack.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Folder containing Positive and Negative class folders.")
    parser.add_argument("--output", type=Path, default=DEFAULT_CHECKPOINT, help="Output path for the trained .pth model.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_model(args.data_dir, args.output, args.epochs, args.batch_size, args.seed)


if __name__ == "__main__":
    main()
