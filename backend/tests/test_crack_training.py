import tempfile
import unittest
from pathlib import Path

import torch

from app.crack_model import CrackClassifier
from train_crack_model import discover_class_directories, stratified_split


class CrackTrainingTests(unittest.TestCase):
    def test_discovers_positive_negative_and_spaced_class_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset_root = Path(temporary_directory)
            positive = dataset_root / "Positive"
            negative = dataset_root / "No Crack"
            positive.mkdir()
            negative.mkdir()

            classes = discover_class_directories(dataset_root)

            self.assertEqual(classes[0], negative)
            self.assertEqual(classes[1], positive)

    def test_stratified_split_keeps_both_classes_in_all_splits(self):
        samples = [(Path(f"negative-{index}.jpg"), 0) for index in range(100)]
        samples += [(Path(f"positive-{index}.jpg"), 1) for index in range(100)]

        train, validation, test = stratified_split(samples, seed=7)

        self.assertEqual((len(train), len(validation), len(test)), (140, 30, 30))
        for split in (train, validation, test):
            self.assertEqual(sum(label == 0 for _, label in split), sum(label == 1 for _, label in split))

    def test_cnn_returns_one_logit_per_image(self):
        model = CrackClassifier()
        predictions = model(torch.zeros(2, 3, 224, 224))

        self.assertEqual(tuple(predictions.shape), (2,))


if __name__ == "__main__":
    unittest.main()
