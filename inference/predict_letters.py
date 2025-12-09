"""
Predict letters from extracted images using a trained model.
Outputs all predictions as a single paragraph.
"""

import numpy as np
from pathlib import Path
from tensorflow import keras
import cv2


class LetterPredictor:
    """Predict letters from images and combine into text."""

    # ============== CONFIGURE THESE ==============
    images_path = "../detection/extracted_letters"
    model_path = "../models/saved/simple_model_combined2.keras"
    # =============================================

    # EMNIST letter mapping (1-26 -> A-Z)
    LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    def __init__(self):
        self.images_path = Path(self.images_path)
        self.model_path = Path(self.model_path)
        self.model = None

    def load_model(self):
        """Load the trained model."""
        print(f"Loading model from {self.model_path}...")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded.")

    def preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image to match EMNIST training format."""
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Resize to 28x28 if needed
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        # Apply EMNIST orientation fix (transpose + flip)
        # EMNIST images are transposed and flipped compared to natural writing
        img = np.transpose(img)
        img = np.fliplr(img)

        # Normalize to 0-1
        img = img.astype(np.float32) / 255.0

        # Reshape for model input (batch, height, width, channels)
        img = img.reshape(1, 28, 28, 1)

        return img

    def predict_single(self, img: np.ndarray) -> str:
        """Predict a single letter from an image."""
        processed = self.preprocess_image(img)
        prediction = self.model.predict(processed, verbose=0)
        class_idx = np.argmax(prediction[0])
        return self.LABELS[class_idx]

    def load_images(self) -> list:
        """Load all images from the images path, sorted by filename."""
        image_files = sorted(self.images_path.glob("*.png"))

        images = []
        for img_path in image_files:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((img_path.name, img))

        return images

    def run(self) -> str:
        """Run prediction on all images and return combined text."""
        if not self.images_path.exists():
            print(f"Error: Images path not found: {self.images_path}")
            return ""

        if not self.model_path.exists():
            print(f"Error: Model not found: {self.model_path}")
            return ""

        # Load model
        self.load_model()

        # Load images
        print(f"Loading images from {self.images_path}...")
        images = self.load_images()
        print(f"Found {len(images)} images")

        # Predict each letter
        print("Predicting letters...")
        letters = []
        for filename, img in images:
            letter = self.predict_single(img)
            letters.append(letter)

        # Combine into paragraph
        result = "".join(letters)

        print(f"\n{'=' * 50}")
        print("PREDICTED TEXT:")
        print('=' * 50)
        print(result)
        print('=' * 50)

        return result


if __name__ == "__main__":
    predictor = LetterPredictor()
    text = predictor.run()
    print(text)