"""
Extract individual letters from a handwritten PDF and save as PNG files.
"""

import cv2
import numpy as np
from pdf2image import convert_from_path
from pathlib import Path


class LetterExtractor:
    """Extract individual letters from handwritten PDFs."""

    # ============== CONFIGURE THESE ==============
    input_path = "example_onenote.pdf"
    output_path = "extracted_letters"
    # =============================================

    # Detection parameters
    min_area = 100
    max_area = 10000
    output_size = 28
    padding = 5

    def __init__(self):
        self.input_path = Path(self.input_path)
        self.output_path = Path(self.output_path)

    def pdf_to_image(self) -> np.ndarray:
        """Convert PDF to a numpy array (grayscale)."""
        pages = convert_from_path(str(self.input_path), dpi=300)
        img = np.array(pages[0])
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    def preprocess_image(self, gray: np.ndarray) -> np.ndarray:
        """Preprocess image for better contour detection."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def extract_letter_contours(self, binary: np.ndarray) -> list:
        """Find contours that likely represent individual letters."""
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        letter_boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = w / h if h > 0 else 0
                if 0.2 < aspect < 5.0:
                    letter_boxes.append((x, y, w, h))

        # Sort by position: top-to-bottom, then left-to-right
        letter_boxes.sort(key=lambda b: (b[1] // 50, b[0]))

        return letter_boxes

    def extract_and_save_letters(self, gray: np.ndarray, binary: np.ndarray, boxes: list) -> int:
        """Extract each letter and save as a PNG."""
        self.output_path.mkdir(parents=True, exist_ok=True)

        saved_count = 0
        for i, (x, y, w, h) in enumerate(boxes):
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(gray.shape[1], x + w + self.padding)
            y2 = min(gray.shape[0], y + h + self.padding)

            letter_img = binary[y1:y2, x1:x2]

            h_letter, w_letter = letter_img.shape
            max_dim = max(h_letter, w_letter)
            square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)

            y_offset = (max_dim - h_letter) // 2
            x_offset = (max_dim - w_letter) // 2
            square_img[y_offset:y_offset + h_letter, x_offset:x_offset + w_letter] = letter_img

            resized = cv2.resize(square_img, (self.output_size, self.output_size), interpolation=cv2.INTER_AREA)

            output_file = self.output_path / f"letter_{i:04d}.png"
            cv2.imwrite(str(output_file), resized)
            saved_count += 1

        return saved_count

    def run(self):
        """Run the full extraction pipeline."""
        if not self.input_path.exists():
            print(f"Error: PDF not found: {self.input_path}")
            return

        print(f"Processing: {self.input_path}")

        print("Converting PDF to image...")
        gray = self.pdf_to_image()

        print("Preprocessing...")
        binary = self.preprocess_image(gray)

        print("Detecting letters...")
        boxes = self.extract_letter_contours(binary)
        print(f"Found {len(boxes)} potential letters")

        print(f"Saving letters to {self.output_path}/")
        count = self.extract_and_save_letters(gray, binary, boxes)
        print(f"Saved {count} letter images")


if __name__ == "__main__":
    extractor = LetterExtractor()
    extractor.run()
