"""
Prediction Evaluation Script for Handwritten Letters

Scans input_data/onenote_raw folder, predicts each letter,
and generates a comparison table with expected vs predicted results.
"""

import os
import numpy as np
from PIL import Image
from keras import models
from collections import defaultdict


class LetterPredictor:
    """Handles model loading and letter predictions."""

    def __init__(self, model_path='../models/saved/simple_model_preprocessed.keras'):
        self.model = models.load_model(model_path)
        # Mapping: index to letter (adjust if your model uses different mapping)
        self.index_to_letter = {i: chr(i + ord('A')) for i in range(26)}
        self.letter_to_index = {chr(i + ord('A')): i for i in range(26)}

    def preprocess_image(self, image_path):
        """Preprocess image for model prediction."""
        image = Image.open(image_path).convert('L')  # Grayscale
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        image = np.array(image, dtype=np.float32)

        # Invert if needed (EMNIST expects white on black)
        image = 255.0 - image

        # Normalize
        image = image / 255.0

        # Reshape for model: (1, 28, 28, 1)
        image = image.reshape(1, 28, 28, 1)
        return image

    def predict(self, image_path):
        """
        Predict the letter in an image.

        Returns:
            dict with 'predicted_letter', 'confidence', 'all_probabilities'
        """
        processed = self.preprocess_image(image_path)
        probabilities = self.model.predict(processed, verbose=0)[0]

        predicted_index = np.argmax(probabilities)
        predicted_letter = self.index_to_letter.get(predicted_index, '?')
        confidence = probabilities[predicted_index]

        return {
            'predicted_letter': predicted_letter,
            'predicted_index': predicted_index,
            'confidence': confidence,
            'all_probabilities': probabilities
        }


def scan_folder(folder_path):
    """
    Scan folder and get one PNG for each starting letter.

    Returns:
        dict: {expected_letter: file_path}
    """
    files_by_letter = defaultdict(list)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.png'):
            first_letter = filename[0].upper()
            if first_letter.isalpha():
                files_by_letter[first_letter].append(
                    os.path.join(folder_path, filename)
                )

    # Take first file for each letter
    result = {}
    for letter in sorted(files_by_letter.keys()):
        result[letter] = files_by_letter[letter][0]

    return result


def generate_report(predictor, files_dict):
    """Generate prediction report for all files."""
    results = []

    for expected_letter, file_path in sorted(files_dict.items()):
        prediction = predictor.predict(file_path)

        results.append({
            'file': os.path.basename(file_path),
            'expected': expected_letter,
            'predicted': prediction['predicted_letter'],
            'confidence': prediction['confidence'],
            'correct': expected_letter == prediction['predicted_letter'],
            'probabilities': prediction['all_probabilities']
        })

    return results


def print_summary_table(results):
    """Print a summary table with expected vs predicted."""
    print("\n" + "=" * 70)
    print("PREDICTION SUMMARY TABLE")
    print("=" * 70)
    print(f"{'File':<20} {'Expected':<10} {'Predicted':<10} {'Confidence':<12} {'Match'}")
    print("-" * 70)

    correct_count = 0
    for r in results:
        match = "✓" if r['correct'] else "✗"
        if r['correct']:
            correct_count += 1
        print(f"{r['file']:<20} {r['expected']:<10} {r['predicted']:<10} {r['confidence']:>10.2%}   {match}")

    print("-" * 70)
    accuracy = correct_count / len(results) * 100 if results else 0
    print(f"Accuracy: {correct_count}/{len(results)} ({accuracy:.1f}%)")
    print("=" * 70)


def print_detailed_table(results):
    """Print detailed table with all class probabilities."""
    print("\n" + "=" * 120)
    print("DETAILED PROBABILITY TABLE")
    print("=" * 120)

    # Header
    header = f"{'Expected':<10}"
    for i in range(26):
        header += f" {chr(i + ord('A')):>5}"
    print(header)
    print("-" * 120)

    # Each row
    for r in results:
        row = f"{r['expected']:<10}"
        for i, prob in enumerate(r['probabilities']):
            # Highlight if this is the expected letter
            if i == ord(r['expected']) - ord('A'):
                row += f" [{prob:>4.1%}]"[:-1]  # Remove last char to fit
            else:
                row += f" {prob:>5.1%}"
        print(row)

    print("=" * 120)


def print_top3_predictions(results):
    """Print top 3 predictions for each letter."""
    print("\n" + "=" * 80)
    print("TOP 3 PREDICTIONS PER LETTER")
    print("=" * 80)
    print(f"{'Expected':<10} {'#1':<18} {'#2':<18} {'#3':<18}")
    print("-" * 80)

    for r in results:
        probs = r['probabilities']
        top3_indices = np.argsort(probs)[-3:][::-1]

        predictions = []
        for idx in top3_indices:
            letter = chr(idx + ord('A'))
            prob = probs[idx]
            predictions.append(f"{letter}: {prob:>6.2%}")

        expected = r['expected']
        match_symbol = "✓" if r['correct'] else "✗"
        print(f"{expected} {match_symbol:<8} {predictions[0]:<18} {predictions[1]:<18} {predictions[2]:<18}")

    print("=" * 80)


def main():
    # Configuration
    folder_path = '../input_data/onenote_raw'
    model_path = '../models/saved/simple_model_preprocessed.keras'

    print(f"Loading model from: {model_path}")
    predictor = LetterPredictor(model_path)

    print(f"Scanning folder: {folder_path}")
    files_dict = scan_folder(folder_path)
    print(f"Found {len(files_dict)} letters: {', '.join(sorted(files_dict.keys()))}")

    print("\nRunning predictions...")
    results = generate_report(predictor, files_dict)

    # Print all tables
    print_summary_table(results)
    print_top3_predictions(results)
    print_detailed_table(results)

    # Optional: Save to CSV
    save_csv = input("\nSave results to CSV? (y/n): ").lower().strip()
    if save_csv == 'y':
        import csv
        csv_path = 'prediction_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            header = ['File', 'Expected', 'Predicted', 'Confidence', 'Correct']
            header += [chr(i + ord('A')) for i in range(26)]
            writer.writerow(header)
            # Data
            for r in results:
                row = [r['file'], r['expected'], r['predicted'],
                       f"{r['confidence']:.4f}", r['correct']]
                row += [f"{p:.6f}" for p in r['probabilities']]
                writer.writerow(row)
        print(f"Saved to {csv_path}")


if __name__ == "__main__":
    main()