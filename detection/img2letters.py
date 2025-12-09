from pathlib import Path
from PIL import Image
import string

"""
Description of this script:

Given an image of every letter in the alphabet, each image has a grid of 5x5 letters.
This script crops the grid coming inside them. We know the position of the grid in the image.
Then, it splits the grid into 25 images, one for each letter. These images are then going
to be used in a classifier to recognize handwritten letters.
"""

INPUT_FOLDER = Path("./output/page_images")
OUTPUT_FOLDER = Path("./output/letter_images")
IMAGE_EXTENSION = ".png"
NUM_ROWS = 5
NUM_COLS = 5
FIRST_ROWS_IN_CAPS = 2

def process_images() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    # All the 26 letters
    letters = string.ascii_uppercase
    for letter in letters:
        image_path = INPUT_FOLDER / f"{letter}{IMAGE_EXTENSION}"
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue

        image = Image.open(image_path)

        # Valid from B and onward
        top_left_x = 320
        top_left_y = 528
        bottom_right_x = 1657
        bottom_right_y = 1864

        if letter == 'A': # this is the exception
            top_left_x = 310
            top_left_y = 1018
            bottom_right_x = 1647
            bottom_right_y = 2355

        image_to_letters(
            image,
            letter,
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y
        )

def image_to_letters(image: Image.Image,
                     letter: str,
                     top_left_x: int,
                     top_left_y: int,
                     bottom_right_x: int,
                     bottom_right_y: int) -> None:
    """
    Given an image containing a grid of letters, crop and save each letter as a separate image.

    (0,0) ------------------> x
      |
      |
      v
      y

    """
    # Compute grid cell size
    grid_width = bottom_right_x - top_left_x
    grid_height = bottom_right_y - top_left_y
    cell_width = grid_width / NUM_COLS
    cell_height = grid_height / NUM_ROWS

    # Process each of the 25 grid cells
    for row_index in range(NUM_ROWS):
        for column_index in range(NUM_COLS):
            left_coordinate = int(top_left_x + column_index * cell_width)
            top_coordinate = int(top_left_y + row_index * cell_height)
            right_coordinate = int(left_coordinate + cell_width)
            bottom_coordinate = int(top_coordinate + cell_height)

            cropped_cell = image.crop((
                left_coordinate,
                top_coordinate,
                right_coordinate,
                bottom_coordinate
            ))

            # Convert to grayscale
            grayscale_cell = cropped_cell.convert("L")

            # Save the final processed cell
            letter_type = letter.upper() if row_index < FIRST_ROWS_IN_CAPS else letter.lower()
            output_filepath = OUTPUT_FOLDER / f"{letter_type}_row{row_index}_col{column_index}.png"
            grayscale_cell.save(output_filepath)
            print(f"Saved letter image: {output_filepath}")

if __name__ == "__main__":
    process_images()