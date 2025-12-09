import fitz
from pathlib import Path
import string

INPUT_FILE = Path("./data/AC_handwriting.pdf")
OUTPUT_FOLDER = Path("./output/page_images")

def from_pdf_to_images(input_file: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)

    # Read the pdf file
    if not input_file.exists():
        print(f"File not found: {input_file}")
        exit(1)

    doc = fitz.open(input_file)
    letters = string.ascii_uppercase # array of A-Z

    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=300)
        filepath = output_folder / f"{letters[i]}.png"
        pix.save(filepath)
        print(f"Saved page {i} as image to {filepath}")

if __name__ == "__main__":
    from_pdf_to_images(INPUT_FILE, OUTPUT_FOLDER)