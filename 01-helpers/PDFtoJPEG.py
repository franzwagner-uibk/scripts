import fitz  # PyMuPDF
import os

# Input Variables
INPUT_PDF = r"02-Literature\CCCA_Factsheet_Klimawandel_2024pdf.pdf"  # Replace with your input PDF path
OUTPUT_DIR = r"02-Literature\jpg\CCCA_Factsheet_Klimawandel_2024pdf"  # Replace with your desired output directory

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Open the PDF
doc = fitz.open(INPUT_PDF)

# Convert each page to an image and save it
for i, page in enumerate(doc):
    pix = page.get_pixmap(dpi=300)  # High DPI for better quality
    output_path = os.path.join(OUTPUT_DIR, f"page_{i + 1}.jpg")
    pix.save(output_path, "jpeg")

print(f"Conversion completed! Images saved in '{OUTPUT_DIR}'")
