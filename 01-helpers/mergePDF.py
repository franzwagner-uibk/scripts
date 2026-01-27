import os
from PyPDF2 import PdfMerger

# Define the input directory and output file
input_directory = r'C:\Users\franz\Nextcloud\Tickets\Abrechnung_Bozen_2025_12'
output_file = r'C:\Users\franz\Nextcloud\Tickets\Abrechnung_Bozen_2025_12\Merged_Output.pdf'

def merge_pdfs_in_directory(directory, output_filename):
    # Initialize a PdfMerger object
    merger = PdfMerger()

    # List all files in the directory
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith('.pdf'):
            filepath = os.path.join(directory, filename)
            merger.append(filepath)
            print(f"Added {filename}")

    # Write the merged PDF to the output file
    merger.write(output_filename)
    merger.close()
    print(f"Merged PDF saved as {output_filename}")

# Usage
merge_pdfs_in_directory(input_directory, output_file)
