###############################################
# Description: Script to create a list of filenames from an input directory
# Author: Franz Wagner
# Date: 2025-01-07
###############################################

import os
import logging

# Input and Output Variables
INPUT_DIR = r"G:\2023 AlpSnow\data\satellite\processed\S1_WSM\tif"  # Replace with your input directory path
OUTPUT_FILE = r"C:\Daten\PhD\openamundsen_da\examples\rofental\misc\filenames_WSM.txt"  # Replace with your output file path
WRITE_TO_FILE = True  # Set to False if you do not want to write to a file

# Task List
# 1. Validate the input directory
# 2. Get a sorted list of folders and filenames by file type from the directory
# 3. Print the list of folders and filenames to the console
# 4. Optionally write the list of folders and filenames to the output file

def validate_input_directory(directory):
    """Validate if the input directory exists."""
    if not os.path.exists(directory):
        logging.error(f"Input directory does not exist: {directory}")
        raise FileNotFoundError(f"Directory not found: {directory}")
    if not os.path.isdir(directory):
        logging.error(f"Path is not a directory: {directory}")
        raise NotADirectoryError(f"Not a directory: {directory}")
    logging.info(f"Input directory validated: {directory}")

def get_sorted_directory_contents(directory):
    """Get sorted lists of folders and filenames by file type from the specified directory."""
    contents = os.listdir(directory)
    folders = sorted([item for item in contents if os.path.isdir(os.path.join(directory, item))])
    files = sorted([item for item in contents if os.path.isfile(os.path.join(directory, item))], key=lambda x: (os.path.splitext(x)[1], x))
    logging.info(f"Found {len(folders)} folders and {len(files)} files in directory: {directory}")
    return folders, files

def print_directory_contents(folders, files):
    """Print the sorted lists of folders and filenames to the console."""
    logging.info("Printing directory contents to console:")
    for folder in folders:
        print(folder)
    for file in files:
        print(file)

def write_directory_contents_to_file(folders, files, output_file):
    """Write the sorted lists of folders and filenames to a specified output file."""
    with open(output_file, "w") as f:
        for folder in folders:
            f.write(f"{folder}\n")
        for file in files:
            f.write(f"{file}\n")
    logging.info(f"Directory contents written to file: {output_file}")

def main():
    """Main function to execute the task list."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        # Step 1: Validate the input directory
        validate_input_directory(INPUT_DIR)

        # Step 2: Get sorted lists of folders and filenames by file type
        folders, files = get_sorted_directory_contents(INPUT_DIR)

        # Step 3: Print folders and filenames to the console
        print_directory_contents(folders, files)

        # Step 4: Optionally write folders and filenames to the output file
        if WRITE_TO_FILE:
            write_directory_contents_to_file(folders, files, OUTPUT_FILE)

        logging.info("Task completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
