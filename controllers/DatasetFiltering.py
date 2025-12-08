"""
DatasetFiltering.py
Description: Controller for cleaning the dataset(s).

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import os
import shutil
from pathlib import Path
from controllers.CarMakeData import car_brands

#Constants
SOURCE_DIR = Path("./data/stanford_cars/car_data/car_data")
TARGET_DIR = Path("./data/filtered_cars")
SHEETS_DIR = Path("./data/stanford_cars/")
SHEETS_TARGET_DIR = Path("./data/")
DATA_SPLITS = ["train", "test"]


def reorganize_dataset():
    """
    Cleans dataset by removing unnecessary files and structuring to fit model requirements.

    Reorganizes a dataset by splitting images into target directories based on their subfolder names,
    representing car brands. Each car brand folder is grouped and copied to the corresponding target
    dataset structure while ensuring that only valid car brands are included.

    :raises FileNotFoundError: If the source directory or one of its subdirectories does not exist.
    :raises OSError: If there is an error creating directories or copying files.
    """

    for split in DATA_SPLITS:

        current_source_dir = SOURCE_DIR / split
        current_target_dir = TARGET_DIR / split

        if not current_target_dir.exists():
            os.makedirs(current_target_dir)

        # List subfolders in source directory
        subfolders = [f for f in current_source_dir.iterdir() if f.is_dir()]

        for folder in subfolders:
            folder_name = folder.name

            car_make = folder_name.split(' ')[0]#Note: 2+ word makes will be shortened to 1 word (e.g. Aston Martin -> Aston)

            #Filters out removed makes from dataset
            if car_make in car_brands:
                for image_file in folder.glob("*.jpg"):
                    shutil.copy2(image_file, current_target_dir / image_file.name)


    move_sheets()

def move_sheets():
    """
    Moves all CSV sheets from the source directory to the target directory.

    This function iterates through all the files in the directory specified
    by the global variable ``SHEETS_DIR``, filtering only the files with a
    ``.csv`` extension. Each matching file is then moved to the directory
    specified by the global variable ``TARGET_DIR`` using the ``shutil.move``
    function.

    :raises FileNotFoundError: If the source directory or files do not exist.
    :raises OSError: For other errors during file moving, such as invalid
        file paths.
    """
    for sheet in SHEETS_DIR.glob("*.csv"):
        shutil.move(sheet, SHEETS_TARGET_DIR)