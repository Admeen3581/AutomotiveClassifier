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
DATA_SPLITS = ["train", "test"]


def reorganize_dataset():

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
                make_dest_dir = current_target_dir / car_make
                os.makedirs(make_dest_dir, exist_ok=True)

                for image_file in folder.glob("*.jpg"):
                    shutil.copy2(image_file, make_dest_dir / image_file.name)