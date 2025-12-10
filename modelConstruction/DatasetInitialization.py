"""
DatasetInitialization.py
Description: Ensures the dataset is good to go!

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import pandas as pd

from modelConstruction.DatasetNormalization import *
from controllers.APIComm import *

def dataset_init():
    """
    Initializes and preprocesses datasets including downloading, reorganizing,
    and pre-cropping images based on the provided datasheet.

    This method automates dataset preparation, ensuring all required pre-processing
    steps such as downloading, restructuring, and cropping images are executed.

    :param datasheet: DataFrame containing metadata for dataset images. Expected to
        include the column 'image_path' which specifies the file path for each image.
    :type datasheet: pd.DataFrame
    :return: None
    """

    download_dataset()
    delete_dataset()

    #Now that we have data, lets get the training datasheet.
    datasheet = get_datasheet()

    print("Initializing dataset via pre-cropping to a bounding box...")
    #Pre-Crop all images in dataset
    for _, row in datasheet.iterrows():
        path = f"./data/filtered_cars/train/{row['image_path']}"
        crop_dataset_image(row, path)
    print("Dataset cropped successfully...")


def get_datasheet(csv_path: str = "./data/anno_train_filtered.csv"):
    datasheet = pd.read_csv(csv_path, header=None)
    datasheet.columns = ['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
    return datasheet