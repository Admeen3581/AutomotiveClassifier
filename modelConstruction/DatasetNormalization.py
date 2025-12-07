"""
DatasetNormalization.py
Description: Normalizes the entire dataset.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import cv2 as cv
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

def get_datasheet(image_path: str):
    """
    Reads a CSV datasheet corresponding to a given image path. The method determines whether
    the image path corresponds to 'test' or 'train' and fetches the relevant CSV file.

    :param image_path: The path to an image file as a string.
    :type image_path: str
    :return: A pandas DataFrame containing the data read from the appropriate CSV file.
    :rtype: pandas.DataFrame
    """

    split = image_path.split("/")[3]#test or train
    return pd.read_csv(f"./data/filtered_cars/anno_{split}.csv")

def get_bounding_box(datasheet: pd.DataFrame, image_path: str):
    """
    Extracts the bounding box coordinates for a given image from a dataset.

    :param datasheet: The DataFrame containing the bounding box data, where
        each row corresponds to a specific image.
    :param image_path: The path of the image for which the bounding box
        coordinates are to be extracted.
    :type image_path: str
    :return: A tuple containing the bounding box coordinates
        (x_min, y_min, x_max, y_max).
    :rtype: tuple
    :raises ValueError: If the image path is not found in the dataset.
    """

    image_data = datasheet[datasheet[0] == image_path.split()[-1]]
    if image_data.empty:
        raise ValueError(f"Image {image_path} not found in the dataset.")

    coordinates = image_data.iloc[0][1:5].tolist()

    x_min = coordinates[0]
    y_min = coordinates[1]
    x_max = coordinates[2]
    y_max = coordinates[3]

    return x_min, y_min, x_max, y_max

def crop_dataset_image(datasheet: pd.DataFrame, image_path: str):
    """
    Crops a region of interest from an image using bounding box coordinates derived
    from the provided datasheet and converts the cropped image to RGB format.

    :param datasheet: Pandas DataFrame containing the data used to determine the
        bounding box for cropping the image.
        Type hints enforce ensuring the correctness of inputs.
    :param image_path: The path of the image for which the bounding box
        coordinates are to be cropped into the image.
    :type image_path: str
    """

    img_bgr = cv.imread(image_path)
    if(img_bgr is None):
        raise FileNotFoundError(f"Image not found at {image_path}.")

    x_min, y_min, x_max, y_max = get_bounding_box(datasheet, image_path)

    crop_img_bgr = img_bgr[y_min:y_max, x_min:x_max] #crops to bounding box <- (for model accuracy)
    if img_bgr.size == 0: #failsafe incase of faulty bounds
        print(f"Warning: Empty crop for {image_path}. BBox: {x_min, y_min, x_max, y_max}")
        crop_img_bgr = img_bgr

    return cv.cvtColor(crop_img_bgr, cv.COLOR_BGR2RGB)

def normalize_image(img : np.ndarray, target_size = 224):
    """
    Normalizes an input image to the specified target size and
    applies Imagenet normalization. The function resizes an image to the target
    dimensions, converts it to a tensor, and applies standard normalization
    (mean and standard deviation) for an ImageNET pre-trained CNN learning model.

    :param img: Image to be normalized.
    :type img: numpy.ndarray
    :param target_size: Size to resize the image, default is 224.
    :type target_size: int
    :return: Normalized image as a tensor.
    :rtype: torch.Tensor
    """
    resized_img = cv.resize(
        img,
        (target_size, target_size),
        interpolation=cv.INTER_AREA
    )

    tensor_img = transforms.ToTensor()(resized_img)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_img)#IMG_NET values

    return normalize(tensor_img)