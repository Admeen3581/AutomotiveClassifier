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
import pandas as pd
import torch
from torchvision import transforms



def get_datasheet(image_path: str):
    """
    Reads a CSV datasheet corresponding to a given image path. The method determines whether
    the image path corresponds to 'test' or 'train' and fetches the relevant CSV file.

    :param image_path: The path to an image file as a string. Must be of path 'train' or 'test'.
    :type image_path: str
    :return: A pandas DataFrame containing the data read from the appropriate CSV file.
    :rtype: pandas.DataFrame
    """

    split = image_path.split("/")[3]#test or train
    datasheet = pd.read_csv(f"./data/anno_{split}.csv", header=None)
    datasheet.columns = ['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
    return datasheet

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
    Crops and processes an image based on the bounding box obtained from a dataset.

    The function reads an image, computes a bounding box from the datasheet, crops
    the image to the bounding box dimensions, converts it to RGB  format, and then
    converts it into a tensor preparation for further processing.

    :param datasheet: A pandas DataFrame containing metadata provided by KaggleAPI
    :type datasheet: pandas.DataFrame
    :param image_path: The file path to the image that is to be processed. It should point to a valid image file.
    :type image_path: str
    :return: A tensor object that represents the processed version of the cropped RGB image.
    :rtype: torch.Tensor
    :raises FileNotFoundError: If the image could not be found at the specified `image_path`.
    """

    img_bgr = cv.imread(image_path)
    if(img_bgr is None):
        raise FileNotFoundError(f"Image not found at {image_path}.")

    x_min, y_min, x_max, y_max = get_bounding_box(datasheet, image_path)

    crop_img_bgr = img_bgr[y_min:y_max, x_min:x_max] #crops to bounding box <- (for model accuracy)
    if img_bgr.size == 0: #failsafe incase of faulty bounds
        print(f"Warning: Empty crop for {image_path}. BBox: {x_min, y_min, x_max, y_max}")
        crop_img_bgr = img_bgr

    img_rgb =  cv.cvtColor(crop_img_bgr, cv.COLOR_BGR2RGB)

    return transforms.ToTensor()(img_rgb)

def normalize_image(img : torch.Tensor, target_size = 224):
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
    normalization = transforms.Compose([
        transforms.Resize((target_size, target_size)),  # Re-scale to final size if augmentation changed it
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])#IMG_NET values
    ])

    return normalization(img)

def augment_image(img : torch.Tensor):
    """
    Applies a sequence of data augmentation transformations to an input image.

    This function performs random rotation, affine transformations (with
    random translation and scaling), horizontal flipping with a fixed
    probability, and color jittering (adjusting brightness, contrast,
    saturation, and hue).

    :param img: A tensor representing an image to be augmented. It is expected
        to conform to the input format required by torchvision.transforms.
    :type img: torch.Tensor
    :return: The augmented image after applying the transformations.
    :rtype: torch.Tensor
    """
    augmentation = transforms.Compose([
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
    ])

    return augmentation(img)
