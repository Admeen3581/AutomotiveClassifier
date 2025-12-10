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
import torch
from torchvision import transforms


def get_bounding_box(row: list):
    """
    Extracts the bounding box coordinates from a given data row.

    This function reads coordinate values from a provided data row.
    It returns the extracted bounding box coordinates as a tuple.

    :param row: A dictionary containing bounding box coordinates with keys
        'x_min', 'y_min', 'x_max', and 'y_max'.
    :type row: list
    :return: A tuple containing the bounding box coordinates in the order
        (x_min, y_min, x_max, y_max).
    :rtype: tuple
    """

    # Extract the coordinates from the row
    x_min = row['x_min']
    y_min = row['y_min']
    x_max = row['x_max']
    y_max = row['y_max']

    return x_min, y_min, x_max, y_max

def crop_dataset_image(row: list, image_path: str):
    """
    Crops an image based on the bounding box defined by the input data.

    This function performs image preprocessing by reading an image from a specified path, extracting
    a region from the image defined by bounding box coordinates, resizing the region to the required
    dimensions, converting color format for consistency, and saving the processed result over the
    original input file.

    :param row: List containing the bounding box coordinates used to define the region
        of interest in the format [x_min, y_min, x_max, y_max].
    :type row: list
    :param image_path: Path to the image file that will be cropped and scaled.
    :type image_path: str

    :raises FileNotFoundError: If the image file cannot be located using the given file path.
    """

    img_bgr = cv.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found at {image_path}.")

    x_min, y_min, x_max, y_max = get_bounding_box(row)
    cropped_img_bgr = img_bgr[y_min:y_max, x_min:x_max]
    scaled_img = cv.resize(cropped_img_bgr, (224,224), interpolation=cv.INTER_AREA)

    cv.cvtColor(scaled_img, cv.COLOR_BGR2RGB)
    cv.imwrite(image_path, scaled_img)

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
