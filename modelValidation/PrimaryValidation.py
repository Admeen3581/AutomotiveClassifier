"""
PrimaryValidation.py
Description: Validates the model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import pandas as pd
import torch.utils.data
import os
import tqdm

from modelValidation.BuildConfusionMatrix import build_matrix
from modelConstruction.ModelConfiguration import get_pretrained_model
from modelConstruction.DatasetNormalization import crop_dataset_image
from modelConstruction.PartDataset import CarPartDataset

#Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def validate_model(datasheet : pd.DataFrame, workers: int = 2, model_path: str = "./model/car_classifier.pt", show_matrix: bool = False):
    """
    Validates the performance of a pretrained machine learning model on a dataset.

    The function evaluates a given machine learning model on a dataset by calculating
    its accuracy. It accepts a dataset in the form of a pandas DataFrame and performs
    validation using a pre-defined model. A test DataLoader is created for loading
    the dataset in batches, and the model is evaluated for correctness by comparing
    predicted outputs with labeled data.

    :param datasheet: A pandas DataFrame containing the dataset for testing.
    :param workers: Number of parallel data loading workers. Defaults to 2.
    :param model_path: Path to the pre-saved model checkpoint. Defaults to
        "./model/car_classifier.pt".
    :return: None
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path at {model_path} not found.")

    test_dataset = CarPartDataset(datasheet, False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        persistent_workers=(workers > 0),
    )

    print("Validating model...")

    model = get_pretrained_model()
    model = model.to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)

    #Change PyTorch to evaluation mode
    model.eval()

    correct = 0;
    total = 0;
    all_preds = []
    all_labels = []

    #Disables training gradient calculation
    with torch.no_grad():
        progbar = tqdm.tqdm(test_loader, desc="Testing Model")

        for images, labels in progbar:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predict == labels).sum().item()

            all_preds.extend(predict.tolist())
            all_labels.extend(labels.tolist())
            accuracy = 100 * correct / total

    print(f"Test Accuracy: {accuracy:.2f}%")

    #Build & Show confusion matrix
    build_matrix(all_preds, all_labels)


def crop_test_image(datasheet : pd.DataFrame):
    # Pre-Crop all images in dataset
    for _, row in datasheet.iterrows():
        path = f"./data/filtered_cars/test/{row['image_path']}"
        crop_dataset_image(row, path)
    print("Dataset cropped successfully...")