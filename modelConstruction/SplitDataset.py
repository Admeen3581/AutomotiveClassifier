"""
SplitDataset.py
Description: Parts the dataset as required by PyTorch.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
from torch.utils.data import Dataset
from modelConstruction.DatasetNormalization import *


# Import your function from the previous script (assuming it's in a file named data_utils.py)
# from data_utils import standardize_and_augment_image

# If it's in the same file, just ensure the function is defined above this class.

class CarPartDataset(Dataset):
    def __init__(self, datasheet, is_training=True):
        self.datasheet = datasheet#call the datasheet funcy wuncy
        self.is_training = is_training

    def __len__(self):
        return len(self.datasheet)

    def __getitem__(self, idx):
        #get row of datasheet as needed.
        row = self.datasheet.iloc[idx]
        #grabs the image path for the various image.
        image_path = f"./data/filtered_cars/train/{row[0]}"

        image_tensor = crop_dataset_image(self.datasheet, image_path)

        if self.is_training:
            image_tensor = augment_image(image_tensor)

        image_tensor = normalize_image(image_tensor)

        label = row[5]

        return image_tensor, torch.tensor(label, dtype=torch.long)