"""
PartDataset.py
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


class CarPartDataset(Dataset):
    def __init__(self, datasheet, is_training=True):
        self.is_training = is_training

        self.datasheet_samples = []

        if is_training:
            dataset_type= 'train'
        else:
            dataset_type= 'test'

        for _, row in datasheet.iterrows():
            path = f"./data/filtered_cars/{dataset_type}/{row['image_path']}"
            label = row['class_id']
            self.datasheet_samples.append((path, label))

    def __len__(self):
        return len(self.datasheet_samples)

    def __getitem__(self, idx):
        #get row of datasheet as needed.
        image_path, label = self.datasheet_samples[idx]

        img = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
        image_tensor = transforms.ToTensor()(img)

        if self.is_training:
            image_tensor = augment_image(image_tensor)

        image_tensor = normalize_image(image_tensor)

        return image_tensor, torch.tensor(label, dtype=torch.long)