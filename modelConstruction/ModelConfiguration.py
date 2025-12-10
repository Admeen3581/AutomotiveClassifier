"""
ModelConfiguration.py
Description: Pre-train and build the model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import torch.nn as nn
import torchvision.models as models
from controllers.CarMakeData import car_brands


def get_pretrained_model():

    #Using ResNet18 as the pretrained model.
    model = models.resnet18(weights=models.ResNet50_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    #Replace final FC layer with CarMakeData.py.length number of outputs.
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(car_brands))

    return model
