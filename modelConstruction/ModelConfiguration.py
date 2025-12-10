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

    #Using ResNet101 as the pretrained model.
    model = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

    for param in model.parameters():
        param.requires_grad = False

    #Unfreeze layers allowing for more fine tuning options.
    for param in model.layer4.parameters():
        param.requires_grad = True

    #Add a custom layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5),
        nn.Linear(256, 40),

    )

    return model
