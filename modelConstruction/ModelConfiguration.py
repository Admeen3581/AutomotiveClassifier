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
    """
    Retrieve and customize a pretrained ResNet101 model.

    The function initializes a ResNet101 model pretrained on a standard dataset
    (defined by `ResNet101_Weights.DEFAULT`). The model's parameters are frozen
    to prevent gradients from being computed during backward pass, except for
    the parameters of the fourth layer (`layer4`), which are unfrozen to allow
    fine-tuning. Additionally, the original fully connected (fc) layer is replaced
    with a custom sequential block of layers for a specific output configuration.

    :rtype: torch.nn.Module
    :return: A modified ResNet101 model with a custom fully connected layer.
    """

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
