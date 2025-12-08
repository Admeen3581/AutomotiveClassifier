"""
ModelTraining.py
Description: Train the final layer of the model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import torch
import torch.optim as optim
import pandas as pd
import torch.utils.data

from modelConstruction.SplitDataset import CarPartDataset
from modelConstruction.ModelConfiguration import get_pretrained_model
from controllers.CarMakeData import car_brands

#Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_CLASSES = len(car_brands)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():

    datasheet = get_datasheet()#default is training datasheet

    train_dataset = CarPartDataset(datasheet, True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    #Initialize Model
    model = get_pretrained_model()
    model = model.to(DEVICE)

    #Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            #Transfer CPU/GPU data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            #Forward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #Stats
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_accuracy = 100 * correct / total
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Acc: {epoch_accuracy:.2f}%")

    print("\n\t---Training Complete---\n")


def get_datasheet(csv_path: str = "./data/anno_train_filtered.csv"):
    datasheet = pd.read_csv(csv_path, header=None)
    datasheet.columns = ['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
    return datasheet