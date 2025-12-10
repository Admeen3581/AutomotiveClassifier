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
import os

from modelConstruction.SplitDataset import CarPartDataset
from modelConstruction.ModelConfiguration import get_pretrained_model
from controllers.CarMakeData import car_brands
from tqdm import tqdm

#Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.01
NUM_EPOCHS = 20
NUM_CLASSES = len(car_brands)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_EXPORT_PATH = "./car_classifier.pt"

def train_model(datasheet: pd.DataFrame, workers: int = 2):

    train_dataset = CarPartDataset(datasheet, True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=workers,
        persistent_workers=True,
        pin_memory=True)

    #Initialize Model
    model = get_pretrained_model()
    model = model.to(DEVICE)

    #Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    print("\n\t---Training Started---\n")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}\n-------------------------------")
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=True)

        for images, labels in loop:
            #Transfer CPU/GPU data
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            #Forward Pass
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16):
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
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Loss: {running_loss / len(train_loader):.4f} | Accuracy: {epoch_accuracy:.2f}%")

    print("\n\t---Training Complete---\n")

    os.makedirs(os.path.dirname(MODEL_EXPORT_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_EXPORT_PATH)
    print(f"Model saved to {MODEL_EXPORT_PATH}")