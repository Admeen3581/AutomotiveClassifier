"""
BuildConfusionMatrix.py
Description: Train the final layer of the model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import os
import matplotlib.pyplot as plt

from controllers.CarMakeData import car_brands
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#Constants
NUM_CLASSES = len(car_brands)

def build_matrix(all_preds: list, all_labels: list, show_plot: bool = True):


    matrix = confusion_matrix(all_labels, all_preds, normalize='true', labels=list(range(NUM_CLASSES)))

    plt.figure(figsize=(24, 24))#hardcoded value that fits best

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=car_brands)
    display.plot(
        cmap=plt.cm.get_cmap('Blues', NUM_CLASSES),
        xticks_rotation='vertical',
        values_format='.2f'
    )

    axis = display.ax_
    axis.tick_params(axis='both', which='major', labelsize=4)
    for text in axis.texts:
        text.set_fontsize(2)

    plt.title("Confusion Matrix (VMC Test Dataset)").set_fontsize(7)
    plt.xlabel("Predicted Label").set_fontsize(5)
    plt.ylabel("True Label").set_fontsize(5)

    #Save matrix
    save_path = os.path.join("./output", "ConfusionMatrix.png")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()

    print(f"Confusion Matrix Saved to {save_path}")