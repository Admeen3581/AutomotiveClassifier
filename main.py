"""
main.py
Description: Hello There :)

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
from modelValidation.PrimaryValidation import *
from modelConstruction.DatasetInitialization import *
from modelConstruction.ModelTraining import *

#Constants
MODEL_PATH = "./model/car_classifier.pt"

if __name__ == '__main__':
    print("Hello There :)\nVehicular Classifier Ver.1.1\n\n\t---Initializing---\n\n")

    dataset_init()

    if not os.path.exists(MODEL_PATH):
        #Best model was trained via AWS: Nvidia L40S GPU w/ 8 CPU cores.
        #25 epochs over 4 learning rate chunks off ResNet101 (see ModelTraining.py).
        train_model(get_datasheet(), 4)
    else:
        print("Model detected. Skipping training...")

    validate_model(get_datasheet("./data/anno_test_filtered.csv"), 4, MODEL_PATH)
