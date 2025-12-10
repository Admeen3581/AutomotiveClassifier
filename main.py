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

if __name__ == '__main__':
    print("Hello World")

    dataset_init()


    if not os.path.exists("./model/car_classifier(best).pt"):
        #Best model was trained via AWS: Nvidia L40S GPU w/ 8 CPU cores.
        #25 epochs over 4 learning rate chunks off ResNet101 (see ModelTraining.py).
        train_model(get_datasheet(), 4)

    validate_model(get_datasheet("./data/anno_test_filtered.csv"))
