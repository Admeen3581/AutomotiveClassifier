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

    ds = get_datasheet()

    if not os.path.exists("./model/car_classifier(best).pt"):
        train_model(ds, 4)

    validate_model(ds)
