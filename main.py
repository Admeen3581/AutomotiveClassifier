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
from modelConstruction.DatasetInitialization import *
from modelConstruction.ModelTraining import *

if __name__ == '__main__':
    print("Hello World")

    dataset_init()

    train_model(get_datasheet(), 4)
