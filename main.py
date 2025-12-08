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
from controllers.APIComm import *
from controllers.DatasetFiltering import *
from modelConstruction.ModelTraining import *


if __name__ == '__main__':
    print("Hello World")

    download_dataset()
    reorganize_dataset()
    delete_dataset()

    train_model()
