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


def get_datasheet(csv_path: str = "./data/anno_train_filtered.csv"):
    datasheet = pd.read_csv(csv_path, header=None)
    datasheet.columns = ['image_path', 'x_min', 'y_min', 'x_max', 'y_max', 'class_id']
    return datasheet

if __name__ == '__main__':
    print("Hello World")

    dataset_init(get_datasheet())

    train_model(get_datasheet())
