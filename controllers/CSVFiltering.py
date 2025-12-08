"""
CSVFiltering.py
Description: Cleans the Kaggle CSV files into something usable.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""
import os

#Imports
import pandas as pd
from controllers.CarMakeData import car_brands as ALLOWED_MAKES


def process_csv():
    """
    Process a CSV file by filtering and mapping entries using a predefined list of allowed brands.

    This function reads a CSV file containing names, filters the entries based on their match
    with the allowed list of brands (ALLOWED_MAKES), creates a mapping of original IDs to new IDs,
    and then processes annotations for both training and testing datasets.

    :raises FileNotFoundError: If any of the specified files are not found.
    :raises KeyError: If there is an issue accessing expected keys in the data.
    """

    brand_to_id = {brand: i for i, brand in enumerate(ALLOWED_MAKES)}

    datasheet = pd.read_csv("./data/names.csv", header=None, names=['name'])

    #Create mapping object
    original_id_to_new_id = {}

    for index, row in datasheet.iterrows():
        original_id = index + 1
        full_name = row['name']

        #Is it an allowed brand?
        brand_match = [brand for brand in ALLOWED_MAKES if full_name.startswith(brand)]

        if brand_match:
            best_match = min(brand_match, key=len)
            original_id_to_new_id[original_id] = brand_to_id[best_match]

    process_annotations("./data/anno_train.csv", "./data/anno_train_filtered.csv", original_id_to_new_id)
    process_annotations("./data/anno_test.csv", "./data/anno_test_filtered.csv", original_id_to_new_id)

    #Clean Names.CSV
    with open("./data/names_filtered.csv", "w") as f:
        for brand in ALLOWED_MAKES:
            f.write(brand + "\n")

    delete_old_csv()

def process_annotations(input_file, output_file, mapping):
    """
    Processes annotation data from an input CSV file, filters rows based on a dataset class
    mapping, updates the class IDs using the provided mapping, and saves the results.

    :param input_file: Input CSV file containing annotation data. Data is expected to be
        in a format with no headers and columns corresponding to filename, x1, y1,
        x2, y2, and class_id.
    :type input_file: str
    :param output_file: Output CSV file path where the cleaned and updated annotation
        data will be saved.
    :type output_file: str
    :param mapping: A dictionary mapping old class IDs to new class IDs. Rows with
        class IDs not in the mapping will be excluded from the output.
    :type mapping: dict
    :return: None
    """

    datasheet = pd.read_csv(input_file, header=None)

    # Column 5 is the Class ID
    # Filter: Keep rows where Class ID is in our mapping
    datasheet_clean = datasheet[datasheet[5].isin(mapping.keys())].copy()

    datasheet_clean[5] = datasheet_clean[5].map(mapping)

    datasheet_clean.to_csv(output_file, index=False, header=False)

def delete_old_csv():
    for file in ["anno_train.csv", "anno_test.csv", "names.csv"]:
        os.remove(f"./data/{file}")