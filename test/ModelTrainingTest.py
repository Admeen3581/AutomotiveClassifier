"""
ModelTrainingTest.py
Description: Test the training of the final layer of the model.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import pandas as pd
import os

def generate_short_test_csv():
    """
    Generates a short test CSV file for model training.
    """
    data = {
        'image_path': [
            '00001.jpg',
            '00002.jpg',
            '00003.jpg',
            '00004.jpg',
            '00005.jpg',
            '00007.jpg',
            '00008.jpg',
            '00009.jpg',
            '00010.jpg',
            '00011.jpg',
            '00012.jpg',
            '00013.jpg',
            '00014.jpg',
            '00015.jpg',
            '00017.jpg',
            '00018.jpg',
            '00019.jpg',
            '00020.jpg',
            '00021.jpg',
            '00022.jpg',
            '00023.jpg',
            '00024.jpg',
            '00025.jpg',
            '00026.jpg',
            '00028.jpg',
            '00030.jpg',
            '00032.jpg',
            '00033.jpg',
            '00034.jpg',
            '00036.jpg',
            '00037.jpg',
            '00038.jpg',
            '00040.jpg'
        ],
        'x_min': [39,36,85,621,14,88,73,20,21,51,6,30,31,32,39,3,247,17,17,212,11,53,34,30,45,82,12,34,37,138,31,16,214],
        'y_min': [116,116,109,393,36,80,79,126,110,93,62,36,246,77,52,8,287,281,156,538,28,126,87,174,115,109,31,197,63,41,87,89,110],
        'x_max': [569,868,601,1484,133,541,591,1269,623,601,499,418,778,589,233,190,1366,961,695,1893,476,973,567,598,585,874,471,1011,614,521,616,459,660],
        'y_max': [375,587,381,1096,99,397,410,771,367,393,286,307,540,379,150,147,761,596,375,1131,234,621,343,379,382,521,350,656,397,223,344,326,403],
        'class_id': [3,0,10,16,12,10,10,28,8,6,37,16,10,39,3,8,38,8,9,4,13,29,39,16,10,39,16,8,16,11,4,10,17]
    } #First 33 elements of data

    df = pd.DataFrame(data)

    output_dir = './data/'
    os.makedirs(output_dir, exist_ok=True) # Ensure the data directory exists
    output_path = os.path.join(output_dir, 'test_anno_train_filtered.csv')

    df.to_csv(output_path, index=False, header=False)
    print(f"Generated short test CSV file at: {output_path}")

#Test Execution
if __name__ == '__main__':
    generate_short_test_csv()

'''
1. **Run the script:** Execute `python generate_test_csv.py` from your project's root directory. 
This will create a file named `test_anno_train_filtered.csv` inside your `data` folder.
2. **Modify :`ModelTraining.py`** Update the function call in within to use this new file. 

csv_path="./data/test_anno_train_filtered.csv"


'''
