"""
APIComm.py
Description: Controller for API communication to Kaggle for datasets. Download & Deletion management.

Contributors:
[
    Adam Long <adam.jacob.long@gmail.com>
]

License: MIT - ALL RIGHTS RESERVED
"""

#Imports
import os
import subprocess
from dotenv import load_dotenv

#Constants
DATASET_ID = 'jutrera/stanford-car-dataset-by-classes-folder'
DOWNLOAD_DIR = './data/stanford_cars'

load_dotenv()
subprocess_env = os.environ.copy()
subprocess_env['KAGGLE_USERNAME'] = os.getenv('KAGGLE_API_USER')
subprocess_env['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN')


def download_dataset():
    # Create the data directory if it doesn't exist'
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    else:
        print(f"Data directory '{DOWNLOAD_DIR}' already exists. Skipping download. If you think this is an error, delete the directory and try again.")
        exit(-1)

    # Construct the download command
    command_list = [
        'kaggle',
        'datasets',
        'download',
        '-d', DATASET_ID,
        '-p', DOWNLOAD_DIR,
        '--force'
    ]

    try:
        print("Starting download...")
        subprocess.run(
            command_list,
            shell=False, #Don't use shell as commands are injected via 'command_list'
            check=True,
            env=subprocess_env
        )
        print(f"Dataset {DATASET_ID} downloaded to {DOWNLOAD_DIR} successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")

    except FileNotFoundError:
        print("Error: 'kaggle' command not found. Ensure the Kaggle CLI is installed and in your PATH.")


def delete_dataset():
    if os.path.exists(DOWNLOAD_DIR):
        print(f"Deleting dataset directory '{DOWNLOAD_DIR}'...")
        os.rmdir(DOWNLOAD_DIR)
    else:
        print(f"Dataset directory '{DOWNLOAD_DIR}' does not exist. Skipping deletion.")