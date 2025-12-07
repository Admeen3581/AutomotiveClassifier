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
import shutil
import subprocess
import zipfile
from dotenv import load_dotenv

#Constants
DATASET_ID = 'jutrera/stanford-car-dataset-by-classes-folder'
DOWNLOAD_DIR = './data/stanford_cars'

load_dotenv()
subprocess_env = os.environ.copy()
subprocess_env['KAGGLE_USERNAME'] = os.getenv('KAGGLE_API_USER')
subprocess_env['KAGGLE_KEY'] = os.getenv('KAGGLE_API_TOKEN')

def download_dataset():
    """
    Downloads and extracts a Kaggle dataset to a specified directory.

    This function first checks if the download directory already exists. If it does, it prints a message
    and exits. Otherwise, it proceeds to download the dataset using the Kaggle CLI, extracts the contents
    of the downloaded zip file, and then removes the zip file to save disk space.

    Raises:
        subprocess.CalledProcessError: If the Kaggle CLI command fails.
        FileNotFoundError: If the 'kaggle' command is not found in the system's PATH.
    """

    # Create the data directory if it doesn't exist'
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)
    else:
        print(f"Data directory '{DOWNLOAD_DIR}' already exists. Skipping download. If you think this is an error, delete the directory and try again.")
        return

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

    #Zip constants
    zip_filename = DATASET_ID.split('/')[-1] + '.zip'
    zip_path = os.path.join(DOWNLOAD_DIR, zip_filename)

    if os.path.exists(zip_path):
        print(f"Unzipping data from {zip_filename}...")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DOWNLOAD_DIR)

        print(f"Extraction complete. Files are in: {DOWNLOAD_DIR}")

        # Delete the large ZIP file to save disk space
        os.remove(zip_path)
        print(f"Cleaned up the ZIP file.")
    else:
        print(f"Could not find the downloaded zip file at {zip_path}. Check for typos in the dataset ID.")


def delete_dataset():
    """
    Deletes the downloaded dataset directory and its contents.

    This function checks if the `DOWNLOAD_DIR` exists. If it does, it iterates through
    all files within the directory, removes them, and then removes the directory itself.
    If the directory does not exist, it prints a message indicating that deletion is skipped.
    """

    if os.path.exists(DOWNLOAD_DIR):
        print(f"Deleting dataset directory '{DOWNLOAD_DIR}'...")
        shutil.rmtree(DOWNLOAD_DIR)
        print(f"Deletion of dataset directory '{DOWNLOAD_DIR}' successful.")
    else:
        print(f"Dataset directory '{DOWNLOAD_DIR}' does not exist. Skipping deletion.")