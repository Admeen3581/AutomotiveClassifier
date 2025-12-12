# AutomotiveClassifier
====================

Computer Vision project to classify vehicle makes (e.g., Toyota, Subaru, Ford) using a transfer‑learned ResNet‑101. The repository automates dataset download from Kaggle, dataset filtering/restructuring, pre‑cropping to bounding boxes, training, and validation with a saved confusion matrix.


## Contents
--------
- What this project does
- Repo structure
- Prerequisites
- Installation (Poetry or pip/venv)
- Configure Kaggle API (.env)
- Quickstart: download data, preprocess, train, and validate
- Outputs and artifacts
- Troubleshooting
- FAQ
- Contributing
- License


## Before you start (read me if you’re new!)
----------------------------------------
If you’re not comfortable with Python tools yet, follow these tips first:

- You need Python 3.11–3.14 installed.
  - Check your version (Windows PowerShell):
    ```powershell
    python --version
    ```
  - Check your version (macOS/Linux):
    ```bash
    python3 --version
    ```

- You need Git to clone this repository.
  - Check Git:
    ```powershell
    git --version
    ```
    ```bash
    git --version
    ```

- How to open a terminal:
  - Windows: Press Win key, type “PowerShell”, press Enter.
  - macOS: Open “Terminal” app.
  - Linux: Open your distribution’s terminal app.

- Where to run commands: First navigate into the folder where you want the project. Example:
  - Windows (PowerShell):
    ```powershell
    cd $HOME\Documents
    ```
  - macOS/Linux:
    ```bash
    cd "$HOME/Documents"
    ```

- Copy exactly what is inside each code box as a single line. Every command below is split so you can copy one line at a time.


## What this project does
----------------------
At a high level:
1. Downloads the “Stanford Car Dataset by classes folder” from Kaggle.
2. Filters/reorganizes the dataset into `./data/filtered_cars/{train,test}` based on supported makes in `controllers/CarMakeData.py`.
3. Moves and filters Kaggle CSVs to generate annotation sheets (e.g., `anno_train_filtered.csv`, `anno_test_filtered.csv`).
4. Pre‑crops all images to their bounding boxes and resizes them to 224×224.
5. Builds a ResNet‑101 model with a custom classifier head; freezes all base layers except `layer4` and trains with a scheduled learning rate.
6. Saves the trained model to `./model/car_classifier.pt` and evaluates on the test set.
7. Produces a normalized confusion matrix image saved to `./output/ConfusionMatrix.png`.


## Repo structure
--------------
Top‑level overview (non‑exhaustive):

- `main.py` — Orchestrates dataset initialization, conditional training, and validation.
- `controllers/` — Dataset filtering, Kaggle API communication, metadata for makes/classes.
- `modelConstruction/` — Dataset init, normalization/augmentation, dataset class, model config, training loop.
- `modelValidation/` — Validation and confusion matrix builder.
- `data/` — Created at runtime; holds downloaded and filtered data and CSVs.
- `model/` — Created at runtime; Trained weights (`car_classifier.pt`). If you want to skip training, place a compatible weights file here.
- `output/` — Created at runtime; Confusion matrix and other outputs.
- `pyproject.toml` — Project metadata and dependencies (Poetry).
- `.env` — Your Kaggle API credentials.


## Prerequisites
-------------
Software
- Python 3.11–3.14 (project targets >=3.11, <3.15).
- Git (to clone the repository).
- One of:
  - Poetry (recommended), or
  - pip + venv.
- Kaggle account and API credentials (see “Configure Kaggle API”).
- Kaggle CLI available in your shell. Installing the `kaggle` Python package provides the `kaggle` command but you may need to re‑open your shell so PATH updates take effect.

Hardware
- CPU‑only works (***extremely slow***); GPU (NVIDIA) recommended for performance. Model was trained on an Nvidia L40S -> ~1hrs on GPU.
- If using GPU, install an appropriate NVIDIA driver. Prebuilt PyTorch usually bundle CUDA runtime.

Disk and bandwidth
- Dataset download, repository files, extraction, and other files require ~4GB gigabytes free. ***Ensure you have 4GB of space free on your disk before cloning.***


## Installation
------------

### Option A — pip w/ Poetry
1. Clone the repository.
   - Windows (PowerShell):
     ```powershell
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
   - macOS/Linux (Bash):
     ```bash
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
2. Install Poetry:
   - Install/Upgrade pip first:
     ```bash
     python -m pip install --upgrade pip
     ```
   - Install Poetry:
     ```bash
     pip install poetry
     ```
3. Install project dependencies from pyproject using Poetry:
   - ```bash
     poetry install
     ```
4. Build your .env file in the *root* directory:
   - ```env
     KAGGLE_API_USER=<your_kaggle_username>
     ```
   - ```env
     KAGGLE_API_TOKEN=<your_kaggle_api_key>
     ```
5. Optional: Use Poetry to run the app:
   - ```bash
     poetry run python main.py
     ```

### Option B — No Poetry (pip exclusively; not recommended)
1. Clone the repository.
   - Windows (PowerShell):
     ```powershell
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
   - macOS/Linux (Bash):
     ```bash
     git clone https://github.com/Admeen3581/AutomotiveClassifier.git
     cd AutomotiveClassifier
     ```
2. Install Poetry:
   - Upgrade pip first:
     ```bash
     python -m pip install --upgrade pip
     ```
   - Install Poetry:
     ```bash
     pip install requests kaggle python-dotenv opencv-python torch torchvision numpy pandas matplotlib scikit-learn tqdm
     ```
3. Build your .env file in the *root* directory:
   - ```env
     KAGGLE_API_USER=<your_kaggle_username>
     ```
   - ```env
     KAGGLE_API_TOKEN=<your_kaggle_api_key>
     ```
4. Optional: Run the app:
   - ```bash
     python main.py
     ```


## Configure Kaggle API (.env)
---------------------------
The dataset is downloaded via the Kaggle CLI and requires API credentials.

1. Create a Kaggle account (https://www.kaggle.com/).
2. Generate an API token: Account settings → Create New API Token. This downloads `kaggle.json` containing `username` and `key`.
3. In this project’s root directory, create a file named `.env` with these two lines (each line is separate):
   - ```env
     KAGGLE_API_USER=<your_kaggle_username>
     ```
   - ```env
     KAGGLE_API_TOKEN=<your_kaggle_api_key>
     ```
4. Ensure the `kaggle` command is available in your shell. If you just installed it, close and reopen your terminal.
5. You must accept the dataset’s terms on Kaggle to download it. Visit the dataset page and click “I Understand and Accept” if prompted:
   - https://www.kaggle.com/datasets/jutrera/stanford-car-dataset-by-classes-folder


## Quickstart
----------
Run the end‑to‑end pipeline. On first run, this will:
- Create `./data/` if it does not exist.
- Download and extract the Kaggle dataset into `./data/stanford_cars`.
- Reorganize and filter images into `./data/filtered_cars/{train,test}`.
- Move and filter CSVs, then pre‑crop training images to bounding boxes.
- Train the model if `./model/car_classifier.pt` does not exist.
- Validate on the test set and save `./output/ConfusionMatrix.png`.

Using Poetry:
```bash
poetry run python main.py
```

Using pip/venv:
```bash
python main.py
```

Notes
- If `./data` already exists, dataset initialization will print a message and skip downloading. Delete `./data` to force a fresh re‑init.
- Training uses the following defaults (see `modelConstruction/ModelTraining.py`):
  - Batch size: 512
  - Epochs per schedule chunk: 25
  - Schedule chunks: 4 with learning rates 1e-2, 1e-3, 1e-4, 1e-5
  - Device: CUDA if available, else CPU


## Outputs and artifacts
---------------------
- `./model/car_classifier.pt` — Saved PyTorch weights after training.
- `./output/ConfusionMatrix.png` — Normalized confusion matrix over the test set.
- `./data/filtered_cars/` — Filtered and preprocessed dataset layout used by the model.


## Troubleshooting
---------------
- “kaggle: command not found” or Kaggle CLI not recognized
  - Ensure the environment has Kaggle installed (it is declared in pyproject and installed by Poetry). Re‑open your terminal so the `kaggle` command is on PATH.
  - Confirm you accepted the dataset terms on Kaggle.
  - Ensure `.env` contains `KAGGLE_API_USER` and `KAGGLE_API_TOKEN`.

- Dataset init keeps skipping with a message that `./data` exists
  - Delete the `./data` directory to force a fresh init. The initializer returns early if it detects an existing data directory.
 
- `FileNotFoundError: Could not find CSV file at ./data/anno_test_filtered.csv. Possible a dataset API call issue.`
  - - Delete the `./data` directory to force a fresh init.

- CUDA/CuDNN errors
  - If you don’t need GPU acceleration, run on CPU (it will auto‑detect). If you do, ensure your NVIDIA driver is up to date and that your Torch install matches your CUDA runtime. Installing the default pip/Poetry wheel typically includes a compatible CUDA runtime.

- OpenCV cannot read images (`Image not found`)
  - Ensure the dataset successfully downloaded and reorganized and that your working directory is the project root when running `main.py`.

- `TypeError: expected str, bytes or os.PathLike object, not NoneType`
  - Ensure you .env file is setup correctly.


## FAQ
---
- Which dataset do we use?
  - Kaggle: “Stanford Car Dataset by classes folder” (jutrera/stanford-car-dataset-by-classes-folder).

- What classes are supported?
  - The list is defined in `controllers/CarMakeData.py` as `car_brands`. This drives the number of output classes and dataset filtering.

- Can I skip training and just validate?
  - Yes, place a compatible weights file at `./model/car_classifier.pt` before running `main.py`. The script will detect the model and skip training.

- How do I re‑run preprocessing?
  - Delete the `./data` directory and run `main.py` again.


## Contributing
------------
Pull requests are welcome. Please keep to the existing code style and structure. If adding dependencies, update `pyproject.toml` and ensure installation works with both Poetry and pip/venv. Consider adding or updating tests under `test/` as appropriate.


## License
-------
This project is licensed under the MIT License. See `LICENSE.md` for details.
