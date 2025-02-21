# Tissue Insight
- TissueInsight is a comprehensive pipeline designed for downloading, preprocessing, and modeling spatial transcriptomics data. The goal of this project is to build a complete workflow for analyzing and visualizing tissue gene expression data.

# Collaborators:
> Amro Rabea
> Omar Hazem
> Aya Sherif
> Amira Sherif
> Omar Hassan

## Project Structure
```
TissueInsight/
│── data/                  # Directory for storing downloaded datasets
│   ├── .gitkeep           # Keeps the data folder in version control
│
│── notebooks/             # Jupyter notebooks for exploration and testing
│   ├── download_data.ipynb # Notebook for downloading the dataset
│
│── src/                   # Source code for the pipeline
│   ├── __init__.py        # Makes src a package
│   ├── download/          # Module for dataset downloading
│   │   ├── __init__.py
│   │   ├── downloader.py  # Functions for downloading dataset
│
│── .gitignore             # Files to ignore in version control
│── LICENSE                # Project license
│── README.md              # Project documentation
│── requirements.txt       # Dependencies required to run the project
```

## Installation

1. Clone the Repository:
```bash
$ git clone https://github.com/amrorabea/TissueInsight.git
$ cd TissueInsight
```

2. Create a Virtual Environment (Optional but Recommended):
```bash
$ python -m venv venv
$ source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install Dependencies:
```bash
$ pip install -r requirements.txt
```


