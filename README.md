# Bachelorarbeit Project
### Finding Patterns in Variability: Clustering Skin Conductance Responses to Identify Psychopathological Patterns

## Project Overview

This project is part of a Bachelor thesis. It aims to analyze and compare physiological and psychological data using k-Means clustering. 

## Directory Structure

```plaintext
root/
│
├── bin/                     
│   ├── elbow_method.py
│   ├── kmeans_model.py
│   ├── kmeans_model_psych.py
│   └── prep/
│       ├── prep_physiological.py
│       └── prep_psychological.py
│
├── data/
│   ├── scr_data.csv
│   ├── psychopathological_data.csv # Psychological data
│   └── other CSV files
│
└── plots/

```

## Python Scripts

### 1. Data Preparation
- `prep_physiological.py`: Preprocesses the physiological data (e.g., SCR).
- `prep_psychological.py`: Preprocesses the psychological data.

### 2. Elbow Method (`elbow_method.py`)
This script implements the elbow method to help decide the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters.

### 3. K-Means Models
- `kmeans_model.py`: Runs K-Means clustering on physiological and psychological data.
- `kmeans_model_psych.py`: Runs K-Means clustering specifically for psychological data.

### 4. Comparison of clusters
- `compare_clusters.py`: Runs comparisons between physiological and psychological clusters.


## Data

The `data` folder contains several CSV files:
- `scr_data.csv`: Contains skin conductance response (SCR) data.
- `psychopathological_data.csv`: Contains psychological data relevant to the study.

## Dependencies

You can install the required dependencies by running:

```bash
pip install pandas scikit-learn scipy matplotlib seaborn numpy
```

## Usage
### 1.	Preprocess the data:

Run the preprocessing scripts for both physiological and psychological data:

```bash
python bin/prep/prep_physiological.py
python bin/prep/prep_psychological.py
```

### 2. Run clustering analysis:

Use the elbow method to determine the optimal number of clusters:

```bash
python bin/elbow_method.py
```

Then, run K-Means clustering:
```bash
python bin/kmeans_model.py
python bin/kmeans_model_psych.py
```

### 3. Compare clusters
```bash
python bin/compare_clusters.py
```

## Author

Luísa Hörlle de Castro

## License

This project is licensed under the MIT License.