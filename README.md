# Random Forest Algorithm

## Random Forest Classifier: Breast Cancer Diagnosis

This project demonstrates the use of a **Random Forest Classifier** to predict whether a tumor is benign or malignant using the **Breast Cancer Wisconsin (Diagnostic)** dataset. The goal is to build a simple and efficient classification model using `scikit-learn`.

## Overview

The notebook performs the following steps:
- Loads and preprocesses the dataset  
- Encodes the target variable (`diagnosis`)  
- Splits the data into training and test sets  
- Trains a `RandomForestClassifier`  
- Saves the trained model using `pickle` for future use  

## Dataset

The dataset used is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)).  
It includes 30 numerical features computed from digitized images of fine needle aspirate (FNA) of breast masses.

## Requirements

- Python 3.x  
- pandas  
- scikit-learn  
- pickle (standard library)

Install dependencies using:

```bash
pip install pandas scikit-learn
```

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Open the notebook:
   ```bash
   jupyter notebook random_forest_algorithm.ipynb
   ```

3. Run all cells to:
   - Train the model
   - Save the model as `rf_model.pkl`

## Output

- Trained model file: `rf_model.pkl`
- You can later load the model with:

```python
import pickle

with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)
```
