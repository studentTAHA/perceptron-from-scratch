# Perceptron From Scratch

A simple Perceptron built from scratch using only NumPy. It learns to tell whether a breast tumour is **benign** or **malignant** using the Breast Cancer Wisconsin (Diagnostic) dataset.

This project is my work for **ANN Lab Task 03** at SZABIST Islamabad (BS Artificial Intelligence).

---

## What this project does

- Loads the WDBC dataset and prepares it for training
- Splits the data into 80% training and 20% testing
- Standardizes the features using the training set only
- Builds a Perceptron from scratch (no scikit-learn model)
- Trains it for 100 epochs
- Measures accuracy, precision, recall, F1-score and the confusion matrix
- Plots the training error and the test confusion matrix

---

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic), also called WDBC
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic)
- **Samples:** 569
- **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness and more), calculated from images of cell nuclei
- **Target:** `diagnosis`, where `M` = Malignant (1) and `B` = Benign (0)
- **Class balance:** 357 benign and 212 malignant
- **Missing values:** none

The file is in `data/wdbc_with_header.csv`.

---

## How the Perceptron works

1. Multiply each feature by its weight and add the bias:
   `z = w · x + b`
2. Apply a step function. If `z >= 0` the prediction is 1, otherwise it is 0.
3. Compare the prediction with the real label.
4. Update the weights and bias:
   `w = w + lr × (y − y_hat) × x`
   `b = b + lr × (y − y_hat)`
5. Repeat for every sample, for every epoch.

**Settings used**

| Setting | Value |
|---|---|
| Learning rate | 0.01 |
| Epochs | 100 |
| Weight initialisation | Small random numbers (seed 0) |
| Train / test split | 80% / 20% (seed 42) |

---

## Results

| Set | Accuracy | Precision | Recall | F1-score |
|---|---|---|---|---|
| Train | 0.9824 | 0.9709 | 0.9824 | 0.9766 |
| Test | 0.9737 | 1.0000 | 0.9286 | 0.9630 |

**Test confusion matrix**

| | Predicted 0 | Predicted 1 |
|---|---|---|
| **Actual 0** | 72 | 0 |
| **Actual 1** | 3 | 39 |

![Results](images/results.png)

**What the numbers tell us**

- The model reaches about **97% accuracy** on data it has never seen.
- It made **no false alarms** (precision is 1.0). Every tumour it called malignant really was malignant.
- It missed **3 malignant cases** (recall is 0.93). In a medical problem, missing a real case is the more serious mistake, so recall is the number to watch.
- The training error goes up and down between epochs instead of settling at zero. This is normal for a Perceptron because it only updates on mistakes and the data is not perfectly separable.

---

## Project structure

```
perceptron-from-scratch/
├── README.md
├── perceptron_wdbc.ipynb
├── requirements.txt
├── data/
│   └── wdbc_with_header.csv
└── images/
    └── results.png
```

---

## How to run it

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR-USERNAME/perceptron-from-scratch.git
   cd perceptron-from-scratch
   ```
2. Install the requirements
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook
   ```bash
   jupyter notebook perceptron_wdbc.ipynb
   ```
4. Run all the cells from top to bottom.

---

## Tools used

- Python 3
- NumPy
- pandas
- Matplotlib
- Jupyter Notebook

---

## Author

**TAHA**
BS Artificial Intelligence, SZABIST Islamabad
Course: Artificial Neural Networks Lab
