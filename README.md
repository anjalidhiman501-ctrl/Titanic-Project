# 🚢 Titanic Dataset - Data Preprocessing & Pipeline Report

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-1.x-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

This repository contains the complete **Data Preprocessing and Feature Engineering Pipeline** for the Titanic Dataset. The goal of this pipeline is to transform raw, noisy, and incomplete passenger data into a clean, numerical format ready for Machine Learning models like `RandomForestClassifier`.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [Preprocessing Workflow](#-preprocessing-workflow)
  - [1. Data Loading & Inspection](#1-data-loading--inspection)
  - [2. Handling Missing Values](#2-handling-missing-values)
  - [3. Feature Engineering & Encoding](#3-feature-engineering--encoding)
  - [4. Feature Selection & Cleanup](#4-feature-selection--cleanup)
  - [5. Data Splitting & Scaling](#5-data-splitting--scaling)
- [Summary of Transformations](#-summary-of-transformations)
- [Complete Python Pipeline Code](#-complete-python-pipeline-code)
- [How to Run](#-how-to-run)

---

## 🚀 Project Overview

The raw Titanic dataset contains missing entries, non-numeric columns, and noisy features. This pipeline ensures:
* **Completeness:** All missing values are handled effectively without losing essential records.
* **Compatibility:** All categorical features are converted to numeric representations.
* **Optimization:** High-cardinality/noisy columns are removed, and new family-related features are engineered.

---

## ⚙️ Preprocessing Workflow

### 1. Data Loading & Inspection
The dataset is loaded using Python's `pandas` library. An initial exploratory analysis is conducted to check the structural integrity, data types, and missingness:
* `df.head()` - View initial rows
* `df.shape` - Check dataset dimensions
* `df.info()` & `df.describe()` - Check missing values and statistical summary

### 2. Handling Missing Values
* **`Cabin`**: Dropped due to a high ratio of missing values (`df.drop("Cabin", axis=1)`).
* **`Name`**: Dropped as it does not contribute directly to model performance and risks overfitting.
* **`Age`**: Imputed missing values using **Mean Imputation** (`df['Age'].fillna(df['Age'].mean())`).
* **`Embarked`**: Imputed the 2 missing categorical values using **Backward Fill (`bfill`)**.

### 3. Feature Engineering & Encoding
* **Categorical Encoding:**
  * **`Sex`**: Binary mapped (`male`: `0`, `female`: `1`).
  * **`Embarked`**: Ordinal mapped (`C`: `0`, `S`: `1`, `Q`: `2`).
* **Feature Creation:**
  * Created a new feature **`Family`** by summing `SibSp` (siblings/spouses) and `Parch` (parents/children) to measure total family size on board.

### 4. Feature Selection & Cleanup
* **`Ticket`**: Dropped because of mixed data types and low predictive power, reducing noise in the training set.

### 5. Data Splitting & Scaling
The cleaned dataset is separated into target variable `y` (`Survived`) and features `X`. Then, it is split into an **80/20 Train-Test ratio** (`train_test_split`).

---

## 📊 Summary of Transformations

| Column Name | Initial State | Action Taken | Final State / Encoding |
| :--- | :--- | :--- | :--- |
| **`Cabin`** | High missing values | Dropped | Removed |
| **`Name`** | String / Text | Dropped | Removed |
| **`Ticket`** | Mixed data types | Dropped | Removed |
| **`Age`** | Missing numerical values | Mean Imputation | Continuous float |
| **`Embarked`**| Missing categorical | Backward fill (`bfill`) | Mapped (`C`:0, `S`:1, `Q`:2) |
| **`Sex`** | Categorical | Binary Encoding | `male`: 0, `female`: 1 |
| **`Family`** | *New Feature* | `SibSp` + `Parch` | Discrete integer |

---

## 💻 Complete Python Pipeline Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Data Loading & Inspection
df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

# 2. Handling Missing Values
df = df.drop("Cabin", axis=1)
df = df.drop("Name", axis=1)
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Embarked"].fillna(df["Embarked"].bfill(), inplace=True)

# 3. Feature Engineering & Encoding
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"C": 0, "S": 1, "Q": 2})
df["Family"] = df["SibSp"] + df["Parch"]

# 4. Feature Selection
df = df.drop("Ticket", axis=1)

# 5. Data Splitting
X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Preprocessing complete!")
print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
