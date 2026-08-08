
# 🚢 Titanic Dataset - Data Preprocessing Pipeline

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
```python
import pandas as pd

df = pd.read_csv("titanic.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())

