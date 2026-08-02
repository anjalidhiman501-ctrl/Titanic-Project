# Titanic Survival Prediction Project

Project Overview
This project focuses on predicting the survival of passengers on the Titanic using various machine learning techniques. 

Dataset Source
The dataset used in this project is the famous Titanic dataset, which contains information about passengers aboard the RMS Titanic, including their demographics, travel class, and survival status.

Source: The dataset is titanic dataset.csv
Preprocessing Workflow
The data preprocessing steps were crucial for cleaning and preparing the raw data for machine learning model training. The detailed steps are as follows:

1.  **Initial Data Loading and Inspection:**
    *   Loaded the dataset into a pandas DataFrame.
    *   Performed initial exploration using `head()`, `shape`, `describe()`, and `info()` to understand the data's structure, missing values, and data types.

2.  **Handling Missing Values:**
    *   Dropped the 'Cabin' column due to a high number of missing values.
    *   Dropped the 'Name' column as it was not useful for predictive modeling.
    *   Filled missing 'Age' values with the mean of the 'Age' column.
    *   Filled missing 'Embarked' values using the backward fill method.

3.  **Feature Engineering and Encoding:**
    *   Converted the 'Sex' column from categorical ('male', 'female') to numerical (0, 1).
    *   Converted the 'Embarked' column from categorical ('C', 'S', 'Q') to numerical (0, 1, 2).
    *   Created a new feature 'Famliy' by summing 'SibSp' (siblings/spouses) and 'Parch' (parents/children) to represent family size.

4.  **Feature Selection:**
    *   Removed the 'Ticket' column as it was deemed not useful for modeling due to its unique identifiers and mixed data types.

5.  **Data Splitting:**
    *   Split the dataset into features (`X`) and the target variable (`y`, which is 'Survived').
    *   Further split the data into training (80%) and testing (20%) sets using `train_test_split`.

6.  **Feature Scaling:**
    *   Applied `StandardScaler` to numerical features to standardize their range. (Note: It was applied to the full `X` dataset, but ideally, it should be fit only on `X_train` to avoid data leakage).

## Model Training and Evaluation
*   A `RandomForestClassifier` was trained on the preprocessed training data.
*   The model achieved an accuracy of approximately **82.68%** on the test set.
*   A detailed classification report was generated, providing precision, recall, and F1-score for both classes (Died and Survived).

## Visualizations
Histograms showcasing the data distribution of 'Age' and 'Fare' before and after `StandardScaler` application are available in the notebook, demonstrating the effect of scaling on feature distributions.

## Outputs
*   **Cleaned Dataset:** The final preprocessed dataset is saved as `cleaned_titanic_dataset.csv`.
*   **Preprocessing Report:** A detailed markdown report explaining the preprocessing pipeline and its impact on model performance is available at `preprocessing_report.md`.

