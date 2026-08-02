
 Preprocessing Pipeline Report

This section documents the data preprocessing steps performed on the Titanic dataset.
1. Initial Data Loading and Inspection:
*   The dataset was loaded use Pandas library.
*   An initial overview of the data was taken using `df.head()`, `df.shape`, `df.describe()`, and `df.info()`. This step help understand the dataset structure, Handing missing values, and find data types, which were crucial for subsequent preprocessing steps.

 2. Handling Missing Values:
*   Cabin Column: This column had a large number of missing values, so it was drop the column (`df=df.drop("Cabin",axis=1)`). This prevented the model from using incomplete information.
*   Name Column: This column was not useful for model training ,so it was also drop(`df=df.drop("Name",axis=1)`). This reduced the model complexity and risk of overfitting.
*   Age Column: Missing 'Age' values were fill use the mean of the column (`df['Age']=df['Age'].fillna(df['Age'].mean())`). Mean imputation is a simple method to handle missing numerical data.
*   Embarked Column: There were only 2 missing values in this column, which were filled using the `bfill()` method (backward fill) (`df['Embarked'].fillna(df['Embarked'].bfill(), inplace=True)`). This is an effective method for filling a small number of missing categorical values.

 3. Feature Engineering and Encoding:
*   Sex Column: The categorical 'Sex' column ('male', 'female') was converted into numerical values ('male': 0, 'female': 1).ML models work with numerical data, making it necessary to encode categorical features.
*   Embarked Column:The categorical 'Embarked' column ('C', 'S', 'Q') was converted into numerical values also('C': 0, 'S': 1, 'Q': 2) . While one-hot encoding also.
*   Famliy Feature Creation: A new feature named 'Famliy' was created by summing the 'SibSp' (siblings/spouses) and 'Parch' (parents/children) columns, representing the total count of family members . This feature was engineered to capture the impact of family size on survival prediction.

4. Feature Selection:
*   Ticket Column: This column was not useful because mix data types, so it was removed  its use for reduces noise.

5. Data Splitting:
*   The dataset was split into features ('X') and the target variable ('y'), where 'y' is the 'Survived' column.
*   The data was split into training ('X_train', 'y_train') and testing (X_test, y_test) sets using train_test_split (80% training, 20% testing).

6. Feature Scaling:
*   
 Model Performance:
All the preprocessing steps mention above transform the data into a suitable format for the RandomForestClassifier model. Handling missing values, encoding categorical features, and creating a new 'Famliy' feature all helped the model learn meaningful patterns from the data. 
