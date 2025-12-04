import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def inspect_data(df):
    print("--- Data Structure ---")
    print(df.info())
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicates ---")
    print(f"Duplicates count: {df.duplicated().sum()}")

def detect_outliers(dataset, column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    up = Q3 + 1.5 * IQR

    outliers = dataset[(dataset[column] < low) | (dataset[column] > up)]
    return outliers, low, up

def clean_data(df):
    return df.drop(['CustomerId', 'Surname'], axis=1, errors='ignore').copy()

def feature_engineering(df):
    data = df.copy()
    
    data['Balance_to_Salary_Ratio'] = data['Balance'] / (data['EstimatedSalary'] + 1e-6)
    
    data['CreditScore_to_Age_Ratio'] = data['CreditScore'] / (data['Age'])
    
    data['Balance_per_Product'] = data['Balance'] / (data['NumOfProducts'])
    
    data['Balance_to_Salary_log'] = np.log1p(data['Balance'] / (data['EstimatedSalary'] + 1))
    
    data['Balance_Salary_category'] = pd.cut(
        data['Balance'] / (data['EstimatedSalary'] + 1),
        bins=[-0.1, 0, 0.5, 1.0, 2.0, np.inf],
        labels=['Zero', 'Low', 'Medium', 'High', 'VeryHigh']
    )
    
    return data

def split_data(df):
    data_encoded = pd.get_dummies(
        df, 
        columns=['Geography', 'Gender', 'Balance_Salary_category']
    ).astype('int')
    
    X = data_encoded.drop('Exited', axis=1)
    y = data_encoded['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    
    feature_names = X.columns
    
    return X_train_smote, X_test_scaled, y_train_smote, y_test, feature_names