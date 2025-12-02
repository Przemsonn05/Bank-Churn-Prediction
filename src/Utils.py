import pandas as pd

def detect_outliers(dataset, column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1

    low = Q1 - 1.5 * IQR
    up = Q3 + 1.5 * IQR

    outliers = dataset[(dataset[column] < low) | (dataset[column] > up)]
    return outliers, low, up
