import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import detect_outliers, save_plot

def plot_outliers_boxplots(data, columns=None, save_path='../images/'):
    if columns is None:
        columns = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
    
    for col in columns:
        outliers, low, up = detect_outliers(data, col)
        print(f'{col}:')
        print(f'Number of outliers: {outliers.shape[0]}')
        print(f'Lower bound: {low:.2f}, Upper bound: {up:.2f}')
        if outliers.shape[0] != 0:
            print(f'Sample outliers (first 3):')
            print(outliers[[col]].head(3))
        print('-' * 50)
    
    plt.figure(figsize=(12, 8))
    sns.catplot(
        kind='box',
        data=data[columns].melt(),
        y='value',
        col="variable",
        col_wrap=2,
        sharey=False,
        color='skyblue',
        width=0.5,
        height=4
    )
    plt.suptitle('Boxplots for Detecting Outliers', y=1.02)
    save_plot(plt.gcf(), 'Boxplots_outliers.png', path=save_path)
    plt.show()

def plot_churn_distribution(data, target_column='Exited', save_path='../images/'):
    exit_counter = data.groupby(target_column).size()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=exit_counter.index, y=exit_counter.values, 
                hue=exit_counter.index, width=0.3, palette='viridis')
    plt.title('Distribution of churn vs non-churn clients', fontsize=14, fontweight='bold')
    plt.xlabel('Exited (0 = stayed, 1 = churned)', fontsize=12)
    plt.ylabel('Number of clients', fontsize=12)
    plt.legend(title='Churn Status', labels=['Stayed', 'Churned'])
    plt.grid(axis='y', alpha=0.3)
    save_plot(plt.gcf(), 'Distribution_churn_vs_not_churn.png', path=save_path)
    plt.show()

def plot_numerical_distributions_by_churn(data, columns=None, target_column='Exited', save_path='../images/'):
    if columns is None:
        columns = ['Age', 'Balance', 'CreditScore']
    
    fig, axes = plt.subplots(1, len(columns), figsize=(5*len(columns), 5))
    
    if len(columns) == 1:
        axes = [axes]
    
    for idx, col in enumerate(columns):
        sns.histplot(data=data, x=col, hue=target_column, kde=True, 
                    palette='icefire', ax=axes[idx], alpha=0.7)
        axes[idx].set_title(f'Distribution of {col} by Churn Status', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].legend(title='Churn', labels=['Stayed', 'Churned'])
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'Numerical_values_histplots.png', path=save_path)
    plt.show()

def plot_categorical_distributions_by_churn(data, columns=None, target_column='Exited', save_path='../images/'):
    if columns is None:
        columns = ['NumOfProducts', 'IsActiveMember']
    
    fig, axes = plt.subplots(1, len(columns), figsize=(5*len(columns), 5))
    
    if len(columns) == 1:
        axes = [axes]
    
    if 'NumOfProducts' in columns:
        idx = columns.index('NumOfProducts')
        sns.countplot(data=data, x='NumOfProducts', hue=target_column, 
                     ax=axes[idx], palette='muted')
        axes[idx].set_title('Churned vs Non-Churned by Number of Products', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Number of products', fontsize=11)
        axes[idx].set_ylabel('Count', fontsize=11)
        axes[idx].legend(title='Churn', labels=['Stayed', 'Churned'])
        axes[idx].grid(axis='y', alpha=0.3)
    
    if 'IsActiveMember' in columns:
        idx = columns.index('IsActiveMember')
        sns.countplot(data=data, x='IsActiveMember', hue=target_column, 
                     ax=axes[idx], palette='muted', width=0.4)
        axes[idx].set_title('Churned vs Non-Churned by Activity Status', 
                           fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Is Active Member (0=No, 1=Yes)', fontsize=11)
        axes[idx].set_ylabel('Count', fontsize=11)
        axes[idx].legend(title='Churn', labels=['Stayed', 'Churned'])
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'Number_of_products_and_active_clients_countplots.png', path=save_path)
    plt.show()

def plot_geographical_distribution_by_churn(data, geography_column='Geography', 
                                          target_column='Exited', save_path='../images/'):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=geography_column, hue=target_column, 
                  palette='colorblind', width=0.5)
    plt.title('Churned vs Non-Churned Clients in Different Countries', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Countries', fontsize=12)
    plt.ylabel('Number of Clients', fontsize=12)
    plt.legend(title='Churn', labels=['Stayed', 'Churned'])
    plt.grid(axis='y', alpha=0.3)
    save_plot(plt.gcf(), 'Geographical_influence_countplots.png', path=save_path)
    plt.show()

def plot_gender_distribution_by_churn(data, gender_column='Gender', 
                                    target_column='Exited', save_path='../images/'):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x=gender_column, hue=target_column, 
                  palette='pastel', width=0.3)
    plt.title('Churn Distribution by Gender', fontsize=14, fontweight='bold')
    plt.xlabel('Gender', fontsize=12)
    plt.ylabel('Number of Clients', fontsize=12)
    plt.legend(title='Churn', labels=['Stayed', 'Churned'])
    plt.grid(axis='y', alpha=0.3)
    save_plot(plt.gcf(), 'Clients_by_gender_countplot.png', path=save_path)
    plt.show()

def plot_new_features_distributions(data, new_columns=None, target_column='Exited', save_path='../images/'):
    if new_columns is None:
        new_columns = ['Balance_to_Salary_log', 'CreditScore_to_Age_Ratio', 'Balance_per_Product']
    
    existing_columns = [col for col in new_columns if col in data.columns]
    
    if not existing_columns:
        print("No new features found in the dataset.")
        return
    
    fig, axes = plt.subplots(1, len(existing_columns), figsize=(5*len(existing_columns), 5))
    
    if len(existing_columns) == 1:
        axes = [axes]
    
    for idx, col in enumerate(existing_columns):
        bins = 30 if 'log' in col.lower() else None
        
        sns.histplot(
            data=data, 
            x=col, 
            hue=target_column, 
            kde=True, 
            palette='icefire', 
            ax=axes[idx],
            bins=bins,
            alpha=0.7
        )
        
        if 'log' in col.lower():
            title = 'Log(Balance/Salary) by Churn Status'
            xlabel = "Log(Balance to Salary Ratio)"
        elif 'CreditScore' in col:
            title = 'Credit Score to Age Ratio by Churn Status'
            xlabel = "Credit Score to Age Ratio"
        elif 'Balance_per' in col:
            title = 'Balance per Product by Churn Status'
            xlabel = "Balance per Product"
        else:
            title = f'{col} by Churn Status'
            xlabel = col
        
        axes[idx].set_title(title, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(xlabel, fontsize=11)
        axes[idx].set_ylabel("Count", fontsize=11)
        axes[idx].legend(title='Churn', labels=['Stayed', 'Churned'])
        axes[idx].grid(alpha=0.3)
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'New_Numerical_values_histplots.png', path=save_path)
    plt.show()

def plot_correlation_heatmap(data, save_path='../images/'):
    numeric_cols = data.select_dtypes(include=['int64', 'float64'])
    
    if numeric_cols.shape[1] < 2:
        print("Not enough numerical columns for correlation analysis.")
        return
    
    corr = numeric_cols.corr()
    
    plt.figure(figsize=(max(12, len(corr.columns)//2), max(10, len(corr.columns)//2)))
    
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                fmt='.2f', linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    
    plt.title('Correlation Heatmap (Numerical Features)', fontsize=14, fontweight='bold')
    save_plot(plt.gcf(), 'Numerical_values_heatmap.png', path=save_path)
    plt.show()

def run_complete_eda(data, target_column='Exited', save_path='../images/'):
    plot_outliers_boxplots(data, save_path=save_path)
    
    plot_churn_distribution(data, target_column, save_path=save_path)
    
    plot_numerical_distributions_by_churn(data, target_column=target_column, save_path=save_path)
    
    plot_categorical_distributions_by_churn(data, target_column=target_column, save_path=save_path)
    
    if 'Geography' in data.columns:
        plot_geographical_distribution_by_churn(data, target_column=target_column, save_path=save_path)
    
    if 'Gender' in data.columns:
        plot_gender_distribution_by_churn(data, target_column=target_column, save_path=save_path)
    
    new_features = ['Balance_to_Salary_log', 'CreditScore_to_Age_Ratio', 'Balance_per_Product']
    existing_new_features = [f for f in new_features if f in data.columns]
    if existing_new_features:
        plot_new_features_distributions(data, new_columns=existing_new_features, 
                                       target_column=target_column, save_path=save_path)
    
    plot_correlation_heatmap(data, save_path=save_path)