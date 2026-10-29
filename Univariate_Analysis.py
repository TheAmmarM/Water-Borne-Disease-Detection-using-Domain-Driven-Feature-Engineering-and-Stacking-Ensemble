# ==================================================
# Phase II: EDA - Univariate Analysis
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# Data_Sanity.py
from Load_Data import df

# Now df is available
print(df.info())
print("=== PHASE II: EXPLORATORY DATA ANALYSIS ===")

# 1. Univariate Analysis - Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

print(f"\n--- NUMERIC COLUMNS ({len(numeric_cols)}) ---")
print(list(numeric_cols))
print(f"\n--- CATEGORICAL COLUMNS ({len(categorical_cols)}) ---")
print(list(categorical_cols))

# Create histograms and boxplots for each numeric feature
if len(numeric_cols) > 0:
    print("\n--- 1. UNIVARIATE ANALYSIS: NUMERIC FEATURES ---")

    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, 2 * n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, col in enumerate(numeric_cols):
        row_idx = i // n_cols
        col_idx = (i % n_cols) * 2

        # Histogram
        axes[row_idx, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[row_idx, col_idx].set_title(f'Histogram of {col}', fontsize=12)
        axes[row_idx, col_idx].set_xlabel(col)
        axes[row_idx, col_idx].set_ylabel('Frequency')

        # Boxplot
        axes[row_idx, col_idx + 1].boxplot(df[col].dropna())
        axes[row_idx, col_idx + 1].set_title(f'Boxplot of {col}', fontsize=12)
        axes[row_idx, col_idx + 1].set_ylabel(col)

    # Remove empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row_idx = i // n_cols
        col_idx = (i % n_cols) * 2
        fig.delaxes(axes[row_idx, col_idx])
        fig.delaxes(axes[row_idx, col_idx + 1])

    plt.tight_layout()
    plt.show()

# Analyze categorical columns
if len(categorical_cols) > 0:
    print("\n--- 2. UNIVARIATE ANALYSIS: CATEGORICAL FEATURES ---")

    n_cat = min(4, len(categorical_cols))
    fig, axes = plt.subplots(1, n_cat, figsize=(20, 5))
    if n_cat == 1:
        axes = [axes]

    for i, col in enumerate(categorical_cols[:n_cat]):
        value_counts = df[col].value_counts()
        if len(value_counts) > 10:  # Too many categories to plot nicely
            print(f"\n{col} has {len(value_counts)} unique values (showing top 10):")
            value_counts.head(10).plot(kind='bar', ax=axes[i], color='lightcoral')
        else:
            value_counts.plot(kind='bar', ax=axes[i], color='lightcoral')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()