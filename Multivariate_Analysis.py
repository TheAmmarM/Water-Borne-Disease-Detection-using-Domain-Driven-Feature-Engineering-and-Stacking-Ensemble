# ==================================================
# Phase II: EDA - Multivariate Analysis
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

# 3. Multivariate Analysis
print("\n--- 3. MULTIVARIATE ANALYSIS ---")

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Correlation matrix heatmap for numeric features
if len(numeric_cols) > 1:
    plt.figure(figsize=(12, 8))
    correlation_matrix = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle

    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Correlation Matrix Heatmap (Lower Triangle)')
    plt.tight_layout()
    plt.show()

    # Find high correlations
    print("\n--- STRONG CORRELATIONS (|r| > 0.7) ---")
    high_corr = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.7:
                high_corr.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    correlation_matrix.iloc[i, j]
                ))

    if high_corr:
        for corr in high_corr:
            print(f"{corr[0]} - {corr[1]}: {corr[2]:.3f}")
    else:
        print("No strong correlations found (|r| > 0.7)")

# 4. Target Variable Analysis
print("\n--- 4. TARGET VARIABLE IDENTIFICATION ---")

# Let's try to identify potential target columns
print("Potential target columns analysis:")

# Look for columns that might be targets (classification labels or continuous targets)
potential_targets = []

for col in df.columns:
    unique_vals = df[col].nunique()
    null_count = df[col].isnull().sum()

    # Criteria for potential targets
    if unique_vals == 2:  # Binary classification
        potential_targets.append((col, 'Binary Classification', unique_vals, null_count))
    elif 2 < unique_vals <= 10:  # Multi-class classification
        potential_targets.append((col, 'Multi-class Classification', unique_vals, null_count))
    elif unique_vals > 10 and pd.api.types.is_numeric_dtype(df[col]):  # Regression
        potential_targets.append((col, 'Regression', unique_vals, null_count))

if potential_targets:
    targets_df = pd.DataFrame(potential_targets,
                              columns=['Column', 'Problem Type', 'Unique Values', 'Null Count'])
    print("\nPotential target variables found:")
    print(targets_df)
else:
    print("No clear target variables identified automatically.")
    print("\nAll columns summary:")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}, {df[col].nunique()} unique values, {df[col].isnull().sum()} nulls")
