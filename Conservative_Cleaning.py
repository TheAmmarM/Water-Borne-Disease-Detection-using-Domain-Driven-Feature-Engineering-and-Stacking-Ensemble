# ==================================================
# STRATEGY 1: Conservative Cleaning (Recommended)
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import dataset
from Load_Data import df

# Now df is available
print(df.info())
print("\n=== STRATEGY 1: CONSERVATIVE CLEANING ===")

# ------------------------------------------
# Define domain limits (example realistic ranges)
# Adjust these according to your dataset's meaning
# ------------------------------------------
domain_limits = {
    'pH': (0, 14),
    'Temperature': (0, 100),
    'Turbidity': (0, 1000),
    'Dissolved Oxygen': (0, 20),
    'Conductivity': (0, 10000),
    # Add more columns if applicable
}

# Create a cleaned version of the dataset
df_clean = df.copy()

# Remove only physically impossible values (set to NaN)
rows_removed = 0
for col, (min_val, max_val) in domain_limits.items():
    if col in df_clean.columns:
        mask = (df_clean[col] < min_val) | (df_clean[col] > max_val)
        rows_removed += mask.sum()
        df_clean.loc[mask, col] = np.nan

print(f"Values set to NaN: {rows_removed}")
print(f"Original shape: {df.shape}")
print(f"After domain cleaning - Rows with any NaN: {df_clean.isnull().any(axis=1).sum()}")

# Option 1: Remove rows with any domain violations
df_strict = df_clean.dropna()
print(f"After removing rows with violations: {df_strict.shape}")

# Option 2: Impute domain violations (more conservative)
from sklearn.impute import SimpleImputer

df_imputed = df.copy()
for col, (min_val, max_val) in domain_limits.items():
    if col in df_imputed.columns:
        mask = (df_imputed[col] < min_val) | (df_imputed[col] > max_val)
        df_imputed.loc[mask, col] = np.nan

# Impute with median for each column
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed),
                         columns=df_imputed.columns)

print(f"After imputation shape: {df_imputed.shape}")

# Compare target distribution before/after cleaning
print("\n--- TARGET DISTRIBUTION COMPARISON ---")
if 'Water Quality' in df.columns:
    print("Original target distribution:")
    print(df['Water Quality'].value_counts().sort_index())
    print("\nAfter strict cleaning:")
    print(df_strict['Water Quality'].value_counts().sort_index())
    print("\nAfter imputation:")
    print(df_imputed['Water Quality'].value_counts().sort_index())
else:
    print("Target column 'Water Quality' not found in dataset.")

# Save the cleaned dataset for later use
df_final = df_imputed  # or df_strict, depending on your choice

# Save to CSV or pickle for reusability
df_final.to_csv("Cleaned_Water_Data.csv", index=False)

print("\n✅ Cleaned dataset saved as 'Cleaned_Water_Data.csv'")
