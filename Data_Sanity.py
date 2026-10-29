# ==================================================
# Phase I: Data Sanity & Validation
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

print("=== PHASE I: DATA SANITY & VALIDATION ===")

# 1. Schema Validation & Data Types
print("\n--- 1. DATA TYPES & BASIC INFO ---")
print(df.info())
print("\n--- SUMMARY STATISTICS ---")
print(df.describe())

# 2. Check for missing values
print("\n--- 2. MISSING VALUES ANALYSIS ---")
missing_data = df.isnull().sum()
missing_percent = (missing_data / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing_data,
    'Missing Percentage': missing_percent
})
print("Missing values per column:")
print(missing_df[missing_df['Missing Count'] > 0])
if missing_df[missing_df['Missing Count'] > 0].empty:
    print("No missing values found!")

# 3. Duplicate Check
print("\n--- 3. DUPLICATE CHECK ---")
print(f"Number of duplicate rows: {df.duplicated().sum()}")