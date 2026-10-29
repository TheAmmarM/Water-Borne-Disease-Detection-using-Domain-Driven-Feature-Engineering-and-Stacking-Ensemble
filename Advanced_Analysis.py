# ==================================================
# Phase II: EDA - Advanced Analysis
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import dataset from Load_Data.py
from Load_Data import df

# Identify numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 5. Outlier Detection using IQR
print("\n--- 5. OUTLIER DETECTION (IQR METHOD) ---")

if len(numeric_cols) > 0:
    outlier_summary = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percent = (outlier_count / len(df)) * 100

        outlier_summary.append({
            'Column': col,
            'Outlier_Count': outlier_count,
            'Outlier_Percent': outlier_percent,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })

    outlier_df = pd.DataFrame(outlier_summary)
    print("Outlier summary (IQR method):")
    print(outlier_df)

# 6. Time-series analysis (if datetime column exists)
datetime_cols = df.select_dtypes(include=['datetime64']).columns
if len(datetime_cols) > 0:
    print(f"\n--- 6. TIME-SERIES ANALYSIS ---")
    print(f"Datetime columns found: {list(datetime_cols)}")

    # Plot first datetime column against first numeric column if available
    if len(numeric_cols) > 0:
        time_col = datetime_cols[0]
        value_col = numeric_cols[0]

        plt.figure(figsize=(15, 6))
        plt.plot(df[time_col], df[value_col])
        plt.title(f'{value_col} over Time')
        plt.xlabel(time_col)
        plt.ylabel(value_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
else:
    print("\n--- 6. TIME-SERIES ANALYSIS ---")
    print("No datetime columns found.")

print("\n" + "=" * 50)
print("DATA EXPLORATION COMPLETE")
print("=" * 50)
