# ==================================================
# Smart Outlier Treatment (Final Debugged Version)
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
from Load_Data import df  # ✅ make sure Load_Data.py defines df properly

print(df.info())
print("\n=== SMART OUTLIER TREATMENT STARTED ===")

# --------------------------------------------------
# Domain limits (adjust according to your dataset)
# --------------------------------------------------
domain_limits = {
    'pH': (0, 14),
    'Temperature': (0, 100),
    'Turbidity': (0, 1000),
    'Dissolved Oxygen': (0, 20),
    'Conductivity': (0, 10000),
    'Ammonia (mg L-1 )': (0, 50),
    'H2S (mg L-1 )': (0, 10),
    'Nitrite (mg L-1 )': (0, 10),
}

# --------------------------------------------------
# Identify numeric columns
# --------------------------------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# --------------------------------------------------
# Handle invalid or out-of-range values before imputation
# --------------------------------------------------
df_imputed = df.copy()

for col, (min_val, max_val) in domain_limits.items():
    if col in df_imputed.columns:
        mask = (df_imputed[col] < min_val) | (df_imputed[col] > max_val)
        if mask.sum() > 0:
            print(f"[INFO] {mask.sum()} invalid values in '{col}' replaced with NaN")
        df_imputed.loc[mask, col] = np.nan

# Median imputation for missing values
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df_imputed),
                          columns=df_imputed.columns)

# --------------------------------------------------
# Smart Outlier Treatment Function
# --------------------------------------------------
def smart_outlier_treatment(series, feature_name):
    """Apply domain-aware IQR-based capping for outliers."""
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    # Domain-based logic
    if feature_name in ['Temperature', 'Dissolved Oxygen', 'pH']:
        lower_bound = Q1 - 1.0 * IQR
        upper_bound = Q3 + 1.5 * IQR
    elif feature_name in ['Ammonia (mg L-1 )', 'H2S (mg L-1 )', 'Nitrite (mg L-1 )']:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 3.0 * IQR
    else:
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

    # Apply domain constraints if available
    min_domain, max_domain = domain_limits.get(feature_name, (None, None))
    if min_domain is not None:
        lower_bound = max(lower_bound, min_domain)
    if max_domain is not None:
        upper_bound = min(upper_bound, max_domain)

    return lower_bound, upper_bound

# --------------------------------------------------
# Apply Smart Outlier Treatment
# --------------------------------------------------
df_final = df_imputed.copy()
outlier_report = []

for col in numeric_cols:
    if col == 'Water Quality':  # skip target if present
        continue

    lower_bound, upper_bound = smart_outlier_treatment(df_final[col], col)
    mask = (df_final[col] < lower_bound) | (df_final[col] > upper_bound)
    outlier_count = mask.sum()
    outlier_percent = (outlier_count / len(df_final)) * 100

    # Cap outliers to bounds
    df_final.loc[df_final[col] < lower_bound, col] = lower_bound
    df_final.loc[df_final[col] > upper_bound, col] = upper_bound

    outlier_report.append({
        'Feature': col,
        'Outliers_Found': outlier_count,
        'Outlier_%': round(outlier_percent, 2),
        'Lower_Bound': round(lower_bound, 2),
        'Upper_Bound': round(upper_bound, 2)
    })

# --------------------------------------------------
# Outlier Summary
# --------------------------------------------------
outlier_df = pd.DataFrame(outlier_report)
print("\n=== OUTLIER REPORT ===")
print(outlier_df)

# --------------------------------------------------
# Save cleaned dataset for next phase
# --------------------------------------------------
df_final.to_csv("cleaned_data_with_features.csv", index=False)
print("\n✅ Cleaned dataset saved as 'cleaned_data_with_features.csv'")

# --------------------------------------------------
# Optional: Visualization of capped vs. original data
# --------------------------------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_final[numeric_cols], orient="h")
plt.title("Boxplot After Smart Outlier Treatment", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("outlier_treated_boxplot.png", dpi=300)
plt.close()

print("✅ Visualization saved as 'outlier_treated_boxplot.png'")

print("\n=== SMART OUTLIER TREATMENT COMPLETE ===")
