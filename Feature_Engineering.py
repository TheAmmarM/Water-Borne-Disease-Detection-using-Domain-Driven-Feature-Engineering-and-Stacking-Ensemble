# ==================================================
# Feature Engineering for Water Quality (Debugged)
# ==================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# Import Data
# --------------------------------------------------
# Make sure these modules exist in your project directory
from Load_Data import df
from Outlier_Treatment import df_final  # ✅ import cleaned data correctly

# Now df is available
print(df.info())
print("\n=== FEATURE ENGINEERING ===")

# --------------------------------------------------
# Copy cleaned dataset
# --------------------------------------------------
df_engineered = df_final.copy()

# --------------------------------------------------
# 1. Oxygen-related ratios
# --------------------------------------------------
df_engineered['BOD_DO_Ratio'] = df_engineered['BOD (mg/L)'] / (
    df_engineered['DO(mg/L)'] + 0.001  # avoid div-by-zero
)
df_engineered['Oxygen_Saturation'] = df_engineered['DO(mg/L)'] / 8.0  # approximate saturation

# --------------------------------------------------
# 2. Nutrient pollution indicators
# --------------------------------------------------
df_engineered['Nutrient_Pollution_Index'] = (
    df_engineered['Ammonia (mg L-1 )'] +
    df_engineered['Nitrite (mg L-1 )'] +
    df_engineered['Phosphorus (mg L-1 )']
)

# --------------------------------------------------
# 3. Water stability indicators
# --------------------------------------------------
df_engineered['Alkalinity_Hardness_Ratio'] = df_engineered['Alkalinity (mg L-1 )'] / (
    df_engineered['Hardness (mg L-1 )'] + 0.001
)

# --------------------------------------------------
# 4. Categorize water quality
# --------------------------------------------------
def categorize_water_quality(temp, do, ph, turbidity):
    """Categorical encoding of water quality thresholds."""
    categories = []

    # Temperature
    if temp < 10:
        cat_temp = 'cold'
    elif temp < 25:
        cat_temp = 'moderate'
    else:
        cat_temp = 'warm'
    categories.append(cat_temp)

    # Dissolved Oxygen
    if do < 2:
        cat_do = 'hypoxic'
    elif do < 5:
        cat_do = 'low'
    elif do < 8:
        cat_do = 'moderate'
    else:
        cat_do = 'high'
    categories.append(cat_do)

    # pH
    if ph < 6.5:
        cat_ph = 'acidic'
    elif ph < 7.5:
        cat_ph = 'neutral'
    elif ph < 8.5:
        cat_ph = 'alkaline'
    else:
        cat_ph = 'high_alkaline'
    categories.append(cat_ph)

    return '_'.join(categories)

# --------------------------------------------------
# Apply categorization safely
# --------------------------------------------------
# handle possible column name variations
temp_col = 'Temp' if 'Temp' in df_engineered.columns else 'Temperature'
turbidity_col = 'Turbidity (cm)' if 'Turbidity (cm)' in df_engineered.columns else 'Turbidity'
ph_col = 'pH' if 'pH' in df_engineered.columns else 'pH`'

df_engineered['Water_Quality_Category'] = df_engineered.apply(
    lambda row: categorize_water_quality(
        row[temp_col],
        row['DO(mg/L)'],
        row[ph_col],
        row[turbidity_col]
    ),
    axis=1
)

# --------------------------------------------------
# Feature Summary
# --------------------------------------------------
print("New features created:")
new_features = [col for col in df_engineered.columns if col not in df_final.columns]
print(new_features)
print(f"\nFinal dataset shape: {df_engineered.shape}")
print(f"Total features: {len(df_engineered.columns)}")

# --------------------------------------------------
# Save engineered dataset for next phase
# --------------------------------------------------
df_engineered.to_csv("cleaned_data_with_features.csv", index=False)
print("\n✅ Feature engineered dataset saved as 'cleaned_data_with_features.csv'")
