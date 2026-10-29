# ==================================================
# CRITICAL: Domain-Based Data Cleaning
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
print("=== CRITICAL DATA CLEANING ===")

# Define PHYSICALLY IMPOSSIBLE ranges for water quality parameters
domain_limits = {
    'Temp': (0, 50),  # Water temp in °C (0-50 realistic)
    'Turbidity (cm)': (0, 1000),  # NTU should be positive
    'DO(mg/L)': (0, 20),  # Dissolved Oxygen
    'BOD (mg/L)': (0, 50),  # Biological Oxygen Demand
    'CO2': (0, 100),  # Carbon Dioxide
    'pH`': (0, 14),  # pH scale
    'Alkalinity (mg L-1 )': (0, 500),  # Alkalinity
    'Hardness (mg L-1 )': (0, 500),  # Water hardness
    'Calcium (mg L-1 )': (0, 200),  # Calcium concentration
    'Ammonia (mg L-1 )': (0, 10),  # Ammonia - toxic at high levels
    'Nitrite (mg L-1 )': (0, 10),  # Nitrite
    'Phosphorus (mg L-1 )': (0, 10),  # Phosphorus
    'H2S (mg L-1 )': (0, 5),  # Hydrogen Sulfide - highly toxic
    'Plankton (No. L-1)': (0, 10000)  # Plankton count
}

# Check for domain violations
print("--- DOMAIN VIOLATIONS ---")
domain_violations = {}
original_shape = df.shape

for col, (min_val, max_val) in domain_limits.items():
    violations = df[(df[col] < min_val) | (df[col] > max_val)]
    violation_count = len(violations)
    if violation_count > 0:
        domain_violations[col] = violation_count
        print(f"{col}: {violation_count} violations ({violation_count / len(df) * 100:.2f}%)")

print(f"\nTotal domain violations: {sum(domain_violations.values())}")

# Show the most extreme domain violations
print("\n--- EXTREME DOMAIN VIOLATIONS (First 10) ---")
extreme_violations = []
for col in domain_limits.keys():
    min_allowed, max_allowed = domain_limits[col]
    extreme_low = df[df[col] < min_allowed]
    extreme_high = df[df[col] > max_allowed]

    if not extreme_low.empty:
        extreme_violations.extend(extreme_low.head(5).to_dict('records'))
    if not extreme_high.empty:
        extreme_violations.extend(extreme_high.head(5).to_dict('records'))

extreme_df = pd.DataFrame(extreme_violations[:10])  # Show first 10
print(extreme_df)