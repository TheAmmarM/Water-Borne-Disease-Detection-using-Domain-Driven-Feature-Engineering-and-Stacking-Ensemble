import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load your cleaned dataset
df = pd.read_csv('cleaned_data_with_features.csv')  # Adjust path

# Separate features and target
X = df.drop(['Water Quality', 'Water_Quality_Category'], axis=1)
y = df['Water Quality']

print("=== FEATURE SCALING ===")
print(f"Original feature shape: {X.shape}")
print(f"Features to scale: {X.columns.tolist()}\n")

# Split data BEFORE scaling (important!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTarget distribution in train:")
print(y_train.value_counts().sort_index())

# ============================================
# METHOD 1: StandardScaler (Most Common)
# ============================================
# Use when: Data roughly normally distributed
# Formula: (x - mean) / std_dev
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train)
X_test_standard = scaler_standard.transform(X_test)

# Convert back to DataFrame for readability
X_train_standard_df = pd.DataFrame(
    X_train_standard,
    columns=X.columns,
    index=X_train.index
)

print("\n--- StandardScaler Results ---")
print("Sample scaled values (first 3 rows):")
print(X_train_standard_df.head(3))
print(f"\nMean after scaling: {X_train_standard_df.mean().mean():.6f}")
print(f"Std after scaling: {X_train_standard_df.std().mean():.6f}")

# ============================================
# METHOD 2: MinMaxScaler (For Neural Networks)
# ============================================
# Use when: Need bounded range [0, 1]
# Formula: (x - min) / (max - min)
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train)
X_test_minmax = scaler_minmax.transform(X_test)

print("\n--- MinMaxScaler Results ---")
print(f"Min value: {X_train_minmax.min()}")
print(f"Max value: {X_train_minmax.max()}")

# ============================================
# METHOD 3: RobustScaler (For Outlier-Heavy Data)
# ============================================
# Use when: Still have outliers after treatment
# Formula: (x - median) / IQR
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train)
X_test_robust = scaler_robust.transform(X_test)

print("\n--- RobustScaler Results ---")
print(f"Median after scaling: {np.median(X_train_robust):.6f}")

# ============================================
# VISUALIZATION: Before vs After Scaling
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Before scaling
axes[0, 0].boxplot(X_train.values, labels=X.columns)
axes[0, 0].set_title('Before Scaling', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Features')
axes[0, 0].set_ylabel('Original Values')
axes[0, 0].tick_params(axis='x', rotation=90)

# StandardScaler
axes[0, 1].boxplot(X_train_standard, labels=X.columns)
axes[0, 1].set_title('StandardScaler (Mean=0, Std=1)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Features')
axes[0, 1].set_ylabel('Scaled Values')
axes[0, 1].tick_params(axis='x', rotation=90)

# MinMaxScaler
axes[1, 0].boxplot(X_train_minmax, labels=X.columns)
axes[1, 0].set_title('MinMaxScaler (Range [0,1])', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Features')
axes[1, 0].set_ylabel('Scaled Values')
axes[1, 0].tick_params(axis='x', rotation=90)

# RobustScaler
axes[1, 1].boxplot(X_train_robust, labels=X.columns)
axes[1, 1].set_title('RobustScaler (Median/IQR)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Features')
axes[1, 1].set_ylabel('Scaled Values')
axes[1, 1].tick_params(axis='x', rotation=90)

plt.tight_layout()
plt.savefig('scaling_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Visualization saved as 'scaling_comparison.png'")

# ============================================
# SAVE SCALED DATA
# ============================================
# Save StandardScaler version (recommended for most ML models)
X_train_final = pd.DataFrame(
    X_train_standard,
    columns=X.columns,
    index=X_train.index
)
X_test_final = pd.DataFrame(
    X_test_standard,
    columns=X.columns,
    index=X_test.index
)

# Combine with target
train_final = pd.concat([X_train_final, y_train], axis=1)
test_final = pd.concat([X_test_final, y_test], axis=1)

train_final.to_csv('train_scaled.csv', index=False)
test_final.to_csv('test_scaled.csv', index=False)

print("\n✅ Scaled datasets saved:")
print("   - train_scaled.csv")
print("   - test_scaled.csv")

# Save scalers for future use
import joblib
joblib.dump(scaler_standard, 'standard_scaler.pkl')
joblib.dump(scaler_minmax, 'minmax_scaler.pkl')
joblib.dump(scaler_robust, 'robust_scaler.pkl')

print("\n✅ Scalers saved for deployment:")
print("   - standard_scaler.pkl")
print("   - minmax_scaler.pkl")
print("   - robust_scaler.pkl")

print("\n" + "="*60)
print("SCALING COMPLETE - READY FOR MODELING!")
print("="*60)