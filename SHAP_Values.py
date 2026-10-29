"""
SHAP Analysis - Production Ready Version
=========================================
Handles all edge cases including 3D array SHAP values
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================
# LOAD DATA
# ============================================
print("=" * 70)
print("SHAP (SHapley Additive exPlanations) ANALYSIS")
print("=" * 70)
print("\nSHAP values explain the contribution of each feature to predictions")
print("Positive SHAP = pushes prediction higher")
print("Negative SHAP = pushes prediction lower\n")

try:
    train_df = pd.read_csv('train_scaled.csv')
    test_df = pd.read_csv('test_scaled.csv')
    print("✓ Data loaded successfully")
except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("Please ensure 'train_scaled.csv' and 'test_scaled.csv' exist.")
    sys.exit(1)

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}")

# ============================================
# LOAD OR TRAIN MODEL
# ============================================
print("\n" + "=" * 70)
print("MODEL LOADING")
print("=" * 70)

try:
    model = joblib.load('final_champion_model.pkl')
    print("✓ Loaded existing model: final_champion_model.pkl")
except:
    print("✗ Model not found. Training new Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    joblib.dump(model, 'final_champion_model.pkl')
    print("✓ Model trained and saved")

print(f"Model type: {type(model).__name__}")

# ============================================
# PREPARE SAMPLE DATA
# ============================================
print("\n" + "=" * 70)
print("PREPARING DATA SAMPLE")
print("=" * 70)

sample_size = min(500, len(X_test))
X_test_sample = X_test.iloc[:sample_size].copy()
y_test_sample = y_test.iloc[:sample_size].copy()

print(f"Using {sample_size} test samples for SHAP analysis")

# ============================================
# CREATE SHAP EXPLAINER AND COMPUTE VALUES
# ============================================
print("\n" + "=" * 70)
print("COMPUTING SHAP VALUES")
print("=" * 70)

print("🔄 Creating SHAP explainer... (this may take a few minutes)")
explainer = shap.TreeExplainer(model)

print(f"🔄 Computing SHAP values for {sample_size} samples...")
shap_values = explainer.shap_values(X_test_sample)

# Debug output
print(f"✓ SHAP values computed")
print(f"SHAP values type: {type(shap_values)}")

# ============================================
# HANDLE DIFFERENT SHAP VALUE FORMATS
# ============================================
if isinstance(shap_values, list):
    # Format 1: List of arrays (standard multi-class)
    print(f"Format: List of arrays (multi-class)")
    print(f"Number of classes: {len(shap_values)}")
    print(f"Shape per class: {shap_values[0].shape}")
    n_classes = len(shap_values)

elif isinstance(shap_values, np.ndarray):
    if shap_values.ndim == 3:
        # Format 2: 3D array (samples, features, classes)
        print(f"Format: 3D numpy array")
        print(f"Original shape: {shap_values.shape}")
        print(f"Converting to list format...")

        n_classes = shap_values.shape[2]
        shap_values = [shap_values[:, :, i] for i in range(n_classes)]

        print(f"✓ Converted to {n_classes} arrays")
        print(f"Shape per class: {shap_values[0].shape}")

    elif shap_values.ndim == 2:
        # Format 3: 2D array (binary or regression)
        print(f"Format: 2D numpy array (single output)")
        print(f"Shape: {shap_values.shape}")
        shap_values = [shap_values]
        n_classes = 1

    else:
        print(f"❌ ERROR: Unexpected SHAP values shape: {shap_values.shape}")
        sys.exit(1)
else:
    print(f"❌ ERROR: Unexpected SHAP values type: {type(shap_values)}")
    sys.exit(1)

# Validate shapes
for i, sv in enumerate(shap_values):
    expected_shape = (sample_size, X_train.shape[1])
    if sv.shape != expected_shape:
        print(f"⚠️  Warning: Class {i} shape {sv.shape} != expected {expected_shape}")

# Define class names
class_names = [f'Class {i}' for i in range(n_classes)]
if n_classes == 3:
    class_names = ['Poor (0)', 'Moderate (1)', 'Good (2)']

print(f"\n✓ SHAP values ready for analysis")
print(f"Classes: {class_names}")

# ============================================
# 1. GLOBAL FEATURE IMPORTANCE
# ============================================
print("\n" + "=" * 70)
print("1. GLOBAL FEATURE IMPORTANCE (ALL CLASSES)")
print("=" * 70)

try:
    # Calculate mean |SHAP| for each class
    mean_shap_per_class = []
    for i, sv in enumerate(shap_values):
        class_mean = np.abs(sv).mean(axis=0)
        mean_shap_per_class.append(class_mean)
        print(f"Class {i} mean shape: {class_mean.shape}")

    # Average across classes
    if n_classes > 1:
        mean_shap_values = np.mean(mean_shap_per_class, axis=0)
    else:
        mean_shap_values = mean_shap_per_class[0]

    # Ensure 1D
    mean_shap_values = np.atleast_1d(mean_shap_values).flatten()
    print(f"Final mean shape: {mean_shap_values.shape}")

    # Validate length
    if len(mean_shap_values) != len(X_train.columns):
        print(f"⚠️  Length mismatch: {len(mean_shap_values)} vs {len(X_train.columns)}")
        min_len = min(len(mean_shap_values), len(X_train.columns))
        mean_shap_values = mean_shap_values[:min_len]
        features_list = X_train.columns[:min_len].tolist()
    else:
        features_list = X_train.columns.tolist()

    # Create DataFrame
    feature_importance = pd.DataFrame({
        'Feature': features_list,
        'Mean |SHAP|': mean_shap_values.tolist()
    }).sort_values('Mean |SHAP|', ascending=False)

    print("\n✓ Feature importance calculated")
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))

except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Visualize
plt.figure(figsize=(12, 8))
plt.barh(
    feature_importance['Feature'][:15],
    feature_importance['Mean |SHAP|'][:15],
    color='steelblue'
)
plt.xlabel('Mean |SHAP value|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Global Feature Importance (SHAP)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('shap_global_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: shap_global_importance.png")
plt.close()

# ============================================
# 2. SHAP SUMMARY PLOTS (PER CLASS)
# ============================================
print("\n" + "=" * 70)
print("2. SHAP SUMMARY PLOTS (PER CLASS)")
print("=" * 70)

for i, (class_shap, class_name) in enumerate(zip(shap_values, class_names)):
    print(f"Creating summary plot for {class_name}...")

    try:
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            class_shap,
            X_test_sample,
            show=False
        )
        plt.title(f'SHAP Summary - {class_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'shap_summary_class_{i}.png', dpi=300, bbox_inches='tight')
        print(f"✅ Saved: shap_summary_class_{i}.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Failed: {e}")

# ============================================
# 3. PER-CLASS FEATURE IMPORTANCE (BAR PLOTS)
# ============================================
print("\n" + "=" * 70)
print("3. PER-CLASS FEATURE IMPORTANCE")
print("=" * 70)

fig, axes = plt.subplots(1, n_classes, figsize=(7 * n_classes, 6))

# Handle single class case
if n_classes == 1:
    axes = [axes]

for i, (class_shap, class_name) in enumerate(zip(shap_values, class_names)):
    mean_shap_class = np.abs(class_shap).mean(axis=0)

    feature_imp_class = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'Mean |SHAP|': mean_shap_class.tolist()
    }).sort_values('Mean |SHAP|', ascending=False).head(10)

    axes[i].barh(feature_imp_class['Feature'], feature_imp_class['Mean |SHAP|'], color='seagreen')
    axes[i].set_xlabel('Mean |SHAP value|', fontsize=11)
    axes[i].set_title(f'{class_name}', fontsize=12, fontweight='bold')
    axes[i].invert_yaxis()

plt.tight_layout()
plt.savefig('shap_per_class_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: shap_per_class_importance.png")
plt.close()

# ============================================
# 4. SAVE RESULTS
# ============================================
print("\n" + "=" * 70)
print("SAVING RESULTS")
print("=" * 70)

feature_importance.to_csv('shap_feature_importance.csv', index=False)
print("✅ Feature importance saved: 'shap_feature_importance.csv'")

# Save per-class importance
for i, class_name in enumerate(class_names):
    mean_shap_class = np.abs(shap_values[i]).mean(axis=0)
    class_importance = pd.DataFrame({
        'Feature': X_train.columns.tolist(),
        'Mean |SHAP|': mean_shap_class.tolist()
    }).sort_values('Mean |SHAP|', ascending=False)

    class_importance.to_csv(f'shap_importance_class_{i}.csv', index=False)
    print(f"✅ Saved: shap_importance_class_{i}.csv")

# ============================================
# SUMMARY
# ============================================
print("\n" + "=" * 70)
print("SHAP ANALYSIS COMPLETE!")
print("=" * 70)

print("\n📊 Generated Files:")
print("   - shap_global_importance.png")
for i in range(n_classes):
    print(f"   - shap_summary_class_{i}.png")
print("   - shap_per_class_importance.png")
print("   - shap_feature_importance.csv")
for i in range(n_classes):
    print(f"   - shap_importance_class_{i}.csv")

print("\n💡 Key Insights:")
print(f"   Number of classes: {n_classes}")
print(f"   Top Feature: {feature_importance.iloc[0]['Feature']}")
print(f"   Mean |SHAP|: {feature_importance.iloc[0]['Mean |SHAP|']:.4f}")
print(f"\n   Top 5 Features:")
for idx in range(min(5, len(feature_importance))):
    row = feature_importance.iloc[idx]
    print(f"      {idx + 1}. {row['Feature']}: {row['Mean |SHAP|']:.4f}")

print("\n✅ Core SHAP analysis completed successfully!")
print("=" * 70)