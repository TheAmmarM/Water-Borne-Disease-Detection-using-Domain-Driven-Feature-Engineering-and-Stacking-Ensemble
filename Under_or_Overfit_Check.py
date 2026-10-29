"""
COMPREHENSIVE OVER/UNDERFITTING DIAGNOSTIC TOOL
================================================
Analyzes models for bias-variance tradeoff issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    learning_curve, validation_curve, cross_val_score, StratifiedKFold
)
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

print("=" * 80)
print("COMPREHENSIVE OVER/UNDERFITTING DIAGNOSTIC")
print("=" * 80)

# Load data and model
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

try:
    model = joblib.load('champion_model.pkl')
    selected_features = joblib.load('selected_features.pkl')
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    print("✓ Loaded champion model and selected features")
except:
    print("⚠️  Champion model not found. Run ultimate_pipeline.py first!")
    import sys

    sys.exit(1)

# ============================================
# DIAGNOSTIC 1: BIAS-VARIANCE DECOMPOSITION
# ============================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 1: BIAS-VARIANCE ANALYSIS")
print("=" * 80)

# Train/Test accuracy gap
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)
gap = train_acc - test_acc

print(f"\n📊 Accuracy Analysis:")
print(f"   Training Accuracy:   {train_acc:.4f}")
print(f"   Test Accuracy:       {test_acc:.4f}")
print(f"   Accuracy Gap:        {gap:.4f}")

# Cross-validation for variance estimate
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')

print(f"\n📊 Cross-Validation Analysis:")
print(f"   CV Mean Accuracy:    {cv_scores.mean():.4f}")
print(f"   CV Std Deviation:    {cv_scores.std():.4f}")
print(f"   CV Score Range:      {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# Diagnosis
print(f"\n🔍 DIAGNOSIS:")

if train_acc < 0.85 and test_acc < 0.85:
    print("   ⚠️  HIGH BIAS (UNDERFITTING)")
    print("   Problem: Model is too simple")
    print("   Solutions:")
    print("      • Increase model complexity (more layers, more trees)")
    print("      • Add more features")
    print("      • Reduce regularization")
    print("      • Train longer")

elif gap > 0.10:
    print("   ⚠️  HIGH VARIANCE (OVERFITTING)")
    print("   Problem: Model memorizes training data")
    print("   Solutions:")
    print("      • Increase regularization")
    print("      • Get more training data")
    print("      • Reduce model complexity")
    print("      • Use dropout/early stopping")
    print("      • Feature selection")

elif gap > 0.05:
    print("   ⚠️  MILD OVERFITTING")
    print("   Problem: Slight memorization")
    print("   Solutions:")
    print("      • Light regularization tuning")
    print("      • Ensemble methods")

elif cv_scores.std() > 0.05:
    print("   ⚠️  HIGH VARIANCE (Unstable)")
    print("   Problem: Model performance varies across folds")
    print("   Solutions:")
    print("      • More training data")
    print("      • Stabilize with ensemble methods")

else:
    print("   ✅ GOOD FIT!")
    print("   Model has appropriate bias-variance tradeoff")

# ============================================
# DIAGNOSTIC 2: LEARNING CURVES
# ============================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 2: LEARNING CURVES")
print("=" * 80)

print("\n🔄 Computing learning curves...")

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    model, X_train, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

# Analyze curves
final_gap = train_mean[-1] - val_mean[-1]
convergence = abs(train_mean[-1] - train_mean[-2]) < 0.01

print(f"\n📊 Learning Curve Analysis:")
print(f"   Final Training Score:     {train_mean[-1]:.4f}")
print(f"   Final Validation Score:   {val_mean[-1]:.4f}")
print(f"   Final Gap:                {final_gap:.4f}")
print(f"   Convergence:              {'Yes' if convergence else 'No'}")

print(f"\n🔍 DIAGNOSIS:")
if val_mean[-1] < 0.85:
    if final_gap < 0.05:
        print("   ⚠️  UNDERFITTING")
        print("   Both curves plateau at low accuracy")
        print("   Solution: Increase model complexity")
    else:
        print("   ⚠️  HIGH VARIANCE + LOW CAPACITY")
        print("   Large gap but low validation score")
        print("   Solution: More data + better features")

elif final_gap > 0.10:
    print("   ⚠️  OVERFITTING")
    print("   Large gap between training and validation")
    print("   Solution: Regularization or more data")

elif not convergence:
    print("   ⚠️  UNDERFITTING (Insufficient Training)")
    print("   Model hasn't converged yet")
    print("   Solution: Train with more data or longer")

else:
    print("   ✅ OPTIMAL FIT")
    print("   Curves converge at high accuracy")

# Plot learning curves
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Learning curves
axes[0].plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
             label='Training score', linewidth=2, markersize=8)
axes[0].fill_between(train_sizes_abs, train_mean - train_std,
                     train_mean + train_std, alpha=0.2, color='steelblue')

axes[0].plot(train_sizes_abs, val_mean, 'o-', color='coral',
             label='Validation score', linewidth=2, markersize=8)
axes[0].fill_between(train_sizes_abs, val_mean - val_std,
                     val_mean + val_std, alpha=0.2, color='coral')

axes[0].set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
axes[0].set_title('Learning Curves', fontsize=14, fontweight='bold')
axes[0].legend(loc='best', fontsize=11)
axes[0].grid(alpha=0.3)
axes[0].set_ylim([0.7, 1.02])

# Gap analysis
gaps = train_mean - val_mean
axes[1].plot(train_sizes_abs, gaps, 'o-', color='red',
             linewidth=2, markersize=8)
axes[1].axhline(y=0.05, color='orange', linestyle='--',
                label='Acceptable Gap (5%)')
axes[1].axhline(y=0.10, color='red', linestyle='--',
                label='High Gap (10%)')

axes[1].set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Train-Validation Gap', fontsize=12, fontweight='bold')
axes[1].set_title('Overfitting Gap Analysis', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=11)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_learning_curves.png', dpi=300, bbox_inches='tight')
print("\n✅ Learning curves saved: 'comprehensive_learning_curves.png'")
plt.close()

# ============================================
# DIAGNOSTIC 3: VALIDATION CURVES
# ============================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 3: VALIDATION CURVES (Hyperparameter Sensitivity)")
print("=" * 80)

# Try to get model complexity parameter
model_type = type(model).__name__

print(f"\nModel type: {model_type}")

complexity_param = None
param_range = None

if 'RandomForest' in model_type:
    complexity_param = 'max_depth'
    param_range = [5, 10, 15, 20, 25, 30, None]
    print(f"Testing parameter: {complexity_param}")

elif 'XGB' in model_type or 'LightGBM' in model_type:
    complexity_param = 'max_depth'
    param_range = [3, 5, 7, 9, 11, 13, 15]
    print(f"Testing parameter: {complexity_param}")

elif 'Gradient' in model_type:
    complexity_param = 'max_depth'
    param_range = [3, 5, 7, 9, 11]
    print(f"Testing parameter: {complexity_param}")

if complexity_param and param_range:
    print(f"🔄 Computing validation curves for {complexity_param}...")

    # Handle None in param_range
    param_range_numeric = [p if p is not None else 999 for p in param_range]

    try:
        train_scores_val, test_scores_val = validation_curve(
            type(model)(**{k: v for k, v in model.get_params().items()
                           if k != complexity_param}),
            X_train, y_train,
            param_name=complexity_param,
            param_range=param_range_numeric,
            cv=5,
            scoring='accuracy',
            n_jobs=-1
        )

        train_mean_val = train_scores_val.mean(axis=1)
        train_std_val = train_scores_val.std(axis=1)
        test_mean_val = test_scores_val.mean(axis=1)
        test_std_val = test_scores_val.std(axis=1)

        # Find optimal complexity
        optimal_idx = np.argmax(test_mean_val)
        optimal_value = param_range[optimal_idx]

        print(f"\n📊 Optimal {complexity_param}: {optimal_value}")
        print(f"   Score at optimal: {test_mean_val[optimal_idx]:.4f}")

        # Plot validation curves
        plt.figure(figsize=(10, 6))

        # Convert param_range for plotting
        x_labels = [str(p) for p in param_range]
        x = np.arange(len(param_range))

        plt.plot(x, train_mean_val, 'o-', color='steelblue',
                 label='Training score', linewidth=2, markersize=8)
        plt.fill_between(x, train_mean_val - train_std_val,
                         train_mean_val + train_std_val,
                         alpha=0.2, color='steelblue')

        plt.plot(x, test_mean_val, 'o-', color='coral',
                 label='Validation score', linewidth=2, markersize=8)
        plt.fill_between(x, test_mean_val - test_std_val,
                         test_mean_val + test_std_val,
                         alpha=0.2, color='coral')

        plt.axvline(x=optimal_idx, color='green', linestyle='--',
                    label=f'Optimal: {optimal_value}')

        plt.xlabel(complexity_param, fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        plt.title(f'Validation Curve - {complexity_param}',
                  fontsize=14, fontweight='bold')
        plt.xticks(x, x_labels)
        plt.legend(loc='best', fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('validation_curves.png', dpi=300, bbox_inches='tight')
        print("✅ Validation curves saved: 'validation_curves.png'")
        plt.close()

    except Exception as e:
        print(f"⚠️  Could not compute validation curves: {e}")
else:
    print("⚠️  Validation curves not applicable for this model type")

# ============================================
# DIAGNOSTIC 4: RESIDUAL ANALYSIS
# ============================================
print("\n" + "=" * 80)
print("DIAGNOSTIC 4: PREDICTION CONFIDENCE ANALYSIS")
print("=" * 80)

# Get prediction probabilities
y_pred_proba = model.predict_proba(X_test)
y_pred = model.predict(X_test)

# Confidence scores
confidence_scores = y_pred_proba.max(axis=1)
correct_mask = y_test == y_pred

correct_confidence = confidence_scores[correct_mask]
wrong_confidence = confidence_scores[~correct_mask]

print(f"\n📊 Confidence Analysis:")
print(f"   Correct predictions: {correct_mask.sum()} ({correct_mask.mean():.2%})")
print(f"   Wrong predictions:   {(~correct_mask).sum()} ({(~correct_mask).mean():.2%})")
print(f"   Avg confidence (correct): {correct_confidence.mean():.4f}")
print(f"   Avg confidence (wrong):   {wrong_confidence.mean():.4f}")

print(f"\n🔍 DIAGNOSIS:")
if wrong_confidence.mean() > 0.7:
    print("   ⚠️  OVERCONFIDENT ERRORS")
    print("   Model is very confident even when wrong")
    print("   Solution: Calibration or ensemble uncertainty")
elif correct_confidence.mean() < 0.7:
    print("   ⚠️  UNDERCONFIDENT")
    print("   Model is uncertain even when correct")
    print("   Solution: More training or better features")
else:
    print("   ✅ WELL-CALIBRATED")
    print("   Confidence matches accuracy appropriately")

# Plot confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(correct_confidence, bins=20, alpha=0.7, color='green',
             label='Correct', density=True)
axes[0].hist(wrong_confidence, bins=20, alpha=0.7, color='red',
             label='Wrong', density=True)
axes[0].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
data_to_plot = [correct_confidence, wrong_confidence]
axes[1].boxplot(data_to_plot, labels=['Correct', 'Wrong'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Prediction Confidence', fontsize=11, fontweight='bold')
axes[1].set_title('Confidence Comparison', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Confidence analysis saved: 'confidence_analysis.png'")
plt.close()

# ============================================
# FINAL SUMMARY REPORT
# ============================================
print("\n" + "=" * 80)
print("COMPREHENSIVE DIAGNOSTIC SUMMARY")
print("=" * 80)

# Overall assessment
issues = []
if train_acc - test_acc > 0.10:
    issues.append("High Overfitting")
elif train_acc - test_acc > 0.05:
    issues.append("Mild Overfitting")

if train_acc < 0.85 and test_acc < 0.85:
    issues.append("Underfitting")

if cv_scores.std() > 0.05:
    issues.append("High Variance")

if wrong_confidence.mean() > 0.7:
    issues.append("Overconfident Errors")

print(f"\n🎯 MODEL HEALTH STATUS:")
if len(issues) == 0:
    print("   ✅ EXCELLENT - Model is well-fitted")
    print("   No significant bias-variance issues detected")
elif len(issues) == 1:
    print(f"   ⚠️  MINOR ISSUE - {issues[0]}")
    print("   Model is mostly healthy with minor tuning needed")
else:
    print(f"   ⚠️  MULTIPLE ISSUES - {', '.join(issues)}")
    print("   Model needs significant improvement")

print(f"\n📊 KEY METRICS:")
print(f"   Training Accuracy:        {train_acc:.4f}")
print(f"   Test Accuracy:            {test_acc:.4f}")
print(f"   Cross-Validation Mean:    {cv_scores.mean():.4f}")
print(f"   Cross-Validation Std:     {cv_scores.std():.4f}")
print(f"   Train-Test Gap:           {gap:.4f}")

# Save report
with open('fitting_diagnostic_report.txt', 'w') as f:
    f.write("COMPREHENSIVE OVER/UNDERFITTING DIAGNOSTIC REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Model Type: {model_type}\n")
    f.write(f"Training Accuracy: {train_acc:.4f}\n")
    f.write(f"Test Accuracy: {test_acc:.4f}\n")
    f.write(f"Accuracy Gap: {gap:.4f}\n")
    f.write(f"CV Mean: {cv_scores.mean():.4f}\n")
    f.write(f"CV Std: {cv_scores.std():.4f}\n\n")
    f.write("Issues Detected:\n")
    if len(issues) == 0:
        f.write("  - None (Model is healthy)\n")
    else:
        for issue in issues:
            f.write(f"  - {issue}\n")

print("\n✅ Diagnostic report saved: 'fitting_diagnostic_report.txt'")

print("\n" + "=" * 80)
print("✅ DIAGNOSTIC COMPLETE!")
print("=" * 80)