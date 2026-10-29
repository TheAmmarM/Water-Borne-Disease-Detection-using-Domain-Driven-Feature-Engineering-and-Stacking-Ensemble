import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import joblib

# Load data and model
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

# Load best model
best_model = joblib.load('final_champion_model.pkl')

print("=== COMPREHENSIVE MODEL EVALUATION ===\n")

# ============================================
# 1. CONFUSION MATRIX ANALYSIS
# ============================================
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print("--- Confusion Matrix ---")
print(cm)

# Detailed metrics
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Poor (0)', 'Moderate (1)', 'Good (2)'],
    yticklabels=['Poor (0)', 'Moderate (1)', 'Good (2)']
)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix - Champion Model', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_detailed.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved\n")

# Per-class accuracy
print("--- Per-Class Performance ---")
for i in range(3):
    class_total = cm[i].sum()
    class_correct = cm[i, i]
    class_accuracy = class_correct / class_total
    print(f"Class {i}: {class_correct}/{class_total} correct ({class_accuracy:.2%})")

# ============================================
# 2. CLASSIFICATION REPORT
# ============================================
print("\n--- Classification Report ---")
report = classification_report(
    y_test, y_pred,
    target_names=['Poor (0)', 'Moderate (1)', 'Good (2)'],
    output_dict=True
)
print(classification_report(
    y_test, y_pred,
    target_names=['Poor (0)', 'Moderate (1)', 'Good (2)']
))

# ============================================
# 3. ROC CURVES (One-vs-Rest)
# ============================================
print("\n--- ROC-AUC Analysis ---")

# Get probability predictions
y_proba = best_model.predict_proba(X_test)

# Binarize labels for ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3

# Compute ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"Class {i} ROC-AUC: {roc_auc[i]:.4f}")

# Micro-average ROC curve
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
print(f"Micro-average ROC-AUC: {roc_auc['micro']:.4f}")

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'green', 'red']
class_names = ['Poor (0)', 'Moderate (1)', 'Good (2)']

for i, color in enumerate(colors):
    plt.plot(
        fpr[i], tpr[i], color=color, lw=2,
        label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})'
    )

plt.plot(
    fpr["micro"], tpr["micro"],
    color='black', linestyle='--', lw=2,
    label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})'
)

plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Multiclass', fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
print("\n✅ ROC curves saved")

# ============================================
# 4. ERROR ANALYSIS
# ============================================
print("\n--- Error Analysis ---")

# Get misclassified samples
misclassified_idx = np.where(y_test != y_pred)[0]
print(f"Total misclassifications: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test):.2%})")

# Create error analysis DataFrame
error_df = pd.DataFrame({
    'True_Label': y_test.iloc[misclassified_idx].values,
    'Predicted_Label': y_pred[misclassified_idx],
    'Confidence': y_proba[misclassified_idx].max(axis=1)
})

# Add original test data
X_test_errors = X_test.iloc[misclassified_idx].reset_index(drop=True)
error_df = pd.concat([error_df.reset_index(drop=True), X_test_errors], axis=1)

print("\nMisclassification Patterns:")
print(error_df[['True_Label', 'Predicted_Label']].value_counts().sort_index())

# Find common error patterns
print("\nMost Confident Wrong Predictions (Top 10):")
top_errors = error_df.nlargest(10, 'Confidence')[
    ['True_Label', 'Predicted_Label', 'Confidence']
]
print(top_errors.to_string(index=False))

# Save error analysis
error_df.to_csv('misclassified_samples.csv', index=False)
print("\n✅ Misclassified samples saved: 'misclassified_samples.csv'")

# ============================================
# 5. PREDICTION CONFIDENCE DISTRIBUTION
# ============================================
print("\n--- Prediction Confidence Analysis ---")

# Get max probability for each prediction
confidence_scores = y_proba.max(axis=1)
correct_mask = y_test == y_pred

plt.figure(figsize=(12, 6))

# Correct predictions
plt.subplot(1, 2, 1)
plt.hist(confidence_scores[correct_mask], bins=30, color='green', alpha=0.7)
plt.xlabel('Confidence Score', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Confidence Distribution - Correct Predictions', fontsize=12, fontweight='bold')
plt.axvline(confidence_scores[correct_mask].mean(), color='darkgreen',
            linestyle='--', linewidth=2, label=f'Mean: {confidence_scores[correct_mask].mean():.3f}')
plt.legend()

# Incorrect predictions
plt.subplot(1, 2, 2)
plt.hist(confidence_scores[~correct_mask], bins=30, color='red', alpha=0.7)
plt.xlabel('Confidence Score', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.title('Confidence Distribution - Incorrect Predictions', fontsize=12, fontweight='bold')
plt.axvline(confidence_scores[~correct_mask].mean(), color='darkred',
            linestyle='--', linewidth=2, label=f'Mean: {confidence_scores[~correct_mask].mean():.3f}')
plt.legend()

plt.tight_layout()
plt.savefig('confidence_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Confidence distribution saved")

print(f"\nAverage confidence (correct): {confidence_scores[correct_mask].mean():.4f}")
print(f"Average confidence (incorrect): {confidence_scores[~correct_mask].mean():.4f}")

# ============================================
# 6. CLASS-WISE FEATURE ANALYSIS
# ============================================
print("\n--- Class-wise Feature Statistics ---")

# For each misclassified pattern, show avg feature values
for true_class in [0, 1, 2]:
    for pred_class in [0, 1, 2]:
        if true_class != pred_class:
            mask = (error_df['True_Label'] == true_class) & \
                   (error_df['Predicted_Label'] == pred_class)
            count = mask.sum()
            if count > 0:
                print(f"\nTrue: {true_class} → Predicted: {pred_class} (n={count})")
                # Show top 3 distinctive features
                feature_cols = X_test.columns
                avg_values = error_df.loc[mask, feature_cols].mean()
                print(f"  Top features: {avg_values.nlargest(3).to_dict()}")

# ============================================
# 7. SUMMARY REPORT
# ============================================
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

overall_accuracy = (y_test == y_pred).mean()
print(f"\n✅ Overall Accuracy: {overall_accuracy:.4f}")
print(f"✅ Macro-avg F1-Score: {report['macro avg']['f1-score']:.4f}")
print(f"✅ Weighted-avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
print(f"✅ Micro-avg ROC-AUC: {roc_auc['micro']:.4f}")

print("\n📊 Files Generated:")
print("   - confusion_matrix_detailed.png")
print("   - roc_curves.png")
print("   - confidence_distribution.png")
print("   - misclassified_samples.csv")

print("\n" + "="*70)
print("EVALUATION COMPLETE!")
print("="*70)