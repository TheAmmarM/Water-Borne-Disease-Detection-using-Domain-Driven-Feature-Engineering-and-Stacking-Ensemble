import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_recall_fscore_support
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import time
import warnings

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print("=== BASELINE MODEL TRAINING & COMPARISON ===\n")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Classes: {sorted(y_train.unique())}\n")

# ============================================
# DEFINE MODELS
# ============================================
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial'
    ),
    'Decision Tree': DecisionTreeClassifier(
        random_state=42,
        max_depth=10
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=5
    ),
    'SVM (RBF)': SVC(
        kernel='rbf',
        random_state=42,
        probability=True
    ),
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5
    ),
    'Naive Bayes': GaussianNB()
}

# ============================================
# TRAIN AND EVALUATE MODELS
# ============================================
results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=" * 70)
print("TRAINING MODELS...")
print("=" * 70)

for name, model in models.items():
    print(f"\n🔄 Training: {name}")

    # Training time
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    # Cross-validation score
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='accuracy', n_jobs=-1
    )

    # Test predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )

    results.append({
        'Model': name,
        'CV Accuracy (Mean)': cv_scores.mean(),
        'CV Accuracy (Std)': cv_scores.std(),
        'Test Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Training Time (s)': train_time
    })

    print(f"   ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   ✓ Test Accuracy: {accuracy:.4f}")
    print(f"   ✓ Training Time: {train_time:.3f}s")

# ============================================
# RESULTS COMPARISON
# ============================================
results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)

print("\n" + "=" * 70)
print("MODEL COMPARISON RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))

# ============================================
# VISUALIZATIONS
# ============================================

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Test Accuracy
axes[0, 0].barh(results_df['Model'], results_df['Test Accuracy'], color='steelblue')
axes[0, 0].set_xlabel('Accuracy', fontsize=11)
axes[0, 0].set_title('Test Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0, 0].set_xlim([0, 1])
for i, v in enumerate(results_df['Test Accuracy']):
    axes[0, 0].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# F1-Score
axes[0, 1].barh(results_df['Model'], results_df['F1-Score'], color='seagreen')
axes[0, 1].set_xlabel('F1-Score', fontsize=11)
axes[0, 1].set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
axes[0, 1].set_xlim([0, 1])
for i, v in enumerate(results_df['F1-Score']):
    axes[0, 1].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)

# Training Time
axes[1, 0].barh(results_df['Model'], results_df['Training Time (s)'], color='coral')
axes[1, 0].set_xlabel('Time (seconds)', fontsize=11)
axes[1, 0].set_title('Training Time Comparison', fontsize=13, fontweight='bold')
for i, v in enumerate(results_df['Training Time (s)']):
    axes[1, 0].text(v + 0.01, i, f'{v:.2f}s', va='center', fontsize=9)

# CV Accuracy with error bars
cv_means = results_df['CV Accuracy (Mean)']
cv_stds = results_df['CV Accuracy (Std)']
axes[1, 1].barh(results_df['Model'], cv_means, xerr=cv_stds,
                color='mediumpurple', capsize=5)
axes[1, 1].set_xlabel('Accuracy', fontsize=11)
axes[1, 1].set_title('Cross-Validation Accuracy (Mean ± Std)', fontsize=13, fontweight='bold')
axes[1, 1].set_xlim([0, 1])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Comparison chart saved: 'model_comparison.png'")

# ============================================
# BEST MODEL DETAILED ANALYSIS
# ============================================
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n{'=' * 70}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'=' * 70}")

# Re-train on full training set
best_model.fit(X_train, y_train)
y_pred_best = best_model.predict(X_test)

# Classification Report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_best,
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Poor', 'Moderate', 'Good'],
    yticklabels=['Poor', 'Moderate', 'Good'],
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('best_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✅ Confusion matrix saved: 'best_model_confusion_matrix.png'")

# Per-class performance
class_accuracy = cm.diagonal() / cm.sum(axis=1)
print("\nPer-Class Accuracy:")
for i, acc in enumerate(class_accuracy):
    print(f"   Class {i}: {acc:.4f} ({cm.diagonal()[i]}/{cm.sum(axis=1)[i]} correct)")

# ============================================
# SAVE RESULTS
# ============================================
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n✅ Results saved: 'model_comparison_results.csv'")

# Save best model
import joblib

joblib.dump(best_model, 'best_baseline_model.pkl')
print(f"✅ Best model saved: 'best_baseline_model.pkl'")

print("\n" + "=" * 70)
print("BASELINE MODELING COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
print(f"⚡ Training Time: {results_df.iloc[0]['Training Time (s)']:.3f}s")
print("\nNext Steps:")
print("1. Hyperparameter tuning for best model")
print("2. Ensemble methods (stacking/voting)")
print("3. Cross-validation with more folds")