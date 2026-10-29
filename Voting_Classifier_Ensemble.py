import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score
)
import joblib
import time

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print("=" * 70)
print("VOTING CLASSIFIER ENSEMBLE")
print("=" * 70)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}\n")

# ============================================
# DEFINE BASE ESTIMATORS
# ============================================
print("--- Base Estimators ---")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

# XGBoost
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7,
    random_state=42,
    n_jobs=-1,
    eval_metric='mlogloss'
)

# Logistic Regression (for diversity)
lr = LogisticRegression(
    max_iter=1000,
    multi_class='multinomial',
    random_state=42
)

base_estimators = [
    ('rf', rf),
    ('gb', gb),
    ('xgb', xgb),
    ('lr', lr)
]

print("✓ Random Forest")
print("✓ Gradient Boosting")
print("✓ XGBoost")
print("✓ Logistic Regression")

# ============================================
# TRAIN INDIVIDUAL MODELS FIRST
# ============================================
print("\n--- Training Individual Models ---")

individual_results = []

for name, model in base_estimators:
    print(f"\n🔄 Training {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    individual_results.append({
        'Model': name.upper(),
        'Accuracy': accuracy,
        'F1-Score': f1,
        'Training Time (s)': train_time
    })

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Time: {train_time:.2f}s")

individual_df = pd.DataFrame(individual_results)

# ============================================
# HARD VOTING CLASSIFIER
# ============================================
print("\n" + "=" * 70)
print("HARD VOTING (Majority Vote)")
print("=" * 70)

voting_hard = VotingClassifier(
    estimators=base_estimators,
    voting='hard',
    n_jobs=-1
)

print("\n🔄 Training Hard Voting Classifier...")
start = time.time()
voting_hard.fit(X_train, y_train)
hard_train_time = time.time() - start

y_pred_hard = voting_hard.predict(X_test)
hard_accuracy = accuracy_score(y_test, y_pred_hard)
hard_f1 = f1_score(y_test, y_pred_hard, average='weighted')

print(f"\n✅ Hard Voting Results:")
print(f"   Accuracy: {hard_accuracy:.4f}")
print(f"   F1-Score: {hard_f1:.4f}")
print(f"   Training Time: {hard_train_time:.2f}s")

# ============================================
# SOFT VOTING CLASSIFIER
# ============================================
print("\n" + "=" * 70)
print("SOFT VOTING (Probability Average)")
print("=" * 70)

voting_soft = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    n_jobs=-1
)

print("\n🔄 Training Soft Voting Classifier...")
start = time.time()
voting_soft.fit(X_train, y_train)
soft_train_time = time.time() - start

y_pred_soft = voting_soft.predict(X_test)
soft_accuracy = accuracy_score(y_test, y_pred_soft)
soft_f1 = f1_score(y_test, y_pred_soft, average='weighted')

print(f"\n✅ Soft Voting Results:")
print(f"   Accuracy: {soft_accuracy:.4f}")
print(f"   F1-Score: {soft_f1:.4f}")
print(f"   Training Time: {soft_train_time:.2f}s")

# ============================================
# WEIGHTED SOFT VOTING (Custom Weights)
# ============================================
print("\n" + "=" * 70)
print("WEIGHTED SOFT VOTING (Based on Individual Performance)")
print("=" * 70)

# Calculate weights based on F1-scores
weights = []
for result in individual_results:
    weights.append(result['F1-Score'])

# Normalize weights
weights = np.array(weights) / np.sum(weights)
print(f"\nCalculated weights: {dict(zip([name for name, _ in base_estimators], weights))}")

voting_weighted = VotingClassifier(
    estimators=base_estimators,
    voting='soft',
    weights=weights.tolist(),
    n_jobs=-1
)

print("\n🔄 Training Weighted Soft Voting Classifier...")
start = time.time()
voting_weighted.fit(X_train, y_train)
weighted_train_time = time.time() - start

y_pred_weighted = voting_weighted.predict(X_test)
weighted_accuracy = accuracy_score(y_test, y_pred_weighted)
weighted_f1 = f1_score(y_test, y_pred_weighted, average='weighted')

print(f"\n✅ Weighted Soft Voting Results:")
print(f"   Accuracy: {weighted_accuracy:.4f}")
print(f"   F1-Score: {weighted_f1:.4f}")
print(f"   Training Time: {weighted_train_time:.2f}s")

# ============================================
# COMPARISON RESULTS
# ============================================
print("\n" + "=" * 70)
print("COMPREHENSIVE COMPARISON")
print("=" * 70)

# Add ensemble results
ensemble_results = pd.DataFrame([
    {
        'Model': 'HARD VOTING',
        'Accuracy': hard_accuracy,
        'F1-Score': hard_f1,
        'Training Time (s)': hard_train_time
    },
    {
        'Model': 'SOFT VOTING',
        'Accuracy': soft_accuracy,
        'F1-Score': soft_f1,
        'Training Time (s)': soft_train_time
    },
    {
        'Model': 'WEIGHTED SOFT VOTING',
        'Accuracy': weighted_accuracy,
        'F1-Score': weighted_f1,
        'Training Time (s)': weighted_train_time
    }
])

# Combine all results
all_results = pd.concat([individual_df, ensemble_results], ignore_index=True)
all_results = all_results.sort_values('F1-Score', ascending=False)

print("\n", all_results.to_string(index=False))

# ============================================
# VISUALIZATION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy comparison
axes[0].barh(all_results['Model'], all_results['Accuracy'], color='steelblue')
axes[0].set_xlabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
axes[0].set_xlim([0.8, 1.0])
for i, v in enumerate(all_results['Accuracy']):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

# F1-Score comparison
axes[1].barh(all_results['Model'], all_results['F1-Score'], color='seagreen')
axes[1].set_xlabel('F1-Score', fontsize=12)
axes[1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
axes[1].set_xlim([0.8, 1.0])
for i, v in enumerate(all_results['F1-Score']):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('voting_ensemble_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Comparison chart saved: 'voting_ensemble_comparison.png'")

# ============================================
# DETAILED ANALYSIS OF BEST ENSEMBLE
# ============================================
best_idx = all_results['F1-Score'].idxmax()
best_model_name = all_results.loc[best_idx, 'Model']

print(f"\n{'=' * 70}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'=' * 70}")

# Select best predictions
if best_model_name == 'HARD VOTING':
    best_predictions = y_pred_hard
    best_model = voting_hard
elif best_model_name == 'SOFT VOTING':
    best_predictions = y_pred_soft
    best_model = voting_soft
elif best_model_name == 'WEIGHTED SOFT VOTING':
    best_predictions = y_pred_weighted
    best_model = voting_weighted
else:
    # Individual model
    model_map = dict(base_estimators)
    best_model = model_map[best_model_name.lower()]
    best_predictions = best_model.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, best_predictions,
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Poor', 'Moderate', 'Good'],
    yticklabels=['Poor', 'Moderate', 'Good']
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('voting_ensemble_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved")

# ============================================
# SAVE MODELS
# ============================================
joblib.dump(voting_hard, 'voting_hard.pkl')
joblib.dump(voting_soft, 'voting_soft.pkl')
joblib.dump(voting_weighted, 'voting_weighted.pkl')

# Save best ensemble
if 'VOTING' in best_model_name:
    joblib.dump(best_model, 'best_voting_ensemble.pkl')
    print(f"\n✅ Best ensemble saved: 'best_voting_ensemble.pkl'")

# Save results
all_results.to_csv('voting_ensemble_results.csv', index=False)
print("✅ Results saved: 'voting_ensemble_results.csv'")

print("\n" + "=" * 70)
print("VOTING ENSEMBLE COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Accuracy: {all_results.loc[best_idx, 'Accuracy']:.4f}")
print(f"🎯 F1-Score: {all_results.loc[best_idx, 'F1-Score']:.4f}")