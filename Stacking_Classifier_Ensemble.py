import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold
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
print("STACKING CLASSIFIER ENSEMBLE")
print("=" * 70)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Features: {X_train.shape[1]}\n")

# ============================================
# LEVEL 0: BASE LEARNERS (Diverse Models)
# ============================================
print("--- Level 0: Base Learners ---")

base_learners = [
    ('rf', RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )),
    ('xgb', XGBClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )),
    ('svm', SVC(
        kernel='rbf',
        probability=True,
        random_state=42
    )),
    ('knn', KNeighborsClassifier(
        n_neighbors=7
    ))
]

for name, model in base_learners:
    print(f"✓ {name.upper()}: {type(model).__name__}")

# ============================================
# LEVEL 1: META-LEARNER OPTIONS
# ============================================
print("\n--- Level 1: Meta-Learners ---")

meta_learners = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        random_state=42
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric='mlogloss'
    )
}

for name in meta_learners.keys():
    print(f"✓ {name}")

# ============================================
# TRAIN BASE LEARNERS INDIVIDUALLY
# ============================================
print("\n" + "=" * 70)
print("TRAINING BASE LEARNERS INDIVIDUALLY")
print("=" * 70)

base_results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in base_learners:
    print(f"\n🔄 Training {name.upper()}...")

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')

    # Test performance
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    base_results.append({
        'Model': name.upper(),
        'CV F1-Score': cv_scores.mean(),
        'Test Accuracy': accuracy,
        'Test F1-Score': f1,
        'Training Time (s)': train_time
    })

    print(f"   CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(f"   Test F1: {f1:.4f}")

base_results_df = pd.DataFrame(base_results)
print("\n", base_results_df.to_string(index=False))

# ============================================
# STACKING WITH DIFFERENT META-LEARNERS
# ============================================
print("\n" + "=" * 70)
print("STACKING ENSEMBLE WITH DIFFERENT META-LEARNERS")
print("=" * 70)

stacking_results = []

for meta_name, meta_model in meta_learners.items():
    print(f"\n🔄 Stacking with {meta_name} as meta-learner...")

    # Create stacking classifier
    stacking = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    start = time.time()
    stacking.fit(X_train, y_train)
    train_time = time.time() - start

    # Predictions
    y_pred = stacking.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')

    stacking_results.append({
        'Meta-Learner': meta_name,
        'Test Accuracy': accuracy,
        'Test F1-Score': f1,
        'Training Time (s)': train_time
    })

    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   Training Time: {train_time:.2f}s")

stacking_results_df = pd.DataFrame(stacking_results).sort_values('Test F1-Score', ascending=False)

# ============================================
# COMPARISON: BASE vs STACKING
# ============================================
print("\n" + "=" * 70)
print("STACKING RESULTS COMPARISON")
print("=" * 70)

print("\nBase Learners (Best Individual):")
best_base = base_results_df.nlargest(1, 'Test F1-Score')
print(best_base.to_string(index=False))

print("\nStacking Ensembles:")
print(stacking_results_df.to_string(index=False))

# Calculate improvement
best_base_f1 = base_results_df['Test F1-Score'].max()
best_stack_f1 = stacking_results_df['Test F1-Score'].max()
improvement = ((best_stack_f1 - best_base_f1) / best_base_f1) * 100

print(f"\n📊 Performance Improvement:")
print(f"   Best Base Model F1: {best_base_f1:.4f}")
print(f"   Best Stacking F1: {best_stack_f1:.4f}")
print(f"   Improvement: {improvement:+.2f}%")

# ============================================
# VISUALIZATION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Base learners comparison
axes[0].barh(base_results_df['Model'], base_results_df['Test F1-Score'], color='steelblue')
axes[0].set_xlabel('F1-Score', fontsize=12)
axes[0].set_title('Base Learners Performance', fontsize=14, fontweight='bold')
axes[0].set_xlim([0.8, 1.0])
for i, v in enumerate(base_results_df['Test F1-Score']):
    axes[0].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

# Stacking comparison
axes[1].barh(stacking_results_df['Meta-Learner'], stacking_results_df['Test F1-Score'], color='seagreen')
axes[1].set_xlabel('F1-Score', fontsize=12)
axes[1].set_title('Stacking Ensemble Performance', fontsize=14, fontweight='bold')
axes[1].set_xlim([0.8, 1.0])
for i, v in enumerate(stacking_results_df['Test F1-Score']):
    axes[1].text(v + 0.005, i, f'{v:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('stacking_ensemble_comparison.png', dpi=300, bbox_inches='tight')
print("\n✅ Comparison chart saved: 'stacking_ensemble_comparison.png'")

# ============================================
# TRAIN BEST STACKING MODEL
# ============================================
best_meta_name = stacking_results_df.iloc[0]['Meta-Learner']
best_meta_model = meta_learners[best_meta_name]

print(f"\n{'=' * 70}")
print(f"🏆 TRAINING BEST STACKING MODEL")
print(f"{'=' * 70}")
print(f"Meta-Learner: {best_meta_name}\n")

best_stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=best_meta_model,
    cv=5,
    n_jobs=-1
)

print("🔄 Training final stacking model...")
best_stacking.fit(X_train, y_train)

# Predictions
y_pred_stack = best_stacking.predict(X_test)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred_stack,
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_stack)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Greens',
    xticklabels=['Poor', 'Moderate', 'Good'],
    yticklabels=['Poor', 'Moderate', 'Good']
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - Stacking ({best_meta_name})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('stacking_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved")

# ============================================
# FEATURE IMPORTANCE FROM META-LEARNER
# ============================================
if hasattr(best_stacking.final_estimator_, 'feature_importances_'):
    print("\n--- Meta-Learner: Base Model Contribution ---")

    # Get feature importances from meta-learner
    importances = best_stacking.final_estimator_.feature_importances_
    meta_feature_names = [f"{name}_class{i}" for name, _ in base_learners for i in range(len(np.unique(y_train)))]

    # Align lengths safely
    min_len = min(len(importances), len(meta_feature_names))
    importance_df = pd.DataFrame({
        'Meta Feature': meta_feature_names[:min_len],
        'Importance': importances[:min_len]
    }).sort_values('Importance', ascending=False)

    print("\n", importance_df.to_string(index=False))

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Meta Feature'], importance_df['Importance'], color='coral')
    plt.xlabel('Meta Feature (Base Model Output)', fontsize=12)
    plt.ylabel('Importance in Meta-Learner', fontsize=12)
    plt.title('Base Model Contributions to Stacking', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('stacking_base_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Base model importance saved")

# ============================================
# SAVE MODELS
# ============================================
joblib.dump(best_stacking, 'best_stacking_model.pkl')
print("\n✅ Best stacking model saved: 'best_stacking_model.pkl'")

# Save all results
base_results_df.to_csv('stacking_base_results.csv', index=False)
stacking_results_df.to_csv('stacking_meta_results.csv', index=False)
print("✅ Results saved:")
print("   - stacking_base_results.csv")
print("   - stacking_meta_results.csv")

print("\n" + "=" * 70)
print("STACKING ENSEMBLE COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Meta-Learner: {best_meta_name}")
print(f"🎯 Test Accuracy: {stacking_results_df.iloc[0]['Test Accuracy']:.4f}")
print(f"🎯 Test F1-Score: {stacking_results_df.iloc[0]['Test F1-Score']:.4f}")
print(f"📈 Improvement over best base: {improvement:+.2f}%")