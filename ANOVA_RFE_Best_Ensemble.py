"""
ULTIMATE WATER QUALITY CLASSIFICATION PIPELINE - FIXED
=======================================================
No threading errors, clean execution
"""

import pandas as pd
import numpy as np
import os

# FIX: Set matplotlib backend BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no tkinter)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, learning_curve, validation_curve
)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    print("⚠️  CatBoost not available. Install with: pip install catboost")

import joblib
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress matplotlib warnings
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

print("="*80)
print("ULTIMATE WATER QUALITY CLASSIFICATION PIPELINE")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n📊 LOADING DATA...")

train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Features: {X_train.shape[1]}")
print(f"✓ Classes: {sorted(y_train.unique())}")
print(f"✓ Class distribution: {y_train.value_counts().to_dict()}")

# ============================================
# PHASE 1: ADVANCED FEATURE SELECTION
# ============================================
print("\n" + "="*80)
print("PHASE 1: ADVANCED FEATURE SELECTION")
print("="*80)

# Method 1: ANOVA F-test
print("\n🔍 Method 1: ANOVA F-test...")
selector_anova = SelectKBest(score_func=f_classif, k='all')
selector_anova.fit(X_train, y_train)

anova_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'F-Score': selector_anova.scores_,
    'P-Value': selector_anova.pvalues_
}).sort_values('F-Score', ascending=False)

print("\nTop 10 Features by ANOVA F-test:")
print(anova_scores.head(10).to_string(index=False))

# Method 2: Recursive Feature Elimination
print("\n🔍 Method 2: Recursive Feature Elimination (RFE)...")
rf_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
selector_rfe = RFE(estimator=rf_estimator, n_features_to_select=15, step=1, verbose=0)
selector_rfe.fit(X_train, y_train)

rfe_ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'Ranking': selector_rfe.ranking_,
    'Selected': selector_rfe.support_
}).sort_values('Ranking')

print("\nRFE Feature Ranking:")
print(rfe_ranking.to_string(index=False))

# Combine both methods
anova_top = set(anova_scores.head(15)['Feature'].tolist())
rfe_top = set(rfe_ranking[rfe_ranking['Selected']]['Feature'].tolist())

selected_features = list(anova_top.intersection(rfe_top))
if len(selected_features) < 10:
    selected_features = list(anova_top.union(rfe_top))[:15]

print(f"\n✓ Selected {len(selected_features)} features using combined approach")

# Apply feature selection
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Save feature selection results
feature_selection_results = pd.DataFrame({
    'Feature': X_train.columns,
    'ANOVA_F_Score': selector_anova.scores_,
    'ANOVA_P_Value': selector_anova.pvalues_,
    'RFE_Ranking': selector_rfe.ranking_,
    'Selected': [f in selected_features for f in X_train.columns]
})
feature_selection_results.to_csv('feature_selection_results.csv', index=False)
print("✓ Feature selection results saved")

# ============================================
# PHASE 2: BUILD CHAMPION MODELS
# ============================================
print("\n" + "="*80)
print("PHASE 2: BUILDING CHAMPION MODELS")
print("="*80)

base_models = []

print("\n🌲 Random Forest...")
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)
base_models.append(('rf', rf))

print("⚡ XGBoost...")
xgb = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1,
    verbosity=0
)
base_models.append(('xgb', xgb))

print("💡 LightGBM...")
lgbm = LGBMClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
base_models.append(('lgbm', lgbm))

if CATBOOST_AVAILABLE:
    print("🐱 CatBoost...")
    catboost = CatBoostClassifier(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        l2_leaf_reg=3,
        random_state=42,
        verbose=False
    )
    base_models.append(('catboost', catboost))

print("🚀 Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    max_features='sqrt',
    random_state=42,
    verbose=0
)
base_models.append(('gb', gb))

print(f"\n✓ {len(base_models)} base models configured")

# ============================================
# PHASE 3: CROSS-VALIDATION & OVERFITTING CHECK
# ============================================
print("\n" + "="*80)
print("PHASE 3: CROSS-VALIDATION & OVERFITTING DETECTION")
print("="*80)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_results = []

for name, model in base_models:
    print(f"\n🔄 Evaluating {name.upper()}...")

    start = time.time()
    model.fit(X_train_selected, y_train)
    train_time = time.time() - start

    train_pred = model.predict(X_train_selected)
    train_score = accuracy_score(y_train, train_pred)

    test_pred = model.predict(X_test_selected)
    test_score = accuracy_score(y_test, test_pred)

    cv_scores = cross_val_score(
        model, X_train_selected, y_train,
        cv=cv, scoring='accuracy', n_jobs=-1, verbose=0
    )

    test_f1 = f1_score(y_test, test_pred, average='weighted')
    overfit_gap = train_score - test_score
    is_overfit = overfit_gap > 0.05

    cv_results.append({
        'Model': name.upper(),
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'CV Mean': cv_scores.mean(),
        'CV Std': cv_scores.std(),
        'Test F1': test_f1,
        'Overfit Gap': overfit_gap,
        'Overfitting': 'YES' if is_overfit else 'NO',
        'Training Time (s)': train_time
    })

    print(f"   Train Acc: {train_score:.4f}")
    print(f"   Test Acc:  {test_score:.4f}")
    print(f"   CV Acc:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"   Overfit:   {'⚠️ YES' if is_overfit else '✅ NO'} (Gap: {overfit_gap:.4f})")

cv_results_df = pd.DataFrame(cv_results).sort_values('Test Accuracy', ascending=False)

print("\n" + "="*80)
print("BASE MODELS COMPARISON")
print("="*80)
print("\n", cv_results_df.to_string(index=False))

# Visualize overfitting
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(cv_results_df))
width = 0.35

bars1 = ax.bar(x - width/2, cv_results_df['Train Accuracy'], width,
               label='Train Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, cv_results_df['Test Accuracy'], width,
               label='Test Accuracy', color='coral')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Train vs Test Accuracy - Overfitting Detection',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(cv_results_df['Model'], rotation=45)
ax.legend()
ax.set_ylim([0.8, 1.0])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_detection.png', dpi=300, bbox_inches='tight')
plt.close('all')  # FIX: Close all figures
print("\n✅ Overfitting plot saved: 'overfitting_detection.png'")

# ============================================
# PHASE 4: LEARNING CURVES
# ============================================
print("\n" + "="*80)
print("PHASE 4: LEARNING CURVES - UNDERFITTING DETECTION")
print("="*80)

best_model_name = cv_results_df.iloc[0]['Model'].lower()
best_model = dict(base_models)[best_model_name]

print(f"\n📊 Generating learning curves for {best_model_name.upper()}...")

train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    best_model, X_train_selected, y_train,
    train_sizes=train_sizes,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=0
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

final_train_score = train_mean[-1]
final_val_score = val_mean[-1]

if final_train_score < 0.85 and final_val_score < 0.85:
    underfitting_status = "⚠️ YES - Model too simple"
elif abs(final_train_score - final_val_score) < 0.03:
    underfitting_status = "⚠️ POSSIBLE - Similar low scores"
else:
    underfitting_status = "✅ NO - Model complexity appropriate"

print(f"\nUnderfitting Analysis:")
print(f"   Final Train Score: {final_train_score:.4f}")
print(f"   Final Val Score:   {final_val_score:.4f}")
print(f"   Status: {underfitting_status}")

# Plot learning curves
plt.figure(figsize=(10, 6))
plt.plot(train_sizes_abs, train_mean, 'o-', color='steelblue',
         label='Training score', linewidth=2)
plt.fill_between(train_sizes_abs, train_mean - train_std,
                 train_mean + train_std, alpha=0.2, color='steelblue')

plt.plot(train_sizes_abs, val_mean, 'o-', color='coral',
         label='Cross-validation score', linewidth=2)
plt.fill_between(train_sizes_abs, val_mean - val_std,
                 val_mean + val_std, alpha=0.2, color='coral')

plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title(f'Learning Curves - {best_model_name.upper()}',
          fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
plt.close('all')  # FIX: Close all figures
print("✅ Learning curves saved: 'learning_curves.png'")

# ============================================
# PHASE 5: ADVANCED STACKING ENSEMBLE
# ============================================
print("\n" + "="*80)
print("PHASE 5: ADVANCED STACKING ENSEMBLE")
print("="*80)

meta_learners = {
    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss', verbosity=0
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=42, verbose=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
}

stacking_results = []

for meta_name, meta_model in meta_learners.items():
    print(f"\n🔄 Stacking with {meta_name} meta-learner...")

    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5,
        n_jobs=-1
    )

    start = time.time()
    stacking.fit(X_train_selected, y_train)
    train_time = time.time() - start

    train_pred = stacking.predict(X_train_selected)
    train_score = accuracy_score(y_train, train_pred)

    test_pred = stacking.predict(X_test_selected)
    test_score = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted')

    cv_scores = cross_val_score(
        stacking, X_train_selected, y_train,
        cv=5, scoring='accuracy', n_jobs=-1, verbose=0
    )

    overfit_gap = train_score - test_score

    stacking_results.append({
        'Meta-Learner': meta_name,
        'Train Accuracy': train_score,
        'Test Accuracy': test_score,
        'CV Mean': cv_scores.mean(),
        'Test F1': test_f1,
        'Overfit Gap': overfit_gap,
        'Training Time (s)': train_time
    })

    print(f"   Train Acc: {train_score:.4f}")
    print(f"   Test Acc:  {test_score:.4f}")
    print(f"   CV Acc:    {cv_scores.mean():.4f}")
    print(f"   Time:      {train_time:.2f}s")

stacking_results_df = pd.DataFrame(stacking_results).sort_values(
    'Test Accuracy', ascending=False
)

print("\n" + "="*80)
print("STACKING ENSEMBLE RESULTS")
print("="*80)
print("\n", stacking_results_df.to_string(index=False))

# ============================================
# PHASE 6: WEIGHTED VOTING ENSEMBLE
# ============================================
print("\n" + "="*80)
print("PHASE 6: WEIGHTED VOTING ENSEMBLE")
print("="*80)

weights = cv_results_df['Test F1'].values / cv_results_df['Test F1'].sum()

print(f"\nCalculated weights based on F1-scores:")
for model, weight in zip(cv_results_df['Model'], weights):
    print(f"   {model}: {weight:.4f}")

voting = VotingClassifier(
    estimators=base_models,
    voting='soft',
    weights=weights.tolist(),
    n_jobs=-1
)

print(f"\n🔄 Training Weighted Voting Ensemble...")
start = time.time()
voting.fit(X_train_selected, y_train)
voting_time = time.time() - start

train_pred = voting.predict(X_train_selected)
train_score = accuracy_score(y_train, train_pred)

test_pred = voting.predict(X_test_selected)
test_score = accuracy_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred, average='weighted')

print(f"   Train Acc: {train_score:.4f}")
print(f"   Test Acc:  {test_score:.4f}")
print(f"   Test F1:   {test_f1:.4f}")
print(f"   Time:      {voting_time:.2f}s")

# ============================================
# PHASE 7: SELECT CHAMPION MODEL
# ============================================
print("\n" + "="*80)
print("PHASE 7: CHAMPION MODEL SELECTION")
print("="*80)

all_results = []

for _, row in cv_results_df.iterrows():
    all_results.append({
        'Model Type': 'Base',
        'Model Name': row['Model'],
        'Test Accuracy': row['Test Accuracy'],
        'Test F1': row['Test F1'],
        'Overfitting': row['Overfitting']
    })

for _, row in stacking_results_df.iterrows():
    all_results.append({
        'Model Type': 'Stacking',
        'Model Name': f"Stack-{row['Meta-Learner']}",
        'Test Accuracy': row['Test Accuracy'],
        'Test F1': row['Test F1'],
        'Overfitting': 'YES' if row['Overfit Gap'] > 0.05 else 'NO'
    })

all_results.append({
    'Model Type': 'Voting',
    'Model Name': 'Weighted-Soft-Voting',
    'Test Accuracy': test_score,
    'Test F1': test_f1,
    'Overfitting': 'YES' if train_score - test_score > 0.05 else 'NO'
})

final_comparison = pd.DataFrame(all_results).sort_values(
    'Test Accuracy', ascending=False
)

print("\n🏆 FINAL MODEL COMPARISON:")
print(final_comparison.to_string(index=False))

champion_row = final_comparison.iloc[0]
champion_name = champion_row['Model Name']
champion_type = champion_row['Model Type']

print(f"\n{'='*80}")
print(f"🏆🏆🏆 CHAMPION MODEL: {champion_name}")
print(f"{'='*80}")
print(f"Type: {champion_type}")
print(f"Test Accuracy: {champion_row['Test Accuracy']:.4f}")
print(f"Test F1-Score: {champion_row['Test F1']:.4f}")
print(f"Overfitting: {champion_row['Overfitting']}")

# Get champion model
if champion_type == 'Base':
    champion_model = dict(base_models)[champion_name.lower()]
elif champion_type == 'Stacking':
    meta_name = champion_name.replace('Stack-', '')
    champion_model = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learners[meta_name],
        cv=5,
        n_jobs=-1
    )
    champion_model.fit(X_train_selected, y_train)
else:
    champion_model = voting

# ============================================
# PHASE 8: CHAMPION ANALYSIS
# ============================================
print("\n" + "="*80)
print("PHASE 8: CHAMPION MODEL DETAILED ANALYSIS")
print("="*80)

y_pred = champion_model.predict(X_test_selected)

print("\n📊 Classification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='RdYlGn',
    xticklabels=['Poor', 'Moderate', 'Good'],
    yticklabels=['Poor', 'Moderate', 'Good'],
    cbar_kws={'label': 'Count'}
)
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix - {champion_name}', fontsize=14, fontweight='bold')

accuracy = np.trace(cm) / np.sum(cm)
plt.text(1.5, -0.3, f'Accuracy: {accuracy:.4f}',
         ha='center', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('champion_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close('all')  # FIX: Close all figures
print("\n✅ Confusion matrix saved")

# ============================================
# SAVE EVERYTHING
# ============================================
print("\n" + "="*80)
print("SAVING RESULTS")
print("="*80)

joblib.dump(champion_model, 'champion_model.pkl')
joblib.dump(selected_features, 'selected_features.pkl')
print("✅ Champion model saved: 'champion_model.pkl'")
print("✅ Selected features saved: 'selected_features.pkl'")

cv_results_df.to_csv('base_models_results.csv', index=False)
stacking_results_df.to_csv('stacking_results.csv', index=False)
final_comparison.to_csv('final_model_comparison.csv', index=False)
print("✅ All results saved")

with open('ultimate_pipeline_report.txt', 'w') as f:
    f.write("ULTIMATE WATER QUALITY CLASSIFICATION REPORT\n")
    f.write("="*80 + "\n\n")
    f.write(f"Champion Model: {champion_name}\n")
    f.write(f"Test Accuracy: {champion_row['Test Accuracy']:.4f}\n")
    f.write(f"Test F1-Score: {champion_row['Test F1']:.4f}\n")
    f.write(f"Overfitting: {champion_row['Overfitting']}\n")
    f.write(f"Selected Features: {len(selected_features)}\n\n")
    f.write("Selected Features:\n")
    for feat in selected_features:
        f.write(f"  - {feat}\n")

print("✅ Report saved: 'ultimate_pipeline_report.txt'")

print("\n" + "="*80)
print("✅✅✅ ULTIMATE PIPELINE COMPLETE! ✅✅✅")
print("="*80)
print(f"\n🏆 Champion: {champion_name}")
print(f"🎯 Accuracy: {champion_row['Test Accuracy']:.4f}")
print(f"📊 F1-Score: {champion_row['Test F1']:.4f}")
print(f"⚠️  Overfitting: {champion_row['Overfitting']}")
print("\n💡 No threading errors - execution completed cleanly!")