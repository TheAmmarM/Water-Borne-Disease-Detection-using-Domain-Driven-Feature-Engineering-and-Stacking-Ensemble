import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
import joblib
import time

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print("=== HYPERPARAMETER TUNING ===\n")

# ============================================
# RANDOM FOREST TUNING
# ============================================
print("🔧 Tuning Random Forest...")
print("-" * 60)

rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# Use RandomizedSearchCV for faster results
rf_random_search = RandomizedSearchCV(
    rf_base,
    rf_param_grid,
    n_iter=50,  # Try 50 random combinations
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
rf_random_search.fit(X_train, y_train)
rf_tune_time = time.time() - start_time

print(f"\n✅ Random Forest tuning completed in {rf_tune_time:.2f}s")
print(f"\n🏆 Best Parameters:")
for param, value in rf_random_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n🎯 Best CV Score (F1-weighted): {rf_random_search.best_score_:.4f}")

# Test on holdout set
rf_best = rf_random_search.best_estimator_
y_pred_rf = rf_best.predict(X_test)
rf_test_score = f1_score(y_test, y_pred_rf, average='weighted')
print(f"🎯 Test F1-Score: {rf_test_score:.4f}")

# ============================================
# GRADIENT BOOSTING TUNING
# ============================================
print("\n" + "=" * 60)
print("🔧 Tuning Gradient Boosting...")
print("-" * 60)

gb_param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0],
    'max_features': ['sqrt', 'log2', None]
}

gb_base = GradientBoostingClassifier(random_state=42)

gb_random_search = RandomizedSearchCV(
    gb_base,
    gb_param_grid,
    n_iter=50,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1,
    random_state=42
)

start_time = time.time()
gb_random_search.fit(X_train, y_train)
gb_tune_time = time.time() - start_time

print(f"\n✅ Gradient Boosting tuning completed in {gb_tune_time:.2f}s")
print(f"\n🏆 Best Parameters:")
for param, value in gb_random_search.best_params_.items():
    print(f"   {param}: {value}")
print(f"\n🎯 Best CV Score (F1-weighted): {gb_random_search.best_score_:.4f}")

# Test on holdout set
gb_best = gb_random_search.best_estimator_
y_pred_gb = gb_best.predict(X_test)
gb_test_score = f1_score(y_test, y_pred_gb, average='weighted')
print(f"🎯 Test F1-Score: {gb_test_score:.4f}")

# ============================================
# COMPARISON: BEFORE vs AFTER TUNING
# ============================================
print("\n" + "=" * 60)
print("TUNING RESULTS SUMMARY")
print("=" * 60)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'Gradient Boosting'],
    'Best CV F1-Score': [rf_random_search.best_score_, gb_random_search.best_score_],
    'Test F1-Score': [rf_test_score, gb_test_score],
    'Tuning Time (s)': [rf_tune_time, gb_tune_time]
})

print("\n", comparison.to_string(index=False))

# Select best overall model
best_idx = comparison['Test F1-Score'].idxmax()
best_model = rf_best if best_idx == 0 else gb_best
best_model_name = comparison.loc[best_idx, 'Model']

print(f"\n{'=' * 60}")
print(f"🏆 CHAMPION MODEL: {best_model_name}")
print(f"{'=' * 60}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(
    y_test,
    best_model.predict(X_test),
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# ============================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {best_model_name}',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('tuned_model_feature_importance.png', dpi=300, bbox_inches='tight')
    print("\n✅ Feature importance plot saved")

# ============================================
# SAVE TUNED MODELS
# ============================================
joblib.dump(rf_best, 'tuned_random_forest.pkl')
joblib.dump(gb_best, 'tuned_gradient_boosting.pkl')
joblib.dump(best_model, 'final_champion_model.pkl')

# Save hyperparameters
with open('best_hyperparameters.txt', 'w') as f:
    f.write(f"CHAMPION MODEL: {best_model_name}\n")
    f.write("=" * 60 + "\n\n")
    f.write("Random Forest Best Parameters:\n")
    for param, value in rf_random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nRandom Forest CV F1-Score: {rf_random_search.best_score_:.4f}\n")
    f.write(f"Random Forest Test F1-Score: {rf_test_score:.4f}\n\n")

    f.write("Gradient Boosting Best Parameters:\n")
    for param, value in gb_random_search.best_params_.items():
        f.write(f"  {param}: {value}\n")
    f.write(f"\nGradient Boosting CV F1-Score: {gb_random_search.best_score_:.4f}\n")
    f.write(f"Gradient Boosting Test F1-Score: {gb_test_score:.4f}\n")

print("\n✅ Models saved:")
print("   - tuned_random_forest.pkl")
print("   - tuned_gradient_boosting.pkl")
print("   - final_champion_model.pkl")
print("   - best_hyperparameters.txt")

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING COMPLETE!")
print("=" * 60)
print("\nNext Steps:")
print("1. Ensemble modeling (Voting/Stacking)")
print("2. Error analysis on misclassified samples")
print("3. Model deployment preparation")