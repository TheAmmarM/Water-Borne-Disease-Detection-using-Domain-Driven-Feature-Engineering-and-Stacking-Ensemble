import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Load scaled training data
train_df = pd.read_csv('train_scaled.csv')
X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']

print("=== CORRELATION ANALYSIS & FEATURE SELECTION ===\n")

# ============================================
# 1. CORRELATION MATRIX
# ============================================
print("--- 1. CORRELATION ANALYSIS ---")
correlation_matrix = X_train.corr()

# Find highly correlated pairs (> 0.85)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.85:
            high_corr_pairs.append({
                'Feature 1': correlation_matrix.columns[i],
                'Feature 2': correlation_matrix.columns[j],
                'Correlation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    print("\n🚨 Highly correlated features (|r| > 0.85):")
    for pair in high_corr_pairs:
        print(f"   {pair['Feature 1']} <-> {pair['Feature 2']}: {pair['Correlation']:.3f}")
else:
    print("\n✅ No highly correlated features found (threshold: 0.85)")

# Visualize correlation matrix
plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8}
)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Correlation matrix saved: 'correlation_matrix.png'")

# ============================================
# 2. RANDOM FOREST FEATURE IMPORTANCE
# ============================================
print("\n--- 2. RANDOM FOREST FEATURE IMPORTANCE ---")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10,
    n_jobs=-1
)
rf.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_rf.png', dpi=300, bbox_inches='tight')
print("✅ Feature importance plot saved: 'feature_importance_rf.png'")

# ============================================
# 3. MUTUAL INFORMATION
# ============================================
print("\n--- 3. MUTUAL INFORMATION (Feature-Target Dependency) ---")

mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    'Feature': X_train.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

print("\nTop 10 Features by Mutual Information:")
print(mi_df.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 8))
plt.barh(mi_df['Feature'], mi_df['MI_Score'])
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Mutual Information with Target', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('mutual_information.png', dpi=300, bbox_inches='tight')
print("✅ Mutual information plot saved: 'mutual_information.png'")

# ============================================
# 4. COMBINED RANKING
# ============================================
print("\n--- 4. COMBINED FEATURE RANKING ---")

# Normalize scores to 0-1 range
feature_importance['Importance_Norm'] = (
        feature_importance['Importance'] / feature_importance['Importance'].max()
)
mi_df['MI_Score_Norm'] = mi_df['MI_Score'] / mi_df['MI_Score'].max()

# Merge rankings
combined = feature_importance.merge(mi_df, on='Feature')
combined['Combined_Score'] = (
                                     combined['Importance_Norm'] + combined['MI_Score_Norm']
                             ) / 2
combined = combined.sort_values('Combined_Score', ascending=False)

print("\nTop 10 Features (Combined Ranking):")
print(combined[['Feature', 'Combined_Score', 'Importance', 'MI_Score']].head(10).to_string(index=False))

# ============================================
# 5. FEATURE SELECTION RECOMMENDATIONS
# ============================================
print("\n" + "=" * 60)
print("FEATURE SELECTION RECOMMENDATIONS")
print("=" * 60)

# Method 1: Keep features with combined score > threshold
threshold = 0.3
selected_features_threshold = combined[combined['Combined_Score'] > threshold]['Feature'].tolist()
print(f"\nMethod 1: Features with Combined Score > {threshold}")
print(f"Selected: {len(selected_features_threshold)} features")
print(f"Features: {selected_features_threshold}")

# Method 2: Keep top K features
k = 12
selected_features_topk = combined.head(k)['Feature'].tolist()
print(f"\nMethod 2: Top {k} Features")
print(f"Features: {selected_features_topk}")

# Method 3: Remove highly correlated features
# Keep the one with higher importance
features_to_remove = []
if high_corr_pairs:
    print("\nMethod 3: Remove Redundant Correlated Features")
    for pair in high_corr_pairs:
        feat1_score = combined[combined['Feature'] == pair['Feature 1']]['Combined_Score'].values[0]
        feat2_score = combined[combined['Feature'] == pair['Feature 2']]['Combined_Score'].values[0]

        if feat1_score > feat2_score:
            features_to_remove.append(pair['Feature 2'])
            print(f"   Remove: {pair['Feature 2']} (keep {pair['Feature 1']})")
        else:
            features_to_remove.append(pair['Feature 1'])
            print(f"   Remove: {pair['Feature 1']} (keep {pair['Feature 2']})")

    final_features = [f for f in X_train.columns if f not in features_to_remove]
else:
    final_features = X_train.columns.tolist()

print(f"\nFinal Recommended Features: {len(final_features)}")
print(f"Features: {final_features}")

# ============================================
# 6. SAVE RESULTS
# ============================================
# Save feature rankings
combined.to_csv('feature_rankings.csv', index=False)
print("\n✅ Feature rankings saved: 'feature_rankings.csv'")

# Save selected features list
with open('selected_features.txt', 'w') as f:
    f.write("RECOMMENDED FEATURES FOR MODELING\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Total features selected: {len(final_features)}\n\n")
    for feat in final_features:
        f.write(f"- {feat}\n")

print("✅ Selected features saved: 'selected_features.txt'")

print("\n" + "=" * 60)
print("CORRELATION ANALYSIS COMPLETE!")
print("=" * 60)
print("\nNext Step: Use 'final_features' list for model training")