import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print("=" * 70)
print("LIME (Local Interpretable Model-agnostic Explanations)")
print("=" * 70)
print("\nLIME explains individual predictions by approximating the model")
print("locally with an interpretable model (linear regression).\n")

# ============================================
# LOAD OR TRAIN MODEL
# ============================================
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
    print("✓ Model trained")

print(f"Model type: {type(model).__name__}")

# ============================================
# CREATE LIME EXPLAINER
# ============================================
print("\n" + "=" * 70)
print("CREATING LIME EXPLAINER")
print("=" * 70)

class_names = ['Poor (0)', 'Moderate (1)', 'Good (2)']
feature_names = X_train.columns.tolist()

print("\n🔄 Initializing LIME explainer...")
explainer = lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    random_state=42
)
print("✓ LIME explainer created")

# ============================================
# EXPLAIN INDIVIDUAL PREDICTIONS
# ============================================
print("\n" + "=" * 70)
print("EXPLAINING INDIVIDUAL PREDICTIONS")
print("=" * 70)

# Select diverse samples
np.random.seed(42)
sample_indices = []

# Get one sample from each class
for class_label in [0, 1, 2]:
    class_samples = test_df[test_df['Water Quality'] == class_label].index.tolist()
    if class_samples:
        sample_indices.append(np.random.choice(class_samples))

# Add one misclassified sample
predictions = model.predict(X_test)
misclassified = np.where(predictions != y_test)[0]
if len(misclassified) > 0:
    sample_indices.append(misclassified[0])

print(f"\nExplaining {len(sample_indices)} samples...")

all_explanations = []

for i, idx in enumerate(sample_indices):
    # Get sample
    sample = X_test.iloc[idx].values
    true_label = y_test.iloc[idx]
    pred_proba = model.predict_proba(sample.reshape(1, -1))[0]
    pred_label = np.argmax(pred_proba)

    print(f"\n--- Sample {i + 1} (Index: {idx}) ---")
    print(f"True Label: Class {true_label}")
    print(f"Predicted: Class {pred_label}")
    print(f"Confidence: {pred_proba[pred_label]:.4f}")
    print(f"All probabilities: {pred_proba}")

    # Generate LIME explanation
    print("🔄 Generating LIME explanation...")
    explanation = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=10,
        top_labels=3
    )

    # Save explanation
    all_explanations.append({
        'sample_idx': idx,
        'true_label': true_label,
        'pred_label': pred_label,
        'confidence': pred_proba[pred_label],
        'explanation': explanation
    })

    # Plot explanation for predicted class
    fig = explanation.as_pyplot_figure(label=pred_label)
    plt.title(f'LIME Explanation - Sample {i + 1}\n' +
              f'True: Class {true_label}, Predicted: Class {pred_label} ' +
              f'(Confidence: {pred_proba[pred_label]:.3f})',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'lime_explanation_sample_{i + 1}.png', dpi=300, bbox_inches='tight')
    print(f"✅ Explanation saved: lime_explanation_sample_{i + 1}.png")
    plt.close()

    # Get feature contributions
    features_contrib = explanation.as_list(label=pred_label)
    print(f"\nTop features for Class {pred_label}:")
    for feat, contrib in features_contrib[:5]:
        print(f"   {feat}: {contrib:+.4f}")

# ============================================
# AGGREGATE FEATURE IMPORTANCE
# ============================================
print("\n" + "=" * 70)
print("AGGREGATE FEATURE IMPORTANCE ACROSS SAMPLES")
print("=" * 70)

# Collect feature importance across all explanations
feature_importance_dict = {feat: [] for feat in feature_names}

for exp_data in all_explanations:
    explanation = exp_data['explanation']
    pred_label = exp_data['pred_label']

    # Get feature contributions for predicted class
    for feat_range, contrib in explanation.as_list(label=pred_label):
        # Extract feature name (LIME returns ranges like "feature <= value")
        for feat_name in feature_names:
            if feat_name in feat_range:
                feature_importance_dict[feat_name].append(abs(contrib))
                break

# Calculate average importance
avg_importance = {
    feat: np.mean(contribs) if contribs else 0
    for feat, contribs in feature_importance_dict.items()
}

importance_df = pd.DataFrame({
    'Feature': list(avg_importance.keys()),
    'Avg |Contribution|': list(avg_importance.values())
}).sort_values('Avg |Contribution|', ascending=False)

print("\nTop 10 Most Important Features (LIME):")
print(importance_df.head(10).to_string(index=False))

# Visualize
plt.figure(figsize=(12, 8))
top_n = 15
plt.barh(
    importance_df['Feature'][:top_n],
    importance_df['Avg |Contribution|'][:top_n],
    color='coral'
)
plt.xlabel('Average |Contribution|', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('LIME Feature Importance (Aggregated)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('lime_aggregate_importance.png', dpi=300, bbox_inches='tight')
print("\n✅ Aggregate importance saved: 'lime_aggregate_importance.png'")
plt.close()

# ============================================
# COMPARE PREDICTIONS ACROSS CLASSES
# ============================================
print("\n" + "=" * 70)
print("CLASS PROBABILITY EXPLANATIONS")
print("=" * 70)

# Take first sample and explain all classes
sample_idx = sample_indices[0]
sample = X_test.iloc[sample_idx].values

print(f"\nExplaining all classes for Sample (Index: {sample_idx})")

explanation = explainer.explain_instance(
    sample,
    model.predict_proba,
    num_features=10,
    top_labels=3
)

# Create subplots for all classes
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for class_idx, class_name in enumerate(class_names):
    # Get explanation for this class
    features = explanation.as_list(label=class_idx)

    # Extract feature names and values
    feat_names = [f.split('<=')[0].split('>')[0].strip() for f, _ in features]
    feat_values = [v for _, v in features]

    # Plot
    colors = ['green' if v > 0 else 'red' for v in feat_values]
    axes[class_idx].barh(feat_names, feat_values, color=colors)
    axes[class_idx].set_xlabel('Contribution', fontsize=11)
    axes[class_idx].set_title(f'{class_name}', fontsize=12, fontweight='bold')
    axes[class_idx].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[class_idx].invert_yaxis()

plt.suptitle('LIME Explanations Across All Classes (Same Sample)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('lime_all_classes_comparison.png', dpi=300, bbox_inches='tight')
print("✅ All-classes comparison saved: 'lime_all_classes_comparison.png'")
plt.close()

# ============================================
# MISCLASSIFICATION ANALYSIS
# ============================================
if len(misclassified) > 0:
    print("\n" + "=" * 70)
    print("MISCLASSIFICATION ANALYSIS")
    print("=" * 70)

    # Explain multiple misclassified samples
    n_misclassified = min(5, len(misclassified))
    print(f"\nAnalyzing {n_misclassified} misclassified samples...")

    misclass_importance = {feat: [] for feat in feature_names}

    for i, idx in enumerate(misclassified[:n_misclassified]):
        sample = X_test.iloc[idx].values
        true_label = y_test.iloc[idx]
        pred_label = predictions[idx]

        print(f"\nMisclassified {i + 1}: True={true_label}, Predicted={pred_label}")

        explanation = explainer.explain_instance(
            sample,
            model.predict_proba,
            num_features=10,
            top_labels=1
        )

        # Collect features
        for feat_range, contrib in explanation.as_list(label=pred_label):
            for feat_name in feature_names:
                if feat_name in feat_range:
                    misclass_importance[feat_name].append(abs(contrib))
                    break

    # Average misclassification importance
    misclass_avg = {
        feat: np.mean(contribs) if contribs else 0
        for feat, contribs in misclass_importance.items()
    }

    misclass_df = pd.DataFrame({
        'Feature': list(misclass_avg.keys()),
        'Avg |Contribution|': list(misclass_avg.values())
    }).sort_values('Avg |Contribution|', ascending=False)

    print("\nTop features in misclassifications:")
    print(misclass_df.head(10).to_string(index=False))

    # Compare with overall importance
    plt.figure(figsize=(14, 8))

    # Merge dataframes
    comparison = importance_df.merge(
        misclass_df,
        on='Feature',
        suffixes=('_overall', '_misclass')
    ).head(10)

    x = np.arange(len(comparison))
    width = 0.35

    plt.barh(x + width / 2, comparison['Avg |Contribution|_overall'],
             width, label='Overall', color='steelblue')
    plt.barh(x - width / 2, comparison['Avg |Contribution|_misclass'],
             width, label='Misclassified', color='coral')

    plt.ylabel('Feature', fontsize=12)
    plt.xlabel('Average |Contribution|', fontsize=12)
    plt.title('LIME: Overall vs Misclassified Feature Importance',
              fontsize=14, fontweight='bold')
    plt.yticks(x, comparison['Feature'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('lime_misclassification_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Misclassification comparison saved")
    plt.close()

# ============================================
# SAVE RESULTS
# ============================================
importance_df.to_csv('lime_feature_importance.csv', index=False)
print("\n✅ Feature importance saved: 'lime_feature_importance.csv'")

print("\n" + "=" * 70)
print("LIME ANALYSIS COMPLETE!")
print("=" * 70)
print("\n📊 Generated Visualizations:")
print("   - lime_explanation_sample_*.png")
print("   - lime_aggregate_importance.png")
print("   - lime_all_classes_comparison.png")
if len(misclassified) > 0:
    print("   - lime_misclassification_comparison.png")
print("\n💡 Key Insights:")
print(f"   Top Feature: {importance_df.iloc[0]['Feature']}")
print(f"   Avg Contribution: {importance_df.iloc[0]['Avg |Contribution|']:.4f}")