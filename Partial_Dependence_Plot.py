import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import partial_dependence
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
import os

warnings.filterwarnings('ignore')

# Create directory for plots if it doesn't exist
os.makedirs('pdp_plots', exist_ok=True)

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1)
y_train = train_df['Water Quality']
X_test = test_df.drop('Water Quality', axis=1)
y_test = test_df['Water Quality']

print("=" * 70)
print("PARTIAL DEPENDENCE PLOTS (PDP)")
print("=" * 70)
print("\nPDP shows the marginal effect of features on predicted outcome")
print("while averaging out effects of all other features.\n")

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
# IDENTIFY TOP FEATURES
# ============================================
print("\n" + "=" * 70)
print("IDENTIFYING TOP FEATURES")
print("=" * 70)

# Get feature importance
if hasattr(model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)

    top_features = feature_importance.head(10)['Feature'].tolist()
    print("\nTop 10 Features by Importance:")
    print(feature_importance.head(10).to_string(index=False))
else:
    # Use all features if no feature_importances_
    top_features = X_train.columns.tolist()[:10]
    print("\nUsing first 10 features")


# ============================================
# COMPATIBLE PARTIAL DEPENDENCE FUNCTION
# ============================================
def get_partial_dependence(model, X, features, class_idx=None, grid_resolution=50):
    """
    Compatibility wrapper for partial_dependence function
    Handles different scikit-learn versions
    """
    try:
        # Try with target parameter (newer versions)
        if class_idx is not None:
            result = partial_dependence(
                model, X, features=features,
                target=class_idx, grid_resolution=grid_resolution
            )
        else:
            result = partial_dependence(
                model, X, features=features,
                grid_resolution=grid_resolution
            )
    except TypeError:
        # Fallback: without target parameter (older versions)
        result = partial_dependence(
            model, X, features=features,
            grid_resolution=grid_resolution
        )
    return result


# ============================================
# 1. INDIVIDUAL PARTIAL DEPENDENCE PLOTS
# ============================================
print("\n" + "=" * 70)
print("1. INDIVIDUAL PARTIAL DEPENDENCE PLOTS")
print("=" * 70)

class_names = ['Poor (0)', 'Moderate (1)', 'Good (2)']

# Plot top 6 features
n_features_to_plot = min(6, len(top_features))
selected_features = top_features[:n_features_to_plot]

print(f"\nCreating PDP for features: {selected_features}")

# Get feature indices
feature_indices = [X_train.columns.get_loc(feat) for feat in selected_features]

# Create PDPs for each class
for class_idx, class_name in enumerate(class_names):
    print(f"\n🔄 Generating PDP for {class_name}...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes_flat = axes.ravel()

    for i, (feat_idx, feature) in enumerate(zip(feature_indices, selected_features)):
        if i < len(axes_flat):
            try:
                pd_result = get_partial_dependence(
                    model, X_train, features=[feat_idx],
                    class_idx=class_idx, grid_resolution=50
                )

                # Handle different result structures
                if hasattr(pd_result, 'grid_values'):
                    grid_values = pd_result.grid_values[0]
                    average_values = pd_result.average[0]
                else:
                    grid_values = pd_result['grid_values'][0]
                    average_values = pd_result['average'][0]

                axes_flat[i].plot(grid_values, average_values, linewidth=2, color='blue')
                axes_flat[i].set_xlabel(feature, fontsize=10)
                axes_flat[i].set_ylabel('Partial Dependence', fontsize=10)
                axes_flat[i].set_title(f'{feature}', fontsize=11, fontweight='bold')
                axes_flat[i].grid(alpha=0.3)

            except Exception as e:
                print(f"   ⚠️ Error plotting {feature}: {str(e)}")
                axes_flat[i].text(0.5, 0.5, f'Error\n{feature}',
                                  ha='center', va='center', transform=axes_flat[i].transAxes, fontsize=10)
                axes_flat[i].set_title(f'{feature}', fontsize=11, fontweight='bold')

    # Remove empty subplots
    for i in range(len(selected_features), len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(f'Partial Dependence Plots - {class_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'pdp_plots/pdp_class_{class_idx}.png', dpi=300, bbox_inches='tight')
    plt.close()  # Explicitly close the figure
    print(f"   ✅ PDP saved: pdp_plots/pdp_class_{class_idx}.png")

# ============================================
# 2. COMBINED PDP (All Classes Together)
# ============================================
print("\n" + "=" * 70)
print("2. COMBINED PARTIAL DEPENDENCE (ALL CLASSES)")
print("=" * 70)

# Plot top 4 features with all classes
n_combined = min(4, len(top_features))
combined_features = top_features[:n_combined]

print(f"\nCreating combined PDP for: {combined_features}")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes_flat = axes.ravel()

colors = ['red', 'blue', 'green']  # Different colors for each class

for idx, feature in enumerate(combined_features):
    if idx < len(axes_flat):
        feature_idx = X_train.columns.get_loc(feature)

        for class_idx, class_name in enumerate(class_names):
            try:
                pd_result = get_partial_dependence(
                    model, X_train, features=[feature_idx],
                    class_idx=class_idx, grid_resolution=50
                )

                if hasattr(pd_result, 'grid_values'):
                    grid_values = pd_result.grid_values[0]
                    average_values = pd_result.average[0]
                else:
                    grid_values = pd_result['grid_values'][0]
                    average_values = pd_result['average'][0]

                axes_flat[idx].plot(
                    grid_values,
                    average_values,
                    label=class_name,
                    linewidth=2,
                    color=colors[class_idx]
                )
            except Exception as e:
                print(f"   ⚠️ Error in combined PDP for {feature}, class {class_name}: {str(e)}")

        axes_flat[idx].set_xlabel(feature, fontsize=11)
        axes_flat[idx].set_ylabel('Partial Dependence', fontsize=11)
        axes_flat[idx].set_title(f'PDP: {feature}', fontsize=12, fontweight='bold')
        axes_flat[idx].legend()
        axes_flat[idx].grid(alpha=0.3)

# Remove empty subplots
for idx in range(len(combined_features), len(axes_flat)):
    fig.delaxes(axes_flat[idx])

plt.suptitle('Partial Dependence Plots - All Classes', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('pdp_plots/pdp_combined_all_classes.png', dpi=300, bbox_inches='tight')
plt.close()  # Explicitly close the figure
print("✅ Combined PDP saved: 'pdp_plots/pdp_combined_all_classes.png'")

# ============================================
# 3. TWO-WAY PARTIAL DEPENDENCE (INTERACTIONS)
# ============================================
print("\n" + "=" * 70)
print("3. TWO-WAY PARTIAL DEPENDENCE (FEATURE INTERACTIONS)")
print("=" * 70)

if len(top_features) >= 2:
    top_2_features = top_features[:2]
    feature_pair = [X_train.columns.get_loc(f) for f in top_2_features]

    print(f"\nAnalyzing interaction between: {top_2_features[0]} and {top_2_features[1]}")

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for class_idx, (ax, class_name) in enumerate(zip(axes, class_names)):
        print(f"🔄 Computing 2D PDP for {class_name}...")

        try:
            pd_result = get_partial_dependence(
                model, X_train, features=feature_pair,
                class_idx=class_idx, grid_resolution=20
            )

            if hasattr(pd_result, 'grid_values'):
                Z = pd_result.average[0]
                X_grid, Y_grid = np.meshgrid(pd_result.grid_values[0], pd_result.grid_values[1])
            else:
                Z = pd_result['average'][0]
                X_grid, Y_grid = np.meshgrid(pd_result['grid_values'][0], pd_result['grid_values'][1])

            contour = ax.contourf(X_grid, Y_grid, Z, levels=15, cmap='viridis')
            plt.colorbar(contour, ax=ax)
            ax.set_xlabel(top_2_features[0])
            ax.set_ylabel(top_2_features[1])
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')

        except Exception as e:
            print(f"   ⚠️ 2D PDP failed for {class_name}: {str(e)}")
            ax.text(0.5, 0.5, f'2D PDP not available\nfor {class_name}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')

    plt.suptitle(f'2D Partial Dependence: {top_2_features[0]} vs {top_2_features[1]}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pdp_plots/pdp_2d_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()  # Explicitly close the figure
    print("✅ 2D interaction PDP saved: 'pdp_plots/pdp_2d_interaction.png'")
else:
    print("❌ Not enough features for 2D interaction plot")

# ============================================
# 4. FEATURE EFFECT COMPARISON
# ============================================
print("\n" + "=" * 70)
print("4. FEATURE EFFECT MAGNITUDE COMPARISON")
print("=" * 70)

print("\n🔄 Computing effect magnitudes...")

effect_magnitudes = []

for feature in top_features:
    feature_idx = X_train.columns.get_loc(feature)

    max_range = 0
    for class_idx in range(3):
        try:
            pd_result = get_partial_dependence(
                model, X_train, features=[feature_idx],
                class_idx=class_idx, grid_resolution=50
            )

            if hasattr(pd_result, 'average'):
                average_values = pd_result.average[0]
            else:
                average_values = pd_result['average'][0]

            pd_range = average_values.max() - average_values.min()
            max_range = max(max_range, pd_range)
        except Exception as e:
            print(f"   ⚠️ Error computing effect for {feature}, class {class_idx}: {str(e)}")
            continue

    effect_magnitudes.append({
        'Feature': feature,
        'Effect Magnitude': max_range
    })

effect_df = pd.DataFrame(effect_magnitudes).sort_values('Effect Magnitude', ascending=False)

print("\nFeature Effect Magnitudes:")
print(effect_df.to_string(index=False))

# Visualize effect magnitudes
plt.figure(figsize=(12, 8))
plt.barh(effect_df['Feature'], effect_df['Effect Magnitude'], color='teal')
plt.xlabel('Effect Magnitude (PD Range)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Effect Magnitudes from Partial Dependence', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('pdp_plots/pdp_effect_magnitudes.png', dpi=300, bbox_inches='tight')
plt.close()  # Explicitly close the figure
print("\n✅ Effect magnitudes saved: 'pdp_plots/pdp_effect_magnitudes.png'")

# ============================================
# 5. PDP SUMMARY STATISTICS
# ============================================
print("\n" + "=" * 70)
print("5. PDP SUMMARY STATISTICS")
print("=" * 70)

summary_stats = []

for feature in top_features[:5]:
    feature_idx = X_train.columns.get_loc(feature)

    for class_idx, class_name in enumerate(class_names):
        try:
            pd_result = get_partial_dependence(
                model, X_train, features=[feature_idx],
                class_idx=class_idx, grid_resolution=50
            )

            if hasattr(pd_result, 'average'):
                average_values = pd_result.average[0]
            else:
                average_values = pd_result['average'][0]

            summary_stats.append({
                'Feature': feature,
                'Class': class_name,
                'Min PD': round(average_values.min(), 4),
                'Max PD': round(average_values.max(), 4),
                'Range': round(average_values.max() - average_values.min(), 4),
                'Mean PD': round(average_values.mean(), 4),
                'Std PD': round(average_values.std(), 4)
            })
        except Exception as e:
            print(f"   ⚠️ Error computing stats for {feature}, class {class_name}: {str(e)}")
            continue

summary_df = pd.DataFrame(summary_stats)

print("\nPDP Summary Statistics (Top 5 Features):")
print(summary_df.to_string(index=False))

# Save results
summary_df.to_csv('pdp_summary_statistics.csv', index=False)
effect_df.to_csv('pdp_effect_magnitudes.csv', index=False)

print("\n✅ Summary statistics saved: 'pdp_summary_statistics.csv'")
print("✅ Effect magnitudes saved: 'pdp_effect_magnitudes.csv'")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("PARTIAL DEPENDENCE ANALYSIS COMPLETE!")
print("=" * 70)

print("\n📊 GENERATED VISUALIZATIONS:")
print("   - pdp_plots/pdp_class_[0-2].png (Individual class PDPs)")
print("   - pdp_plots/pdp_combined_all_classes.png")
print("   - pdp_plots/pdp_2d_interaction.png")
print("   - pdp_plots/pdp_effect_magnitudes.png")

print("\n📈 KEY INSIGHTS FROM PDP ANALYSIS:")

if len(effect_df) > 0:
    top_feature = effect_df.iloc[0]
    print(f"   🎯 Most influential feature: {top_feature['Feature']}")
    print(f"   📊 Effect magnitude: {top_feature['Effect Magnitude']:.4f}")

    # Additional insights based on feature importance
    print(f"\n💧 TOP 5 WATER QUALITY INDICATORS:")
    for i, (_, row) in enumerate(effect_df.head(5).iterrows(), 1):
        print(f"   {i}. {row['Feature']} (Effect: {row['Effect Magnitude']:.4f})")

    if len(top_features) >= 2:
        print(f"\n🔄 Key interaction: {top_features[0]} & {top_features[1]}")

    print(f"\n🎯 RECOMMENDATIONS:")
    print("   1. Focus monitoring on top 3 features for early detection")
    print("   2. Use PDP thresholds for setting water quality alerts")
    print("   3. Consider feature interactions in pollution events")
    print("   4. Regular calibration of top-ranking sensor parameters")

print("\n" + "=" * 70)
print("ANALYSIS SUCCESSFULLY COMPLETED!")
print("=" * 70)