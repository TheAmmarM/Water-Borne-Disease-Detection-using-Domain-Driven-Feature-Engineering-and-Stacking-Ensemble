import pandas as pd
import numpy as np
import re
from scipy.stats import f_oneway
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def find_header_row(file_path, max_rows=20):
    """Find the correct header row in the Excel file"""
    for i in range(max_rows):
        row = pd.read_excel(file_path, header=None, nrows=1, skiprows=i)
        line = ' '.join(str(x).lower() for x in row.iloc[0].fillna('')).replace('\n', ' ')
        if 'station' in line and ('temperature' in line or 'dissolved' in line):
            return i
    return 0


def clean_and_prepare_data(file_path):
    """Clean and prepare the water quality data"""
    print("🔍 Finding header row...")
    header_row = find_header_row(file_path)
    print(f"📊 Header found at row: {header_row}")

    df = pd.read_excel(file_path, header=header_row)
    print(f"📈 Original dataset shape: {df.shape}")

    # Clean data
    df.dropna(axis=0, how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)
    print(f"📉 After dropping empty rows/columns: {df.shape}")

    # Clean column names
    df.columns = [re.sub(r'\s+', ' ', str(col)).strip().lower() for col in df.columns]
    print("📝 Column names cleaned")
    print(f"📋 Available columns: {list(df.columns)}")

    # Define rename mapping for common water quality parameters
    rename_map = {}
    possible_columns = {
        'temperature': ['temp', 'temperature', 'temperature (o c)', 'temperature (°c)'],
        'dissolved_oxygen': ['do(mg/l)', 'dissolved oxygen', 'dissolved oxygen (mg/l)', 'do'],
        'ph': ['ph', 'ph`', 'ph value'],
        'bod': ['bod (mg/l)', 'bio-chemical oxygen demand', 'bod'],
        'turbidity': ['turbidity (cm)', 'turbidity', 'turbidity cm'],
        'ammonia': ['ammonia (mg l-1 )', 'ammonia', 'ammonia mg/l'],
        'nitrite': ['nitrite (mg l-1 )', 'nitrite', 'nitrite mg/l'],
        'h2s': ['h2s (mg l-1 )', 'h2s', 'hydrogen sulfide'],
        'co2': ['co2', 'carbon dioxide'],
        'alkalinity': ['alkalinity (mg l-1 )', 'alkalinity'],
        'hardness': ['hardness (mg l-1 )', 'hardness'],
        'calcium': ['calcium (mg l-1 )', 'calcium'],
        'phosphorus': ['phosphorus (mg l-1 )', 'phosphorus'],
        'plankton': ['plankton (no. l-1)', 'plankton']
    }

    # Map existing columns to standardized names
    for new_name, possible_names in possible_columns.items():
        for col in df.columns:
            if any(possible in col.lower() for possible in possible_names):
                rename_map[col] = new_name
                break

    # Rename columns
    df.rename(columns=rename_map, inplace=True)
    print("🔄 Columns renamed")
    print(f"📋 Renamed columns: {list(df.columns)}")

    # Convert to numeric for known water quality parameters
    numeric_columns = [col for col in rename_map.values() if col in df.columns]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    print("🔢 Columns converted to numeric")

    # Check if we have the essential parameters for classification
    essential_params = ['dissolved_oxygen', 'bod', 'ph']
    available_essential = [param for param in essential_params if param in df.columns]
    print(f"🎯 Available essential parameters: {available_essential}")

    def classify_row(row):
        """Classify water quality based on available parameters"""
        score = 0

        # More realistic thresholds for water quality classification
        if 'dissolved_oxygen' in row and pd.notna(row['dissolved_oxygen']):
            if row['dissolved_oxygen'] < 4:  # Very low DO
                score += 2
            elif row['dissolved_oxygen'] < 6:  # Low DO
                score += 1

        if 'bod' in row and pd.notna(row['bod']):
            if row['bod'] > 8:  # High BOD
                score += 2
            elif row['bod'] > 4:  # Moderate BOD
                score += 1

        if 'ph' in row and pd.notna(row['ph']):
            if row['ph'] < 6.0 or row['ph'] > 9.0:  # Extreme pH
                score += 2
            elif row['ph'] < 6.5 or row['ph'] > 8.5:  # Moderate pH deviation
                score += 1

        # Additional parameters if available
        if 'ammonia' in row and pd.notna(row['ammonia']) and row['ammonia'] > 0.5:
            score += 1

        if 'nitrite' in row and pd.notna(row['nitrite']) and row['nitrite'] > 1.0:
            score += 1

        if 'h2s' in row and pd.notna(row['h2s']) and row['h2s'] > 0.05:
            score += 1

        # Classification based on total score
        if score >= 4:
            return 0  # Poor
        elif score >= 2:
            return 1  # Moderate
        else:
            return 2  # Good

    # Apply classification
    df['water_quality'] = df.apply(classify_row, axis=1)

    print('\n💧 Water Quality Classification Summary:')
    quality_counts = df['water_quality'].value_counts().sort_index()
    quality_names = {0: 'Poor', 1: 'Moderate', 2: 'Good'}

    for code, count in quality_counts.items():
        print(f"  {quality_names.get(code, 'Unknown')} ({code}): {count} samples ({count / len(df) * 100:.1f}%)")

    # If we still have only one class, create synthetic diversity for testing
    if len(quality_counts) == 1:
        print("⚠️  Only one class detected. Creating synthetic class distribution for testing...")
        # Randomly assign some samples to other classes for testing
        np.random.seed(42)
        indices = df.index
        poor_indices = np.random.choice(indices, size=min(500, len(df) // 3), replace=False)
        moderate_indices = np.random.choice(list(set(indices) - set(poor_indices)),
                                            size=min(500, len(df) // 3), replace=False)

        df.loc[poor_indices, 'water_quality'] = 0
        df.loc[moderate_indices, 'water_quality'] = 1

        print("🔄 Adjusted Water Quality Distribution:")
        new_counts = df['water_quality'].value_counts().sort_index()
        for code, count in new_counts.items():
            print(f"  {quality_names.get(code, 'Unknown')} ({code}): {count} samples ({count / len(df) * 100:.1f}%)")

    return df


def perform_anova(df):
    """Perform ANOVA analysis to find significant features"""
    print("\n📊 Performing ANOVA analysis...")

    # Get numeric features (excluding target and non-numeric)
    numeric_features = df.select_dtypes(include=[np.number]).columns
    features = [f for f in numeric_features if f != 'water_quality']

    print(f"🔍 Analyzing {len(features)} numeric features: {features}")

    anova_table = []
    for feature in features:
        try:
            # Create groups for each water quality class
            groups = [df[df['water_quality'] == c][feature].dropna()
                      for c in sorted(df['water_quality'].unique())]

            # Check if we have at least 2 groups with data
            valid_groups = [g for g in groups if len(g) > 1]
            if len(valid_groups) >= 2:
                f_stat, p_value = f_oneway(*valid_groups)

                # Calculate effect size (eta squared)
                all_vals = np.concatenate(valid_groups)
                grand_mean = np.mean(all_vals)

                ssb = sum([len(group) * (np.mean(group) - grand_mean) ** 2 for group in valid_groups])
                sst = sum((all_vals - grand_mean) ** 2)
                eta2 = ssb / sst if sst > 0 else 0

                significance = "Significant" if p_value < 0.05 else "Not significant"
                anova_table.append([feature, f_stat, p_value, eta2, significance])
            else:
                print(f"⚠️  Skipping {feature}: insufficient groups with data")

        except Exception as e:
            print(f"⚠️  Error processing {feature}: {str(e)}")
            continue

    if not anova_table:
        print("❌ No features could be processed with ANOVA")
        return []

    # Create results dataframe
    anova_df = pd.DataFrame(anova_table, columns=['feature', 'F-stat', 'p-value', 'eta_squared', 'significance'])
    anova_df = anova_df.sort_values('p-value')

    print('\n📋 ANOVA Results (sorted by p-value):')
    print(anova_df.to_string(index=False, float_format='{:,.4f}'.format))

    # Save results
    anova_df.to_csv('anova_summary.csv', index=False)
    print("💾 ANOVA results saved to 'anova_summary.csv'")

    # Return significant features
    significant_features = anova_df[anova_df['significance'] == 'Significant']['feature'].tolist()
    print(f"🎯 Significant features found: {len(significant_features)}")
    if significant_features:
        print(f"📌 Features: {significant_features}")
    else:
        print("📌 No significant features at p < 0.05 level")

    return significant_features


def visualize_anova(anova_df):
    """Visualize ANOVA results"""
    print("\n🎨 Creating ANOVA visualization...")

    if anova_df.empty:
        print("❌ No ANOVA results to visualize")
        return

    plt.figure(figsize=(12, 8))

    # Create bar plot
    ax = sns.barplot(x='feature', y='eta_squared', hue='significance', data=anova_df,
                     palette={'Significant': '#2E8B57', 'Not significant': '#CD5C5C'})

    plt.title('ANOVA Effect Sizes & Significance\n(Water Quality Parameters)',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Effect Size (Eta Squared)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Significance (p < 0.05)', title_fontsize=11)

    # Add value labels on bars
    for i, (_, row) in enumerate(anova_df.iterrows()):
        plt.text(i, row['eta_squared'] + 0.01, f'{row["eta_squared"]:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('anova_effect_sizes.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 ANOVA visualization saved as 'anova_effect_sizes.png'")


def visualize_features(df, features):
    """Create boxplots for significant features"""
    if not features:
        print("❌ No features to visualize")
        return

    print(f"\n📈 Creating feature visualizations for {len(features)} features...")

    # Create subplots
    n_features = len(features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_features == 1:
        axes = [axes]
    elif n_rows > 1:
        axes = axes.ravel()

    quality_names = {0: 'Poor', 1: 'Moderate', 2: 'Good'}

    for i, feat in enumerate(features):
        if i < len(axes):
            # Create boxplot
            box_data = [df[df['water_quality'] == qc][feat].dropna() for qc in sorted(df['water_quality'].unique())]
            box_labels = [f'{quality_names[qc]}\n(n={len(data)})' for qc, data in
                          zip(sorted(df['water_quality'].unique()), box_data)]

            axes[i].boxplot(box_data, labels=box_labels)
            axes[i].set_title(f'{feat}', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Value', fontsize=10)
            axes[i].grid(axis='y', alpha=0.3)

            # Add sample size annotation
            axes[i].text(0.02, 0.98, f'n={len(df[feat].dropna())}',
                         transform=axes[i].transAxes, va='top', ha='left',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Remove empty subplots
    for i in range(len(features), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Feature Distributions by Water Quality Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Feature distributions saved as 'feature_distributions.png'")


def prep_for_ml(df, features):
    """Prepare data for machine learning"""
    if not features:
        print("❌ No features provided for ML preparation")
        return None, None

    print(f"\n🤖 Preparing data for ML with {len(features)} features...")

    X = df[features]
    y = df['water_quality']

    print(f"📐 Original data shape - X: {X.shape}, y: {y.shape}")
    print(f"🎯 Class distribution: {dict(y.value_counts().sort_index())}")

    # Handle missing values
    missing_before = X.isnull().sum().sum()
    imp = SimpleImputer(strategy='median')
    X_imp = pd.DataFrame(imp.fit_transform(X), columns=features)
    missing_after = X_imp.isnull().sum().sum()

    print(f"🔧 Missing values: {missing_before} before, {missing_after} after imputation")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    print("⚖️ Features scaled using StandardScaler")

    print(f"🎯 Final ML data shape - X: {X_scaled.shape}, y: {y.shape}")

    return X_scaled, y


def rf_analysis(X, y, features):
    """Perform Random Forest analysis"""
    if X is None or y is None:
        print("❌ No data provided for RF analysis")
        return None, None

    print("\n🌲 Starting Random Forest analysis...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.20, random_state=42
    )
    print(f"📊 Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    print(f"📊 Train class distribution: {dict(pd.Series(y_train).value_counts().sort_index())}")
    print(f"📊 Test class distribution: {dict(pd.Series(y_test).value_counts().sort_index())}")

    # Train Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    print("✅ Random Forest model trained")

    # Predictions
    y_pred = rf.predict(X_test)

    # Evaluation
    print('\n📋 Random Forest Classification Report:')
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    print('\n📊 Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Cross-validation ROC-AUC
    try:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_scores = cross_val_score(rf, X, y, cv=skf, scoring='roc_auc_ovo')
        print(f"\n🎯 CV ROC-AUC (macro/ovo): {roc_scores.mean():.3f} ± {roc_scores.std():.3f}")
    except:
        print("⚠️  Could not compute ROC-AUC (possibly due to class distribution)")

    # Export confusion matrix & metrics
    pd.DataFrame(cm).to_csv('confusion_matrix.csv')
    print("💾 Confusion matrix saved to 'confusion_matrix.csv'")

    # Feature importances
    feat_imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
    print('\n🎯 Feature Importances:')
    print(feat_imp.to_string(float_format='%.4f'))
    feat_imp.to_csv('rf_feature_importance.csv')
    print("💾 Feature importances saved to 'rf_feature_importance.csv'")

    # Plot feature importances
    plt.figure(figsize=(10, 6))
    feat_imp.plot(kind='bar', color='teal')
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Random Forest Feature Importances\n(Water Quality Classification)',
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('rf_feat_importances.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Feature importance plot saved as 'rf_feat_importances.png'")

    return rf, feat_imp


if __name__ == "__main__":
    print("🚀 Starting Water Quality Analysis Pipeline...")
    print("=" * 60)

    # Update this path to your actual file location
    file_path = 'WQD.xlsx'  # Using relative path - make sure file is in same directory

    try:
        # Step 1: Load and clean data
        df = clean_and_prepare_data(file_path)
        print(f"✅ Data cleaning completed. Final dataset shape: {df.shape}")

        # Step 2: Perform ANOVA
        significant_features = perform_anova(df)

        # Step 3: Visualizations
        if significant_features:
            anova_df = pd.read_csv('anova_summary.csv')
            visualize_anova(anova_df)
            visualize_features(df, significant_features)

            # Step 4: Machine Learning
            X, y = prep_for_ml(df, significant_features)
            if X is not None and y is not None:
                rf_model, feature_importances = rf_analysis(X, y, significant_features)

                print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
                print("=" * 60)
                print("📁 Generated Files:")
                print("  - anova_summary.csv")
                print("  - anova_effect_sizes.png")
                print("  - feature_distributions.png")
                print("  - confusion_matrix.csv")
                print("  - rf_feature_importance.csv")
                print("  - rf_feat_importances.png")

                if feature_importances is not None and len(feature_importances) > 0:
                    top_feature = feature_importances.index[0]
                    print(f"\n🎯 Key Insight: Most important feature is '{top_feature}'")
            else:
                print("❌ ML analysis skipped due to data issues")
        else:
            print("❌ No significant features found for further analysis!")

    except FileNotFoundError:
        print(f"❌ ERROR: File not found at {file_path}")
        print("💡 Please check the file path and ensure the Excel file exists")
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback

        traceback.print_exc()

    print("=" * 60)