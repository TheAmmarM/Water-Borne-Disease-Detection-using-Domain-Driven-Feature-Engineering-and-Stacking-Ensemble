import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks
import time

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Load data
train_df = pd.read_csv('train_scaled.csv')
test_df = pd.read_csv('test_scaled.csv')

X_train = train_df.drop('Water Quality', axis=1).values
y_train = train_df['Water Quality'].values
X_test = test_df.drop('Water Quality', axis=1).values
y_test = test_df['Water Quality'].values

# Get feature names for later
feature_names = train_df.drop('Water Quality', axis=1).columns.tolist()

print("=" * 70)
print("DEEP NEURAL NETWORK FOR WATER QUALITY CLASSIFICATION")
print("=" * 70)
print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Input features: {X_train.shape[1]}")
print(f"Classes: {len(np.unique(y_train))}\n")

# ============================================
# MODEL 1: BASIC DEEP NEURAL NETWORK
# ============================================
print("--- Model 1: Basic Deep Neural Network ---")

model_basic = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(3, activation='softmax')  # 3 classes
], name='Basic_DNN')

model_basic.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_basic.summary())

# ============================================
# MODEL 2: DEEP WIDE NETWORK (More Neurons)
# ============================================
print("\n--- Model 2: Deep Wide Network ---")

model_wide = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),

    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(3, activation='softmax')
], name='Wide_DNN')

model_wide.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_wide.summary())

# ============================================
# MODEL 3: RESIDUAL NETWORK (Skip Connections)
# ============================================
print("\n--- Model 3: Residual Network ---")


def create_residual_model():
    inputs = keras.Input(shape=(X_train.shape[1],))

    # First block
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    # Residual block 1
    residual = x
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Add()([x, residual])

    # Residual block 2
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    # Output
    outputs = layers.Dense(3, activation='softmax')(x)

    return keras.Model(inputs=inputs, outputs=outputs, name='Residual_DNN')


model_residual = create_residual_model()
model_residual.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(model_residual.summary())

# ============================================
# CALLBACKS FOR TRAINING
# ============================================
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-7,
    verbose=1
)

# ============================================
# TRAIN ALL MODELS
# ============================================
models = {
    'Basic DNN': model_basic,
    'Wide DNN': model_wide,
    'Residual DNN': model_residual
}

histories = {}
results = []

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )
    train_time = time.time() - start

    histories[name] = history

    # Evaluate
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Test Accuracy': accuracy,
        'Training Time (s)': train_time,
        'Epochs Trained': len(history.history['loss'])
    })

    print(f"   ✓ Accuracy: {accuracy:.4f}")
    print(f"   ✓ Epochs: {len(history.history['loss'])}")
    print(f"   ✓ Time: {train_time:.2f}s")

results_df = pd.DataFrame(results).sort_values('Test Accuracy', ascending=False)

print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)
print("\n", results_df.to_string(index=False))

# ============================================
# TRAINING HISTORY VISUALIZATION
# ============================================
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Accuracy curves
for name, history in histories.items():
    axes[0].plot(history.history['accuracy'], label=f'{name} (Train)', alpha=0.7)
    axes[0].plot(history.history['val_accuracy'], label=f'{name} (Val)', linestyle='--', alpha=0.7)

axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Loss curves
for name, history in histories.items():
    axes[1].plot(history.history['loss'], label=f'{name} (Train)', alpha=0.7)
    axes[1].plot(history.history['val_loss'], label=f'{name} (Val)', linestyle='--', alpha=0.7)

axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('dnn_training_history.png', dpi=300, bbox_inches='tight')
print("\n✅ Training history saved: 'dnn_training_history.png'")

# ============================================
# BEST MODEL ANALYSIS
# ============================================
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]

print(f"\n{'=' * 70}")
print(f"🏆 BEST MODEL: {best_model_name}")
print(f"{'=' * 70}")

# Predictions
y_pred_proba = best_model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_proba, axis=1)

# Classification report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=['Class 0 (Poor)', 'Class 1 (Moderate)', 'Class 2 (Good)']
))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Purples',
    xticklabels=['Poor', 'Moderate', 'Good'],
    yticklabels=['Poor', 'Moderate', 'Good']
)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('dnn_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✅ Confusion matrix saved")

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i in range(3):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"   Class {i}: {class_acc:.4f} ({cm[i, i]}/{cm[i].sum()})")

# ============================================
# MODEL COMPARISON BAR CHART
# ============================================
plt.figure(figsize=(12, 6))
plt.bar(results_df['Model'], results_df['Test Accuracy'], color=['steelblue', 'seagreen', 'coral'])
plt.xlabel('Model', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Deep Learning Model Comparison', fontsize=14, fontweight='bold')
plt.ylim([0.8, 1.0])
for i, (model, acc) in enumerate(zip(results_df['Model'], results_df['Test Accuracy'])):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig('dnn_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Model comparison saved")

# ============================================
# SAVE MODELS
# ============================================
for name, model in models.items():
    filename = name.lower().replace(' ', '_')
    model.save(f'{filename}.h5')
    print(f"✅ {name} saved: {filename}.h5")

# Save results
results_df.to_csv('dnn_results.csv', index=False)
print("✅ Results saved: dnn_results.csv")

print("\n" + "=" * 70)
print("DEEP LEARNING COMPLETE!")
print("=" * 70)
print(f"\n🏆 Best Model: {best_model_name}")
print(f"🎯 Test Accuracy: {results_df.iloc[0]['Test Accuracy']:.4f}")
print(f"⏱️  Training Time: {results_df.iloc[0]['Training Time (s)']:.2f}s")
print(f"📊 Epochs Trained: {results_df.iloc[0]['Epochs Trained']}")