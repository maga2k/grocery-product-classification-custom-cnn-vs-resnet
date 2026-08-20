# ============================================================
# Experiment selector
# ============================================================
# Toggle which experiments to run: [CNN_v1, CNN_v2, CNN_v3]
# Example: [False, False, True] runs only CNN_v3
RUN_EXPERIMENTS = {
    "CNN_v1": False,
    "CNN_v2": False,
    "CNN_v3": True,
}

# ============================================================
# Imports & Setup
# ============================================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from pathlib import Path
import utils
from utils import plot_history, save_model, save_history, save_results_summary

# ============================================================
# Load dataset and classes.csv
# ============================================================
DATASET_PATH = Path("GroceryStoreDataset/dataset")

# Load classes.csv first, needed for the label -> name mapping
classes_df = pd.read_csv(DATASET_PATH / "classes.csv")
label_to_name = dict(zip(
    classes_df['Class ID (int)'].astype(str),
    classes_df['Class Name (str)']
))

print("Classes loaded. Example:")
print(classes_df.head(3))

def load_txt(split):
    rows = []
    with open(DATASET_PATH / f"{split}.txt", "r") as f:
        for line in f:
            parts = line.strip().split(", ")
            rows.append({
                "full_path": str(DATASET_PATH / parts[0]),
                "fine_label": str(int(parts[1])),
                "coarse_label": int(parts[2])
            })
    return pd.DataFrame(rows)

train_df = load_txt("train")
val_df   = load_txt("val")
test_df  = load_txt("test")

print(f"\nTRAIN: {len(train_df)} images")
print(f"VAL:   {len(val_df)} images")
print(f"TEST:  {len(test_df)} images")
print(f"Fine-grained classes: {train_df['fine_label'].nunique()}")

# ============================================================
# Global hyperparameter configuration
# ============================================================
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
NUM_CLASSES = train_df['fine_label'].nunique()  # 81

print(f"\nNumber of classes: {NUM_CLASSES}")
print(f"Image size: {IMG_SIZE}")
print(f"Batch size: {BATCH_SIZE}")

# ============================================================
# Data augmentation + data loading
# ============================================================
# Fixed, ordered list of all 81 classes so that the label -> index
# mapping is identical across train/val/test generators.
all_classes = sorted(classes_df['Class ID (int)'].astype(str).tolist())

def build_generator(df, augment=False):
    if augment:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
        )
    else:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        x_col="full_path",
        y_col="fine_label",
        classes=all_classes,        # forces a fixed, identical class mapping everywhere
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="sparse",
        shuffle=augment,
        seed=42
    )
    return generator

train_generator = build_generator(train_df, augment=True)
val_generator   = build_generator(val_df,   augment=False)

print(f"\nTrain batches: {len(train_generator)}")
print(f"Val batches:   {len(val_generator)}")

# Sanity check: class mapping must match between train and val
assert train_generator.class_indices == val_generator.class_indices, "Class mapping mismatch!"
print("Class mapping consistent between train and val \u2714")

# ============================================================
# Visualize a few training examples
# ============================================================
images, labels = next(iter(train_generator))

plt.figure(figsize=(14, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(images[i])
    class_name = label_to_name.get(str(int(labels[i])), str(int(labels[i])))
    plt.title(class_name, fontsize=7)
    plt.axis('off')
plt.suptitle("Training set examples (with augmentation)", fontsize=12)
plt.tight_layout()
plt.show()


# ============================================================
# Task 1: Baseline CNN (Experiment 1)
# ============================================================
def build_baseline_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Classifier
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_cnn_v1():
    print("\n" + "=" * 60)
    print("Running CNN_v1 (baseline)")
    print("=" * 60)

    model = build_baseline_cnn()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=20,
        validation_data=val_generator,
        verbose=1
    )

    plot_history(history, title="CNN_v1")
    save_history(history, model_name="CNN_v1")
    save_model(model, model_name="CNN_v1")
    save_results_summary({
        "model": "CNN_v1",
        "val_accuracy": max(history.history['val_accuracy']),
        "train_accuracy": max(history.history['accuracy']),
        "params": model.count_params(),
        "epochs_run": len(history.history['loss']),
        "notes": "Baseline: 3 conv blocks + Flatten + Dense256, no BatchNorm, no callbacks"
    })

    return model, history


# ============================================================
# Task 1: CNN v2 (anti-overfitting: GAP + BatchNorm + callbacks)
# ============================================================
def build_cnn_v2(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # GlobalAveragePooling2D instead of Flatten: removes ~25M params
        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_cnn_v2():
    print("\n" + "=" * 60)
    print("Running CNN_v2 (GAP + BatchNorm + callbacks)")
    print("=" * 60)

    model_v2 = build_cnn_v2()
    model_v2.summary()

    model_v2.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # NOTE: both EarlyStopping and ModelCheckpoint monitor 'val_accuracy'
    # so the in-memory restored model and the checkpoint saved to disk
    # correspond to the SAME epoch (fixed inconsistency from earlier version,
    # where EarlyStopping watched val_loss and ModelCheckpoint watched val_accuracy).
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=7, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(utils.MODELS_DIR / "CNN_v2_best.keras"),
            monitor='val_accuracy', mode='max', save_best_only=True
        ),
    ]

    history_v2 = model_v2.fit(
        train_generator,
        epochs=40,  # EarlyStopping may stop earlier
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_v2, title="CNN_v2 (GAP + BatchNorm + Callbacks)")
    save_history(history_v2, model_name="CNN_v2")
    save_results_summary({
        "model": "CNN_v2",
        "val_accuracy": max(history_v2.history['val_accuracy']),
        "train_accuracy": max(history_v2.history['accuracy']),
        "params": model_v2.count_params(),
        "epochs_run": len(history_v2.history['loss']),
        "notes": "GlobalAveragePooling2D instead of Flatten, BatchNorm after every conv, "
                 "EarlyStopping + ReduceLROnPlateau + ModelCheckpoint all monitoring val_accuracy"
    })

    return model_v2, history_v2


# ============================================================
# Task 1: CNN v3 (more capacity, still lightweight thanks to GAP)
# ============================================================
def build_cnn_v3(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        # 4th block
        layers.Conv2D(256, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def run_cnn_v3():
    print("\n" + "=" * 60)
    print("Running CNN_v3 (4 conv blocks, more capacity)")
    print("=" * 60)

    model_v3 = build_cnn_v3()
    model_v3.summary()

    model_v3.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', mode='max', patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(utils.MODELS_DIR / "CNN_v3_best.keras"),
            monitor='val_accuracy', mode='max', save_best_only=True
        ),
    ]

    history_v3 = model_v3.fit(
        train_generator,
        epochs=50,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    plot_history(history_v3, title="CNN_v3 (4 blocks, more capacity)")
    save_history(history_v3, model_name="CNN_v3")
    save_results_summary({
        "model": "CNN_v3",
        "val_accuracy": max(history_v3.history['val_accuracy']),
        "train_accuracy": max(history_v3.history['accuracy']),
        "params": model_v3.count_params(),
        "epochs_run": len(history_v3.history['loss']),
        "notes": "4 conv blocks (32-64-128-256) + GAP + BatchNorm, lr=5e-4, "
                 "consistent val_accuracy monitoring across all callbacks"
    })

    return model_v3, history_v3


# ============================================================
# Run selected experiments
# ============================================================
if __name__ == "__main__":
    if RUN_EXPERIMENTS.get("CNN_v1"):
        run_cnn_v1()

    if RUN_EXPERIMENTS.get("CNN_v2"):
        run_cnn_v2()

    if RUN_EXPERIMENTS.get("CNN_v3"):
        run_cnn_v3()