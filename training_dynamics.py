# ==========================================================
# Day 13 - Diagnose Training Dynamics
# AI Internship
# Dataset : Fashion-MNIST
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)

from tensorflow.keras.optimizers import (
    Adam,
    SGD
)

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau
)

print("=" * 60)
print("DAY 13 - DIAGNOSE TRAINING DYNAMICS")
print("=" * 60)

# ==========================================================
# Create Project Folders
# ==========================================================

os.makedirs("plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("Project folders created successfully.\n")

# ==========================================================
# Load Fashion-MNIST Dataset
# ==========================================================

print("Loading Fashion-MNIST Dataset...\n")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print("Dataset Loaded Successfully!\n")

print("Training Images :", x_train.shape)
print("Training Labels :", y_train.shape)
print("Testing Images  :", x_test.shape)
print("Testing Labels  :", y_test.shape)

# ==========================================================
# Normalize Images
# ==========================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

print("\nImages Normalized Successfully")

print("Training Shape :", x_train.shape)
print("Testing Shape  :", x_test.shape)

# ==========================================================
# Hyperparameters
# ==========================================================

BATCH_SIZE = 128

EPOCHS = 10

NUM_CLASSES = 10

# ==========================================================
# CNN Model
# ==========================================================

def build_model():

    model = Sequential([

        Input(shape=(28,28,1)),

        Conv2D(
            32,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D((2,2)),

        Conv2D(
            64,
            (3,3),
            activation="relu"
        ),

        MaxPooling2D((2,2)),

        Flatten(),

        Dense(
            128,
            activation="relu"
        ),

        Dropout(0.5),

        Dense(
            NUM_CLASSES,
            activation="softmax"
        )

    ])

    return model

# ==========================================================
# Show Model Summary
# ==========================================================

model = build_model()

print("\nCNN Model Summary\n")

model.summary()

print("\nModel Created Successfully.")
print("=" * 60)


# ==========================================================
# Learning Rate Scheduler
# ==========================================================

print("\nCreating CosineDecay Learning Rate Schedule...")



print("CosineDecay Scheduler Created Successfully.")

# ==========================================================
# Reduce Learning Rate Callback
# ==========================================================

reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=2,

    verbose=1,

    min_lr=1e-6

)

print("ReduceLROnPlateau Callback Ready.")

# ==========================================================
# Train using Adam Optimizer
# ==========================================================

print("\nStarting Training with Adam Optimizer...\n")

adam_model = build_model()

adam_model.compile(

    optimizer=Adam(
    learning_rate=0.001
),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

adam_history = adam_model.fit(

    x_train,

    y_train,

    validation_data=(x_test, y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[reduce_lr],

    verbose=1

)

print("\nAdam Training Completed Successfully.")

# ==========================================================
# Save Adam Model
# ==========================================================

adam_model.save("models/adam_model.keras")

print("Adam Model Saved.")

# ==========================================================
# Save Adam Training History
# ==========================================================

adam_history_df = pd.DataFrame(adam_history.history)

adam_history_df.to_csv(

    "logs/adam_history.csv",

    index=False

)

print("Adam History Saved.")

# ==========================================================
# Plot Adam Accuracy
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    adam_history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    adam_history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Adam Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig("plots/adam_accuracy.png")

plt.show()

# ==========================================================
# Plot Adam Loss
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(
    adam_history.history["loss"],
    label="Training Loss"
)

plt.plot(
    adam_history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Adam Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig("plots/adam_loss.png")

plt.show()

print("\nAdam Accuracy & Loss Graphs Saved.")

# ==========================================================
# Train using SGD Optimizer
# ==========================================================

print("\n" + "=" * 60)
print("Training with SGD Optimizer")
print("=" * 60)

sgd_model = build_model()

sgd_model.compile(

    optimizer=SGD(

        learning_rate=0.01,

        momentum=0.9

    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)

sgd_history = sgd_model.fit(

    x_train,

    y_train,

    validation_data=(x_test, y_test),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[reduce_lr],

    verbose=1

)

print("\nSGD Training Completed Successfully.")

# ==========================================================
# Save SGD Model
# ==========================================================

sgd_model.save("models/sgd_model.keras")

print("SGD Model Saved.")

# ==========================================================
# Save SGD History
# ==========================================================

sgd_history_df = pd.DataFrame(

    sgd_history.history

)

sgd_history_df.to_csv(

    "logs/sgd_history.csv",

    index=False

)

print("SGD History Saved.")

# ==========================================================
# Plot SGD Accuracy
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    sgd_history.history["accuracy"],

    label="Training Accuracy"

)

plt.plot(

    sgd_history.history["val_accuracy"],

    label="Validation Accuracy"

)

plt.title("SGD Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(

    "plots/sgd_accuracy.png"

)

plt.show()

# ==========================================================
# Plot SGD Loss
# ==========================================================

plt.figure(figsize=(8,5))

plt.plot(

    sgd_history.history["loss"],

    label="Training Loss"

)

plt.plot(

    sgd_history.history["val_loss"],

    label="Validation Loss"

)

plt.title("SGD Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)

plt.savefig(

    "plots/sgd_loss.png"

)

plt.show()

print("SGD Accuracy & Loss Graphs Saved.")

print("=" * 60)

# ==========================================================
# Compare Adam vs SGD
# ==========================================================

print("\n" + "=" * 60)
print("Comparing Adam and SGD")
print("=" * 60)

# ==========================================================
# Validation Accuracy Comparison
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    adam_history.history["val_accuracy"],
    marker="o",
    label="Adam"
)

plt.plot(
    sgd_history.history["val_accuracy"],
    marker="s",
    label="SGD"
)

plt.title("Validation Accuracy Comparison")

plt.xlabel("Epoch")

plt.ylabel("Validation Accuracy")

plt.legend()

plt.grid(True)

plt.savefig(
    "plots/optimizer_accuracy_comparison.png"
)

plt.show()

# ==========================================================
# Validation Loss Comparison
# ==========================================================

plt.figure(figsize=(10,6))

plt.plot(
    adam_history.history["val_loss"],
    marker="o",
    label="Adam"
)

plt.plot(
    sgd_history.history["val_loss"],
    marker="s",
    label="SGD"
)

plt.title("Validation Loss Comparison")

plt.xlabel("Epoch")

plt.ylabel("Validation Loss")

plt.legend()

plt.grid(True)

plt.savefig(
    "plots/optimizer_loss_comparison.png"
)

plt.show()

print("Comparison graphs saved.")

# ==========================================================
# Validation Loss Table
# ==========================================================

results = pd.DataFrame({

    "Epoch": np.arange(
        1,
        EPOCHS + 1
    ),

    "Adam Validation Loss":
        adam_history.history["val_loss"],

    "SGD Validation Loss":
        sgd_history.history["val_loss"],

    "Adam Validation Accuracy":
        adam_history.history["val_accuracy"],

    "SGD Validation Accuracy":
        sgd_history.history["val_accuracy"]

})

print("\nValidation Results")

print(results)

results.to_csv(

    "logs/validation_results.csv",

    index=False

)

print("\nValidation results saved.")

# ==========================================================
# Final Evaluation
# ==========================================================

print("\nEvaluating Models...\n")

adam_loss, adam_accuracy = adam_model.evaluate(

    x_test,

    y_test,

    verbose=0

)

sgd_loss, sgd_accuracy = sgd_model.evaluate(

    x_test,

    y_test,

    verbose=0

)

print("=" * 60)

print("FINAL RESULTS")

print("=" * 60)

print(f"Adam Test Accuracy : {adam_accuracy:.4f}")
print(f"Adam Test Loss     : {adam_loss:.4f}")

print()

print(f"SGD Test Accuracy  : {sgd_accuracy:.4f}")
print(f"SGD Test Loss      : {sgd_loss:.4f}")

print("=" * 60)

# ==========================================================
# Best Optimizer
# ==========================================================

if adam_accuracy > sgd_accuracy:

    best_optimizer = "Adam"

else:

    best_optimizer = "SGD"

print(f"\nBest Optimizer : {best_optimizer}")

# ==========================================================
# Project Summary
# ==========================================================

print("\nProject Completed Successfully!")

print("\nGenerated Files")

print("- models/adam_model.keras")
print("- models/sgd_model.keras")

print("- plots/adam_accuracy.png")
print("- plots/adam_loss.png")
print("- plots/sgd_accuracy.png")
print("- plots/sgd_loss.png")
print("- plots/optimizer_accuracy_comparison.png")
print("- plots/optimizer_loss_comparison.png")

print("- logs/adam_history.csv")
print("- logs/sgd_history.csv")
print("- logs/validation_results.csv")

print("=" * 60)
print("Day 13 Completed Successfully!")
print("=" * 60)