import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LogNorm
from Test.app.labels import label_to_char


# =========================
# CONFIG
# =========================
SEED = 42
NUM_CLASSES = 62
BATCH = 128

# Train "as long as useful", but stop early:
MAX_EPOCHS = 50

# EMNIST orientation fix (for dataset)
DO_TRANSPOSE = True

# Early stopping settings
EARLYSTOP_PATIENCE = 6          # stop if no val_acc improvement for 6 epochs
LR_PLATEAU_PATIENCE = 3         # reduce LR after 3 epochs without improvement
MIN_LR = 1e-5


# =========================
# Preprocessing
# =========================
def preprocess_emnist(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    if DO_TRANSPOSE:
        x = tf.transpose(x, perm=[1, 0, 2])
    y = tf.cast(y, tf.int32)
    return x, y


# =========================
# Data augmentation
# =========================
def make_augmentation():
    # Mild augmentation: good for EMNIST without changing labels too often
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08, fill_mode="constant"),
        tf.keras.layers.RandomTranslation(0.08, 0.08, fill_mode="constant"),
        tf.keras.layers.RandomZoom(0.10, 0.10, fill_mode="constant"),
    ], name="augment")


# =========================
# CNN model
# =========================
def build_cnn():
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = inputs

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(32, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.15)(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


# =========================
# ULTRA CONFUSION MATRIX (COUNTS ONLY)
# =========================
def plot_cm_ultra_counts(y_true, y_pred, out_path: str, title: str, max_labels: int = 62):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(max_labels)))
    labels = [label_to_char(i) for i in range(max_labels)]

    fig, ax = plt.subplots(figsize=(32, 32))
    norm = LogNorm(vmin=1, vmax=cm.max())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        colorbar=True,
        values_format=None,
        im_kw={"norm": norm},
    )

    ax.set_title(title, fontsize=26, pad=30)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xticks(np.arange(max_labels) - 0.5, minor=True)
    ax.set_yticks(np.arange(max_labels) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)


# =========================
# MAIN
# =========================
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    model_dir = os.path.join(project_root, "Test", "CNN", "models")
    report_dir = os.path.join(project_root, "Test", "CNN", "report")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # Save the BEST model automatically
    best_model_path = os.path.join(model_dir, "emnist_cnn_best.keras")

    # Load EMNIST
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test  = tfds.load("emnist/byclass", split="test",  as_supervised=True)

    ds_train = ds_train.map(preprocess_emnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess_emnist,  num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.cache().shuffle(50_000, seed=SEED).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.cache().batch(256).prefetch(tf.data.AUTOTUNE)

    # Build model with augmentation
    aug = make_augmentation()
    base = build_cnn()

    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = aug(inputs)
    outputs = base(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            best_model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            mode="max",
            factor=0.5,
            patience=LR_PLATEAU_PATIENCE,
            min_lr=MIN_LR,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=EARLYSTOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # -------- TRAIN (stops automatically when it no longer improves) --------
    history = model.fit(
        ds_train,
        epochs=MAX_EPOCHS,
        validation_data=ds_test,
        callbacks=callbacks,
    )

    print("Best model saved to:", best_model_path)

    # =========================
    # Accuracy plot
    # =========================
    acc_path = os.path.join(report_dir, "accuracy.png")
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("CNN Accuracy (EarlyStopping)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.show()

    # =========================
    # Confusion Matrix using BEST model weights (already restored)
    # =========================
    y_true = []
    y_pred = []

    for x_batch, y_batch in ds_test:
        probs = model.predict(x_batch, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(y_batch.numpy().tolist())
        y_pred.extend(preds.tolist())

    cm_path = os.path.join(report_dir, "confusion_matrix_ultra.png")
    plot_cm_ultra_counts(
        y_true,
        y_pred,
        cm_path,
        title="Confusion Matrix (Test) – CNN (Best Weights, Counts, Log Scale)",
    )

    print("Reports saved to:", report_dir)
    print(" -", acc_path)
    print(" -", cm_path)


if __name__ == "__main__":
    main()
