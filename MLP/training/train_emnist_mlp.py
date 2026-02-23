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

BATCH = 256
MAX_EPOCHS = 50

# EMNIST orientation fix
DO_TRANSPOSE = True

# Early stopping settings (MLP usually overfits sooner than CNN)
EARLYSTOP_PATIENCE = 4
LR_PLATEAU_PATIENCE = 2
MIN_LR = 1e-5


# =========================
# Preprocessing
# =========================
def preprocess_emnist(x, y):
    """
    EMNIST -> MLP input:
    - normalize 0..1
    - transpose to fix EMNIST orientation (if enabled)
    - flatten to (784,)
    """
    x = tf.cast(x, tf.float32) / 255.0
    if DO_TRANSPOSE:
        x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reshape(x, [-1])  # (784,)
    y = tf.cast(y, tf.int32)
    return x, y


# =========================
# Model
# =========================
def build_mlp():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(784,)),

        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])


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

    model_dir = os.path.join(project_root, "Test", "MLP", "models")
    report_dir = os.path.join(project_root, "Test", "MLP", "report")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    best_model_path = os.path.join(model_dir, "emnist_mlp_best.keras")

    # Load EMNIST
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test  = tfds.load("emnist/byclass", split="test",  as_supervised=True)

    ds_train = ds_train.map(preprocess_emnist, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess_emnist,  num_parallel_calls=tf.data.AUTOTUNE)

    ds_train = ds_train.cache().shuffle(50_000, seed=SEED).batch(BATCH).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.cache().batch(512).prefetch(tf.data.AUTOTUNE)

    # Build model
    model = build_mlp()
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

    # -------- TRAIN (stops automatically) --------
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
    plt.title("MLP Accuracy (EarlyStopping)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.show()

    # =========================
    # Confusion Matrix (best weights)
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
        title="Confusion Matrix (Test) – MLP (Best Weights, Counts, Log Scale)",
    )

    print("Reports saved to:", report_dir)
    print(" -", acc_path)
    print(" -", cm_path)


if __name__ == "__main__":
    main()
