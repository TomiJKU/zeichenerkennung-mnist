import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LogNorm
from skimage.feature import hog

from Test.app.labels import label_to_char


# =========================
# CONFIG
# =========================
SEED = 42
NUM_CLASSES = 62

# Must match EMNIST orientation fix used in training scripts for TFDS samples
DO_TRANSPOSE_TRAIN = True

# Paths (adjust if needed)
CNN_MODEL_PATH = os.path.join("Test", "CNN", "models", "emnist_cnn_best.keras")
if not os.path.exists(CNN_MODEL_PATH):
    CNN_MODEL_PATH = os.path.join("Test", "CNN", "models", "emnist_cnn_aug.keras")

MLP_MODEL_PATH = os.path.join("Test", "MLP", "models", "emnist_mlp_best.keras")
if not os.path.exists(MLP_MODEL_PATH):
    MLP_MODEL_PATH = os.path.join("Test", "MLP", "models", "emnist_mlp.keras")

SVM_MODEL_PATH = os.path.join("Test", "SVM", "models", "emnist_hog_linear_svm.joblib")

OUT_DIR = os.path.join("Test", "report_compare")

# Speed benchmark (how many test samples to time)
SPEED_SAMPLES = 10_000

# Batch sizes for Keras speed tests
CNN_BATCH = 256
MLP_BATCH = 512


# =========================
# TFDS Preprocessing
# =========================
def preprocess_for_cnn(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    if DO_TRANSPOSE_TRAIN:
        x = tf.transpose(x, perm=[1, 0, 2])
    y = tf.cast(y, tf.int32)
    return x, y


def preprocess_for_mlp(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    if DO_TRANSPOSE_TRAIN:
        x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reshape(x, [-1])  # (784,)
    y = tf.cast(y, tf.int32)
    return x, y


# =========================
# HOG (must match SVM training)
# =========================
def hog_features(img28: np.ndarray) -> np.ndarray:
    return hog(
        img28,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


# =========================
# Plot helpers (PNG only)
# =========================
def plot_accuracy_compare_png(acc_dict, out_path):
    names = list(acc_dict.keys())
    vals = [acc_dict[k] for k in names]

    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison (EMNIST/byclass test)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_model_size_compare_png(size_mb_dict, out_path):
    names = list(size_mb_dict.keys())
    vals = [size_mb_dict[k] for k in names]

    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylabel("Model size (MB)")
    plt.title("Model size comparison")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_speed_compare_png(ms_per_1000_dict, out_path):
    names = list(ms_per_1000_dict.keys())
    vals = [ms_per_1000_dict[k] for k in names]

    plt.figure(figsize=(7, 4))
    plt.bar(names, vals)
    plt.ylabel("ms per 1000 samples")
    plt.title(f"Inference speed comparison ({SPEED_SAMPLES} samples timed)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_cm_labeled_ultra_png(y_true, y_pred, out_path, title):
    """
    Ultra readable CM (counts, log scale) with labels 0-9 A-Z a-z.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    labels = [label_to_char(i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(32, 32))
    norm = LogNorm(vmin=1, vmax=cm.max())

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=90,
        values_format=None,          # counts only
        colorbar=True,
        im_kw={"norm": norm},
    )

    ax.set_title(title, fontsize=26, pad=30)
    ax.set_xlabel("Predicted label", fontsize=18)
    ax.set_ylabel("True label", fontsize=18)

    ax.tick_params(axis="x", labelsize=11)
    ax.tick_params(axis="y", labelsize=11)

    ax.set_xticks(np.arange(NUM_CLASSES) - 0.5, minor=True)
    ax.set_yticks(np.arange(NUM_CLASSES) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle=":", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=400)
    plt.close(fig)


def plot_per_class_accuracy_png(y_true, y_pred, out_path, title):
    """
    Creates a readable per-class accuracy heatmap (1 x 62).
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    row_sums = cm.sum(axis=1).astype(np.float32)
    diag = np.diag(cm).astype(np.float32)
    acc = np.divide(diag, row_sums, out=np.zeros_like(diag), where=row_sums > 0)  # (62,)

    labels = [label_to_char(i) for i in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(20, 2.8))
    img = ax.imshow(acc[None, :], aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_yticks([])

    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels(labels, rotation=90, fontsize=9)

    cbar = fig.colorbar(img, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label("Per-class accuracy", rotation=90)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


# =========================
# Evaluation helpers
# =========================
def eval_keras_model(model, ds):
    y_true = []
    y_pred = []

    for xb, yb in ds:
        probs = model.predict(xb, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(yb.numpy().tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


def eval_svm_model(svm_model, ds_test_raw):
    """
    Evaluate SVM on TFDS test set by extracting HOG features for each sample.
    For TFDS EMNIST:
      - no invert (already white-on-black)
      - transpose if DO_TRANSPOSE_TRAIN True
    """
    y_true = []
    y_pred = []

    for x, y in tfds.as_numpy(ds_test_raw):
        img = np.squeeze(x).astype(np.float32) / 255.0  # (28,28)

        if DO_TRANSPOSE_TRAIN:
            img = img.T

        feat = hog_features(img)
        pred = int(svm_model.predict(feat.reshape(1, -1))[0])

        y_true.append(int(y))
        y_pred.append(pred)

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)
    acc = float((y_true == y_pred).mean())
    return acc, y_true, y_pred


def speed_keras_model_ms_per_1000(model, ds_batched, n_samples, batch_size):
    """
    Measures ms per 1000 samples for Keras model.predict.
    """
    # collect exactly n_samples into one numpy array (fast predictable benchmark)
    xs = []
    collected = 0
    for xb, _ in ds_batched:
        xb_np = xb.numpy()
        xs.append(xb_np)
        collected += xb_np.shape[0]
        if collected >= n_samples:
            break

    x = np.concatenate(xs, axis=0)[:n_samples]

    # warmup
    _ = model.predict(x[:batch_size], verbose=0)

    t0 = time.time()
    _ = model.predict(x, verbose=0)
    dt = time.time() - t0

    ms_per_1000 = (dt * 1000.0) / (n_samples / 1000.0)
    return float(ms_per_1000)


def speed_svm_ms_per_1000(svm_model, ds_test_raw, n_samples):
    """
    Measures ms per 1000 samples including HOG extraction + predict.
    """
    feats = []
    count = 0
    for x, _y in tfds.as_numpy(ds_test_raw):
        img = np.squeeze(x).astype(np.float32) / 255.0
        if DO_TRANSPOSE_TRAIN:
            img = img.T
        feats.append(hog_features(img))
        count += 1
        if count >= n_samples:
            break

    X = np.stack(feats)

    # warmup
    _ = svm_model.predict(X[:32])

    t0 = time.time()
    _ = svm_model.predict(X)
    dt = time.time() - t0

    ms_per_1000 = (dt * 1000.0) / (n_samples / 1000.0)
    return float(ms_per_1000)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


# =========================
# MAIN
# =========================
def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load models
    if not os.path.exists(CNN_MODEL_PATH):
        raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
    if not os.path.exists(MLP_MODEL_PATH):
        raise FileNotFoundError(f"MLP model not found: {MLP_MODEL_PATH}")
    if not os.path.exists(SVM_MODEL_PATH):
        raise FileNotFoundError(f"SVM model not found: {SVM_MODEL_PATH}")

    cnn = tf.keras.models.load_model(CNN_MODEL_PATH)
    mlp = tf.keras.models.load_model(MLP_MODEL_PATH)
    svm_model = joblib.load(SVM_MODEL_PATH)

    # Load EMNIST test set (same for all)
    ds_test = tfds.load("emnist/byclass", split="test", as_supervised=True)

    ds_cnn = ds_test.map(preprocess_for_cnn, num_parallel_calls=tf.data.AUTOTUNE).batch(CNN_BATCH).prefetch(tf.data.AUTOTUNE)
    ds_mlp = ds_test.map(preprocess_for_mlp, num_parallel_calls=tf.data.AUTOTUNE).batch(MLP_BATCH).prefetch(tf.data.AUTOTUNE)

    # Evaluate accuracy + preds
    cnn_acc, cnn_y_true, cnn_y_pred = eval_keras_model(cnn, ds_cnn)
    mlp_acc, mlp_y_true, mlp_y_pred = eval_keras_model(mlp, ds_mlp)
    svm_acc, svm_y_true, svm_y_pred = eval_svm_model(svm_model, ds_test_raw=ds_test)

    # Accuracy compare PNG
    acc_png = os.path.join(OUT_DIR, "accuracy_compare.png")
    plot_accuracy_compare_png({"CNN": cnn_acc, "MLP": mlp_acc, "HOG+SVM": svm_acc}, acc_png)

    # Confusion matrices PNGs
    cm_cnn_png = os.path.join(OUT_DIR, "cm_cnn_labeled.png")
    cm_mlp_png = os.path.join(OUT_DIR, "cm_mlp_labeled.png")
    cm_svm_png = os.path.join(OUT_DIR, "cm_svm_labeled.png")

    plot_cm_labeled_ultra_png(cnn_y_true, cnn_y_pred, cm_cnn_png, "CNN Confusion Matrix (Counts, Log Scale)")
    plot_cm_labeled_ultra_png(mlp_y_true, mlp_y_pred, cm_mlp_png, "MLP Confusion Matrix (Counts, Log Scale)")
    plot_cm_labeled_ultra_png(svm_y_true, svm_y_pred, cm_svm_png, "HOG + Linear SVM Confusion Matrix (Counts, Log Scale)")

    # Per-class accuracy heatmaps PNGs
    pca_cnn_png = os.path.join(OUT_DIR, "per_class_accuracy_cnn.png")
    pca_mlp_png = os.path.join(OUT_DIR, "per_class_accuracy_mlp.png")
    pca_svm_png = os.path.join(OUT_DIR, "per_class_accuracy_svm.png")

    plot_per_class_accuracy_png(cnn_y_true, cnn_y_pred, pca_cnn_png, "Per-class Accuracy – CNN")
    plot_per_class_accuracy_png(mlp_y_true, mlp_y_pred, pca_mlp_png, "Per-class Accuracy – MLP")
    plot_per_class_accuracy_png(svm_y_true, svm_y_pred, pca_svm_png, "Per-class Accuracy – HOG+SVM")

    # Model size compare PNG
    size_png = os.path.join(OUT_DIR, "model_size_compare.png")
    sizes = {
        "CNN": file_size_mb(CNN_MODEL_PATH),
        "MLP": file_size_mb(MLP_MODEL_PATH),
        "HOG+SVM": file_size_mb(SVM_MODEL_PATH),
    }
    plot_model_size_compare_png(sizes, size_png)

    # Inference speed compare PNG (ms per 1000 samples)
    speed_cnn = speed_keras_model_ms_per_1000(cnn, ds_cnn, SPEED_SAMPLES, CNN_BATCH)
    speed_mlp = speed_keras_model_ms_per_1000(mlp, ds_mlp, SPEED_SAMPLES, MLP_BATCH)
    speed_svm = speed_svm_ms_per_1000(svm_model, ds_test, SPEED_SAMPLES)

    speed_png = os.path.join(OUT_DIR, "inference_speed_compare.png")
    plot_speed_compare_png({"CNN": speed_cnn, "MLP": speed_mlp, "HOG+SVM": speed_svm}, speed_png)

    print("Saved PNG-only comparison report to:", OUT_DIR)
    print(" -", acc_png)
    print(" -", cm_cnn_png)
    print(" -", cm_mlp_png)
    print(" -", cm_svm_png)
    print(" -", pca_cnn_png)
    print(" -", pca_mlp_png)
    print(" -", pca_svm_png)
    print(" -", size_png)
    print(" -", speed_png)


if __name__ == "__main__":
    main()
