import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import joblib

from skimage.feature import hog
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.colors import LogNorm

from Test.app.labels import label_to_char


# =========================
# CONFIG
# =========================
SEED = 42
NUM_CLASSES = 62

MAX_TRAIN = 200_000       # number of samples for SGD-SVM training
CALIB_SAMPLES = 20_000    # samples for probability calibration
TEST_SAMPLES = None      # None = full test set

DO_TRANSPOSE = True      # EMNIST orientation fix
BATCH_STREAM = 2048


# =========================
# Preprocessing (TFDS)
# =========================
def preprocess_tf(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    if DO_TRANSPOSE:
        x = tf.transpose(x, perm=[1, 0, 2])
    y = tf.cast(y, tf.int32)
    return x, y


# =========================
# HOG
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
# Confusion Matrix (Ultra)
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
        values_format=None,
        colorbar=True,
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
# Iterators
# =========================
def iter_hog_batches(ds, max_samples=None, batch_size=2048):
    X_buf, y_buf = [], []
    count = 0

    for x, y in tfds.as_numpy(ds):
        img = np.squeeze(x)  # (28,28)
        X_buf.append(hog_features(img))
        y_buf.append(int(y))
        count += 1

        if len(X_buf) >= batch_size:
            yield np.stack(X_buf), np.array(y_buf, dtype=np.int64)
            X_buf, y_buf = [], []

        if max_samples is not None and count >= max_samples:
            break

    if X_buf:
        yield np.stack(X_buf), np.array(y_buf, dtype=np.int64)


def collect_hog(ds, n_samples):
    X_list, y_list = [], []
    for xb, yb in iter_hog_batches(ds, max_samples=n_samples, batch_size=2048):
        X_list.append(xb)
        y_list.append(yb)
    return np.vstack(X_list), np.concatenate(y_list)


# =========================
# MAIN
# =========================
def main():
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    model_dir = os.path.join(project_root, "Test", "SVM", "models")
    report_dir = os.path.join(project_root, "Test", "SVM", "report")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "emnist_hog_linear_svm.joblib")

    # Load datasets
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test = tfds.load("emnist/byclass", split="test", as_supervised=True)

    ds_train = ds_train.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.shuffle(50_000, seed=SEED, reshuffle_each_iteration=False)
    ds_test = ds_test.map(preprocess_tf, num_parallel_calls=tf.data.AUTOTUNE)

    # =========================
    # Train Linear SVM (SGD)
    # =========================
    svm = SGDClassifier(
        loss="hinge",
        alpha=1e-4,
        learning_rate="optimal",
        random_state=SEED,
        n_jobs=-1,
    )

    classes = np.arange(NUM_CLASSES, dtype=np.int64)

    seen = 0
    first = True
    for Xb, yb in iter_hog_batches(ds_train, max_samples=MAX_TRAIN, batch_size=BATCH_STREAM):
        if first:
            svm.partial_fit(Xb, yb, classes=classes)
            first = False
        else:
            svm.partial_fit(Xb, yb)
        seen += len(yb)
        if seen % 50_000 == 0:
            print(f"Trained on {seen:,} samples...")

    print(f"Finished SGD-SVM training on {seen:,} samples.")

    # =========================
    # Calibration (FIXED)
    # =========================
    print(f"Collecting {CALIB_SAMPLES:,} samples for calibration...")
    X_cal, y_cal = collect_hog(ds_train, CALIB_SAMPLES)

    calibrated = CalibratedClassifierCV(
        estimator=svm,
        method="sigmoid",
        cv=3,        # ✔️ sklearn-compatible
    )
    calibrated.fit(X_cal, y_cal)

    joblib.dump(calibrated, model_path, compress=3)
    print("Saved model:", model_path)

    # =========================
    # Evaluation
    # =========================
    print("Evaluating on test set...")
    y_true, y_pred = [], []

    for Xb, yb in iter_hog_batches(ds_test, max_samples=TEST_SAMPLES, batch_size=2048):
        preds = calibrated.predict(Xb)
        y_true.extend(yb.tolist())
        y_pred.extend(preds.tolist())

    y_true = np.array(y_true, dtype=np.int64)
    y_pred = np.array(y_pred, dtype=np.int64)

    acc = float((y_true == y_pred).mean())
    print(f"Test accuracy: {acc:.4f}")

    # =========================
    # Accuracy plot
    # =========================
    acc_path = os.path.join(report_dir, "accuracy.png")
    plt.figure(figsize=(6, 4))
    plt.bar(["HOG + Linear SVM"], [acc])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Accuracy (EMNIST/byclass test)")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=200)
    plt.show()

    # =========================
    # Confusion Matrix
    # =========================
    cm_path = os.path.join(report_dir, "confusion_matrix_ultra.png")
    plot_cm_ultra_counts(
        y_true,
        y_pred,
        cm_path,
        title="Confusion Matrix (Test) – HOG + Linear SVM (Counts, Log Scale)",
    )

    print("Reports saved to:", report_dir)
    print(" -", acc_path)
    print(" -", cm_path)


if __name__ == "__main__":
    main()
