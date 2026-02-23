import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# --------- Einstellungen optimiert für deinen PC ---------
SEED = 42

# Für RandomForest + PCA ist "alles" bei EMNIST/byclass meistens Overkill.
# Diese Werte laufen stabil auf 32 GB RAM und sind flott auf deiner CPU.
MAX_TRAIN = 120_000
MAX_TEST  = 30_000

PCA_COMPONENTS = 150

RF_TREES = 10
RF_MAX_DEPTH = 10          # None = lässt den Baum wachsen (oft bestes Resultat)
RF_MIN_SAMPLES_LEAF = 5      # 1 = maximale Kapazität; 2 kann generalisieren, aber oft etwas schlechter
# --------------------------------------------------------


def preprocess(x, y):
    # wie bei dir:
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.transpose(x, perm=[1, 0, 2])  # (28,28,1) – EMNIST ist oft gedreht
    y = tf.cast(y, tf.int32)
    return x, y


def ds_to_numpy_flat(ds, max_samples):
    """
    Konvertiert tf.data Dataset (x,y) in numpy:
      X: (N, 784) float32
      y: (N,) int64
    """
    xs = np.empty((max_samples, 784), dtype=np.float32)
    ys = np.empty((max_samples,), dtype=np.int64)

    t0 = time.time()
    for i, (x, y) in enumerate(tfds.as_numpy(ds.take(max_samples))):
        # x: (28,28,1) -> (784,)
        xs[i] = np.squeeze(x).reshape(-1)
        ys[i] = int(y)

        if (i + 1) % 20_000 == 0:
            print(f"  loaded {i+1:,}/{max_samples:,} samples ...")

    print(f"  done in {time.time() - t0:.1f}s")
    return xs, ys


def main():
    # Reproduzierbarkeit
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(project_root, "models")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(
        out_dir,
        "PCA_randomForest_emnist_byclass.joblib"
    )

    print("Loading EMNIST/byclass ...")
    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test  = tfds.load("emnist/byclass", split="test",  as_supervised=True)

    # map preprocessing (unbatched ist hier okay, wir ziehen danach in numpy)
    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess,  num_parallel_calls=tf.data.AUTOTUNE)

    # Optional: deterministische Reihenfolge, aber wir wollen für RF eher mischen.
    # shuffle im tf.data bevor wir nach numpy ziehen:
    ds_train = ds_train.shuffle(50_000, seed=SEED, reshuffle_each_iteration=False)

    print(f"Converting train to numpy (max {MAX_TRAIN:,}) ...")
    X_train, y_train = ds_to_numpy_flat(ds_train, MAX_TRAIN)

    print(f"Converting test to numpy (max {MAX_TEST:,}) ...")
    X_test, y_test = ds_to_numpy_flat(ds_test, MAX_TEST)

    print("\nBuilding PCA + RandomForest pipeline ...")
    clf = Pipeline(steps=[
        ("pca", PCA(n_components=PCA_COMPONENTS, random_state=SEED)),
        ("rf", RandomForestClassifier(
            n_estimators=RF_TREES,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=-1,              # nutzt alle Threads deines 16C/32T
            random_state=SEED,
            verbose=1
        ))
    ])

    print("\nTraining ...")
    t0 = time.time()
    clf.fit(X_train, y_train)
    print(f"Training done in {(time.time() - t0)/60:.1f} min")

    print("\nEvaluating ...")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nSaving model pipeline ...")
    joblib.dump(clf, model_path)
    print("Saved:", model_path)

    # Mini sanity check: Laden & Predict
    loaded = joblib.load(model_path)
    print("\nSanity check (first 10 preds):", loaded.predict(X_test[:10]))


if __name__ == "__main__":
    main()
