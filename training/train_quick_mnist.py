import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # Reproduzierbarkeit (optional)
    tf.random.set_seed(42)
    np.random.seed(42)

    # --- Pfade ---
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "best_model.keras")

    # --- MNIST laden ---
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # --- Preprocessing (muss zur App passen: 28x28x1, float32, 0..1) ---
    x_train = (x_train.astype("float32") / 255.0)[..., None]  # (N,28,28,1)
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    num_classes = 10

    # --- Kleines CNN (gut genug für Demo) ---
    model = keras.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5),
    ]

    # --- Training (schnell) ---
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        epochs=5,
        batch_size=128,
        callbacks=callbacks,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    # --- Speichern ---
    model.save(out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()
