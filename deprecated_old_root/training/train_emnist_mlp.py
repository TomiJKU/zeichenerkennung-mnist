import os
import tensorflow as tf
import tensorflow_datasets as tfds

NUM_CLASSES = 62

def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0  # (28,28,1)
    y = tf.cast(y, tf.int32)
    return x, y

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "models", "emnist_mlp.keras")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test  = tfds.load("emnist/byclass", split="test",  as_supervised=True)

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
                       .shuffle(10_000)\
                       .batch(256)\
                       .prefetch(tf.data.AUTOTUNE)

    ds_test  = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)\
                      .batch(512)\
                      .prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(ds_train, epochs=30, validation_data=ds_test)
    model.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
