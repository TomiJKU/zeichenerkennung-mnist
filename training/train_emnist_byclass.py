import os
import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess(example):
    x = tf.cast(example["image"], tf.float32) / 255.0  # (28,28,1)
    y = tf.cast(example["label"], tf.int32)

    # Optional: "human-friendly" orientation:
    # TFDS Hinweis: EMNIST ist gedreht/gespiegelt. 
    # Für das Modell nicht zwingend nötig, aber okay:
    #x = tf.transpose(x, perm=[1, 0, 2])

    return x, y

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "models", "best_model.keras")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=False)
    ds_test  = tfds.load("emnist/byclass", split="test", as_supervised=False)

    ds_train = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(10_000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test  = ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(62, activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(ds_train, epochs=10, validation_data=ds_test)
    model.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
