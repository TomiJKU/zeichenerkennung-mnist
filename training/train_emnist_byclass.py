import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

def show_sample(ds, title="Sample"):
    for x, y in ds.take(1):
        img = x.numpy().squeeze()     # (28,28)
        label = int(y.numpy())        # scalar
        plt.imshow(img, cmap="gray")
        plt.title(f"{title} | label={label}")
        plt.axis("off")
        plt.show()


def preprocess(x, y):
    x = tf.cast(x, tf.float32) / 255.0
    x = tf.transpose(x, perm=[1, 0, 2])  # 3D: (28,28,1) aktuelle version 16.01.26
    y = tf.cast(y, tf.int32)
    return x, y

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(project_root, "models", "emnist_cnn.keras")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    ds_train = tfds.load("emnist/byclass", split="train", as_supervised=True)
    ds_test  = tfds.load("emnist/byclass", split="test", as_supervised=True)

    # map zuerst (noch 3D)
    ds_train_mapped = ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test_mapped  = ds_test.map(preprocess,  num_parallel_calls=tf.data.AUTOTUNE)

    # DEBUG: Sample nach Preprocessing (ohne batch!)
    # ds_train_mapped ist (noch) ungebatcht, perfekt für Preview
    preview = ds_train_mapped.shuffle(10_000, reshuffle_each_iteration=True)
    show_sample(preview, title="Random EMNIST sample")

    # dann erst batchen
    ds_train_final = ds_train_mapped.shuffle(10_000).batch(128).prefetch(tf.data.AUTOTUNE)
    ds_test_final  = ds_test_mapped.batch(256).prefetch(tf.data.AUTOTUNE)

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
    model.fit(ds_train_final, epochs=1, validation_data=ds_test_final)
    model.save(out_path)
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
