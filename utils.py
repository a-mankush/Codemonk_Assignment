import pickle

import tensorflow as tf


def load_encoder(path):
    with open(path, "rb") as f:
        saved_loaded_encoders_dict = pickle.load(f)
    return saved_loaded_encoders_dict


def decode_and_resize(uploaded_file, target_size=(224, 224)):

    # 1. Read the file bytes
    image_bytes = uploaded_file.read()

    # 2. Decode image from bytes
    image = tf.image.decode_image(image_bytes, channels=3)

    # 3. Convert to float32 and normalize
    image = tf.image.convert_image_dtype(image, tf.float32)

    # 4. Resize to target size
    image = tf.image.resize(image, target_size)

    return image
