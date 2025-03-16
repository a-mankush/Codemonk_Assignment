import pickle

import tensorflow as tf


def load_encoder(path):
    with open(path, "rb") as f:
        saved_loaded_encoders_dict = pickle.load(f)
    return saved_loaded_encoders_dict


def decode_and_resize(image_path, target_height=224, target_width=224):
    """
    Decode a JPEG image from a file path and resize it to target dimensions.

    Args:
        image_path (str): Path to the JPEG image file
        target_height (int): Target height for resizing, default 224
        target_width (int): Target width for resizing, default 224

    Returns:
        tf.Tensor: Decoded and resized image tensor with shape (target_height, target_width, 3)
                  and values normalized to [0, 1] range
    """
    # Read the file contents
    image_data = tf.io.read_file(image_path)

    # Decode the JPEG image
    image = tf.io.decode_jpeg(image_data, channels=3)

    # Convert to float32 and normalize to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Resize the image to target dimensions
    resized_image = tf.image.resize(image, [target_height, target_width])

    return resized_image
