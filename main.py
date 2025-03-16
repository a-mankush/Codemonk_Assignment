import os
import subprocess
from pathlib import Path

import gdown
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt

from download_model import download_folder_from_google_drive, download_model
from utils import decode_and_resize, load_encoder

loaded_encoders_dict = load_encoder("artifacts/saved_encoders_dictionary.pkl")

model_path = Path("models/multioutput_model_resnet_v2.keras")


def fetch_model():
    if not model_path.exists():
        result = download_model()
        st.info("model is present")
        return True
    elif model_path.exists():
        return True


@st.cache_resource()
def load_model():
    return tf.keras.models.load_model("models/multioutput_model_resnet_v2.keras")


def prediction_and_display(abstract):
    bar = st.progress(0)
    if fetch_model():
        model = load_model()
    else:
        st.error("Model is not able to load")

    st.title("Fashion Product Predictor")
    uploaded_file = st.file_uploader("Upload an image...")
    if uploaded_file:
        preds_probs = model.predict(
            tf.constant([decode_and_resize(uploaded_file).numpy()])
        )
        preds = [tf.argmax(pred, axis=1) for pred in preds_probs]

        out = ["articleType", "baseColour", "gender", "season"]
        for p, o in zip(preds, out):
            # print(t)
            # print(p)
            print(
                f"Predictions {o}: ",
                loaded_encoders_dict[o].inverse_transform(p),
                end=" ,",
            )


def main():
    if fetch_model():
        model = load_model()
    else:
        st.error("Model is not able to load")

    st.title("Fashion Product Predictor")
    uploaded_file = st.file_uploader("Upload an image...")
    if uploaded_file:
        preds_probs = model.predict(
            tf.constant([decode_and_resize(uploaded_file).numpy()])
        )
        preds = [tf.argmax(pred, axis=1) for pred in preds_probs]
        image = plt.imread(uploaded_file)
        plt.imshow(image)
        plt.show()

        out = ["articleType", "baseColour", "gender", "season"]
        for p, o in zip(preds, out):
            # print(t)
            # print(p)
            print(
                f"Predictions {o}: ",
                loaded_encoders_dict[o].inverse_transform(p),
                end=" ,",
            )


if __name__ == "__main__":
    main()
