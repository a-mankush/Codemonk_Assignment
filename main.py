import os
import subprocess
from pathlib import Path

import gdown
import streamlit as st
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

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
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])
    with st.expander("Some example text to test the model"):
        """
        amazon link: https://www.amazon.in/Allen-Solly-Regular-T-Shirt-ASKPQRGF701338_Medium/dp/B08KTVFFHF/?_encoding=UTF8&pd_rd_w=xZX3J&content-id=amzn1.sym.6a567e3d-fd9a-4932-aa05-d0107e1bcce7&pf_rd_p=6a567e3d-fd9a-4932-aa05-d0107e1bcce7&pf_rd_r=WNG42RV6S66NB0SHTKVD&pd_rd_wg=Uvn9b&pd_rd_r=d50d408a-84d0-4ba9-9870-b025c1b9122c&ref_=pd_hp_d_btf_a2i_gw_cml&th=1&psc=1
        """

    if uploaded_file:
        # Process image
        processed_img = decode_and_resize(uploaded_file)

        # Add batch dimension and predict
        preds_probs = model.predict(tf.expand_dims(processed_img, axis=0))

        # Get class predictions
        preds = [tf.argmax(pred, axis=1).numpy()[0] for pred in preds_probs]

        # Display image
        st.image(
            Image.open(uploaded_file),
            caption="Uploaded Image",
            width=300,  # Adjust this value to control size
            use_column_width=False,  # Disable auto-sizing
        )

        # Show predictions
        out = ["articleType", "baseColour", "gender", "season"]
        result = {}
        for p, o in zip(preds, out):
            decoded = loaded_encoders_dict[o].inverse_transform([p])[0]
            result[o] = decoded
            st.write(f"**{o}**: {decoded}")


if __name__ == "__main__":
    main()
