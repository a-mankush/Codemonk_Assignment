import os
import shutil

import gdown
import tensorflow as tf


def download_folder_from_google_drive(folder_id, output_path):
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    gdown.download_folder(url, output=output_path, quiet=False, use_cookies=False)


def download_model():
    # The ID of the shared Google Drive folder (skimLit_8b)
    folder_id = (
        "1OFAF-Frtv5Oe2EPiIQE0n4o9zrS0l2kR"  # Replace this with the actual folder ID
    )
    output_path = "models"

    # Download the model folder
    print(f"Downloading {output_path} folder...")
    try:
        download_folder_from_google_drive(folder_id, output_path)
    except Exception as e:
        print(f"Error downloading folder: {e}")
        return

    # Check if the folder was downloaded and has content
    if not os.path.exists(output_path) or not os.listdir(output_path):
        print(
            f"Error: The downloaded folder '{output_path}' is empty or does not exist."
        )
        return


