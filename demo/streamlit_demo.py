import streamlit as st
import os
import keras
import config
import io
from PIL import Image
import numpy as np
import time

if __name__ == "__main__":
    # define model
    clf_model = keras.saving.load_model(config.model_path)
    img_uploaded = st.file_uploader("Choose an image", type=["jpg", "png"])
    if img_uploaded is not None:
        # load image as bytes
        img_bytes = img_uploaded.read()
        # convert bytes to PIL image
        img = Image.open(io.BytesIO(img_bytes)).resize((config.img_width, config.img_heigth)) # 224x224
        st.image(img)
        img_array = keras.utils.img_to_array(img)
        batch_img = np.array([img]) # [1, 224, 224]
        start_time = time.time()
        predictions = clf_model.predict(batch_img) # [1, 100]
        end_time = time.time()
        predict_class = np.argmax(predictions, axis=1)[0]
        st.write(f"Label: {config.int2label[predict_class]}, inference time: {round((end_time - start_time) * 1000)}ms")