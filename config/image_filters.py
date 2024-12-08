import cv2
import numpy as np
import streamlit as st
from PIL import Image

def high_pass_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.filter2D(gray, -1, kernel)
    return filtered_image

def low_pass_filter(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    filtered_image = cv2.filter2D(gray, -1, kernel)
    return filtered_image

def upload_image():
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "bmp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        return np.array(image)
    st.warning("Please upload an image file.")
    return None
