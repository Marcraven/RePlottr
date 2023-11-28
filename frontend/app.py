import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import time

st.title("🍩 Donutplot")

uploaded_file = st.file_uploader("Please upload a single image")
if uploaded_file is not None:
    with st.spinner("Wait for it..."):
        # Open and display the original image
        original = Image.open(uploaded_file)
        col1, col2 = st.columns(2)
        col1.header("Original")
        col1.image(original, use_column_width=True)

        # Convert to grayscale and display
        grayscale = original.convert("LA")
        col2.header("Processed")
        col2.image(grayscale, use_column_width=True)


# Feature Request
# side by side comparison of original image vs. new plot
