import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import requests

st.set_page_config(
    page_title="DonutPlot",
    page_icon="🍩",
    layout="wide",
)

st.title("🍩 Donutplot")

img_file_buffer = st.file_uploader("Upload an image")

if img_file_buffer is not None:
    col1, col2 = st.columns(2)

    with col1:
        ### Display the image user uploaded
        st.image(
            Image.open(img_file_buffer), caption="Here's the image you uploaded ☝️"
        )

    with col2:
        with st.spinner("Wait for it..."):
            ### Get bytes from the file buffer
            img_bytes = img_file_buffer.getvalue()

            url = ""

            ### Make request to  API (stream=True to stream response as bytes)
            res = requests.post(url + "/predict", files={"img": img_bytes})

            if res.status_code == 200:
                st.plotly_chart(res.json())
                ### Display the image returned by the API

            else:
                st.markdown("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)
