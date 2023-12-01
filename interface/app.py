import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px

st.set_page_config(
    page_title="DonutPlot",
    page_icon="🍩",
    layout="wide",
)

st.title("🍩 Donutplot")

df = pd.DataFrame(
    {
        "Velocity of Donut": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Sprinkles on Donut": [3, 1, 2, 5, 4, 6, 3, 7, 2, 5],
    }
)

uploaded_file = st.file_uploader("Please upload a single image")
if uploaded_file is not None:
    original = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    col1.header("Original Image")
    col1.image(original, use_column_width=True)
    fig_plotly = px.scatter(
        df,
        x="Velocity of Donut",
        y="Sprinkles on Donut",
        title="Crazy Donut Plot",
    )
    col2.header("Plotly Chart")
    col2.plotly_chart(fig_plotly)
    col2.button("Download JSON")
