import streamlit as st
import pandas as pd
from PIL import Image
import requests
import json
import plotly.express as px

# Page configuration
st.set_page_config(page_title="RePlottr", page_icon="📈", layout="wide")

# Page title and subheader
st.title("RePlottr 📈")
st.subheader("Making data extraction from scatterplots easy")

# Production API URL
# API_URL = "https://donutplot-uz3lg33nzq-no.a.run.app"
# Fallback API URL (for testing purposes)
API_URL = "http://127.0.0.1:8000"

# File uploader
img_file_buffer = st.file_uploader("Upload an image to get started")


def display_uploaded_image(image_buffer):
    st.image(Image.open(image_buffer), caption="Here's the image you uploaded ☝️")


def make_api_request(url, img_bytes):
    return requests.post(url, files={"img": img_bytes})


def process_api_response(response):
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Failed to process image. Please try again.")
        return None


def request_tick_data(api_url, tick_data):
    return requests.post(api_url + "/process_ticks", json=tick_data)


def generate_plot(df, plot_title, x_label, y_label):
    fig = px.scatter(
        df,
        x="x_values",
        y="y_values",
        color="Marker",
        title=plot_title,
        labels={"x_values": x_label, "y_values": y_label},
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


def generate_json(df, title, x_label, y_label):
    df_json = df.to_json(orient="records")
    output_json = {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data": json.loads(df_json),
    }
    st.session_state["download_json"] = json.dumps(output_json, indent=4)


def main():
    if img_file_buffer is not None:
        col1, col2 = st.columns([2, 3])
        img_bytes = img_file_buffer.getvalue()

        with col1:
            display_uploaded_image(img_file_buffer)

        with col2:
            with st.spinner("Processing ... 🤔"):
                response = make_api_request(API_URL + "/predict", img_bytes)
                response_data = process_api_response(response)

            if response_data:
                handle_response(response_data)


def handle_response(data):
    if data.get("status") == "input_required":
        handle_input_required(data)
    elif data.get("status") == "success":
        handle_success(data)


def handle_input_required(data):
    st.warning(data["message"])
    x_ticks = st.text_input(
        "Enter the first two **X ticks** (left to right, comma-separated):"
    )
    y_ticks = st.text_input(
        "Enter the first two **Y ticks** (bottom to top, comma-separated):"
    )
    if st.button("Submit Tick Values"):
        process_ticks(x_ticks, y_ticks, data)


def process_ticks(x_ticks, y_ticks, data):
    tick_data = {
        "x_ticks": x_ticks.split(","),
        "y_ticks": y_ticks.split(","),
        "yolo_data": data["yolo"],
        "title": data["title"],
        "x_label": data["x_label"],
        "y_label": data["y_label"],
    }
    tick_res = request_tick_data(API_URL, tick_data)
    if tick_res.status_code == 200:
        handle_success(tick_res.json())


def handle_success(data):
    df = pd.DataFrame(
        [
            {"x_values": x, "y_values": y, "Marker": d["marc"]}
            for d in data["prediction"]["data_dicts"]
            for x, y in zip(d["x_values"], d["y_values"])
        ]
    )
    editable_title = st.text_input("Edit Plot Title", value=data["prediction"]["title"])
    col1, col2 = st.columns(2)
    with col1:
        editable_x_label = st.text_input(
            "Edit X-Axis Label", value=data["prediction"]["x_label"]
        )
    with col2:
        editable_y_label = st.text_input(
            "Edit Y-Axis Label", value=data["prediction"]["y_label"]
        )
    col1, col2 = st.columns([1, 2])
    with col1:
        edited_df = st.data_editor(df, num_rows="dynamic")
    with col2:
        generate_plot(edited_df, editable_title, editable_x_label, editable_y_label)
    if st.button("Prepare JSON Download"):
        generate_json(edited_df, editable_title, editable_x_label, editable_y_label)
    if "download_json" in st.session_state:
        st.toast("Download is Ready 🚀")
        st.download_button(
            label="Download as JSON",
            data=st.session_state["download_json"],
            file_name="edited_data.json",
            mime="application/json",
        )


if __name__ == "__main__":
    main()


ft = """
<style>
a:link , a:visited{
color: #BFBFBF;  /* theme's text color hex code at 75 percent brightness*/
background-color: transparent;
text-decoration: none;
}

a:hover,  a:active {
color: #0283C3; /* theme's primary color*/
background-color: transparent;
text-decoration: underline;
}

#page-container {
  position: relative;
  min-height: 10vh;
}

footer{
    visibility:hidden;
}

.footer {
position: fixed;  /* Changed from 'relative' to 'fixed' */
left: 0;
bottom: 0;  /* Anchors the footer to the bottom */
width: 100%;
background-color: transparent;
color: #808080; /* theme's text color hex code at 50 percent brightness*/
text-align: center; /* you can replace 'left' with 'center' or 'right' if you want*/
}
</style>

<div id="page-container">

<div class="footer">
<p style='font-size: 0.875em;'>From the Bunker with <img src="https://em-content.zobj.net/source/skype/289/red-heart_2764-fe0f.png" alt="heart" height= "10"/><br 'style= top:3px;'>
<a style='display: inline; text-align: left;' href="https://github.com/Marcraven/" target="_blank">Marc</a>, <a style='display: inline; text-align: left;' href="https://github.com/MaxiaB" target="_blank">Maxi</a>, <a style='display: inline; text-align: left;' href="https://github.com/nemesbence94" target="_blank">Bence</a> & <a style='display: inline; text-align: left;' href="https://github.com/marcelo-nora" target="_blank">Marcelo</a></p>
</div>

</div>
"""
st.write(ft, unsafe_allow_html=True)
