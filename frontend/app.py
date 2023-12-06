import streamlit as st
import pandas as pd
from PIL import Image
from io import BytesIO
import time
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
import requests
import json

st.set_page_config(
    page_title="RePlottr",
    page_icon="📈",
    layout="wide",
)

st.title("RePlottr 📈")
st.subheader("Making data extraction from scatterplots easy")

# Production API URL
API_URL = "https://donutplot-uz3lg33nzq-no.a.run.app"
API_URL = "http://127.0.0.1:8000"

img_file_buffer = st.file_uploader("Upload an image to get started")


def main():
    pass


if img_file_buffer is not None:
    col1, col2 = st.columns([2, 3])

    img_bytes = img_file_buffer.getvalue()

    ### Make request to  API (stream=True to stream response as bytes)

    with col1:
        ### Display the image user uploaded
        st.image(
            Image.open(img_file_buffer),
            caption="Here's the image you uploaded ☝️",
        )

    with col2:
        with st.spinner("Wait for it..."):
            res = requests.post(API_URL + "/predict", files={"img": img_bytes})
            ### Get bytes from the file buffer
            if res.status_code == 200:
                # Assuming 'res' is your response object
                data_dicts = res.json()["prediction"]["data_dicts"]
                plot_title = res.json()["prediction"]["title"]
                x_label = res.json()["prediction"]["x_label"]
                y_label = res.json()["prediction"]["y_label"]

                # Transform data into a DataFrame
                all_data = []
                for data_dict in data_dicts:
                    for x, y in zip(data_dict["x_values"], data_dict["y_values"]):
                        all_data.append(
                            {"x_values": x, "y_values": y, "Marker": data_dict["marc"]}
                        )

                df = pd.DataFrame(all_data)

                # Allow users to edit title and labels
                editable_title = st.text_input("Edit Plot Title", value=plot_title)
                col1, col2 = st.columns(2)
                with col1:
                    editable_x_label = st.text_input("Edit X-Axis Label", value=x_label)
                with col2:
                    editable_y_label = st.text_input("Edit Y-Axis Label", value=y_label)

                col1, col2 = st.columns([1, 2])
                with col1:
                    # Allow users to edit the DataFrame
                    edited_df = st.data_editor(df, num_rows="dynamic")

                with col2:
                    # Create a scatter plot with edited values
                    fig = px.scatter(
                        edited_df,
                        x="x_values",
                        y="y_values",
                        color="Marker",
                        title=editable_title,
                        labels={
                            "x_values": editable_x_label,
                            "y_values": editable_y_label,
                        },
                    )

                    # Display the plot in Streamlit
                    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

                # Function to generate JSON data
                def generate_json():
                    # Convert DataFrame to JSON string
                    df_json = edited_df.to_json(orient="records")

                    # Combine everything into a single dictionary
                    output_json = {
                        "title": editable_title,
                        "x_label": editable_x_label,
                        "y_label": editable_y_label,
                        "data": json.loads(df_json),
                    }

                    # Store in session state
                    st.session_state["download_json"] = json.dumps(
                        output_json, indent=4
                    )

                # Button to prepare download
                if st.button("Prepare JSON Download"):
                    generate_json()

                # Download button
                if "download_json" in st.session_state:
                    st.toast("Download is Ready 🎉")
                    st.download_button(
                        label="Download as JSON",
                        data=st.session_state["download_json"],
                        file_name="edited_data.json",
                        mime="application/json",
                    )

            else:
                st.warning("**Oops**, something went wrong 😓 Please try again.")
                print(res.status_code, res.content)

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
