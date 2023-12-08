from fastapi import FastAPI, UploadFile, File, Body
from typing import List, Any
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2
import numpy as np
import os

from donutplot.interface.predict import make_prediction, make_prediction_manual
from donutplot.ml_logic.merge import merge, merge_manual

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
def hello_world():
    return {"Hello World"}


@app.post("/predict")
async def receive_image(img: UploadFile = File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # type(cv2_img) => numpy.ndarray

    image_directory = "donutplot/api/temp/"
    filename = image_directory + "temp.jpg"
    cv2.imwrite(filename, cv2_img)

    image_path = "donutplot/api/temp/temp.jpg"
    response = make_prediction(image_path)

    # os.remove(filename) for file in os.listdir(image_directory) if file.endswith('.jpg')

    return response


@app.post("/process_ticks")
async def process_ticks(
    x_ticks: List[str] = Body(...),
    y_ticks: List[str] = Body(...),
    yolo_data: Any = Body(...),  # Use 'Any' if the structure is not defined
    title: str = Body(...),
    x_label: str = Body(...),
    y_label: str = Body(...),
):
    yolo_data = np.array(yolo_data).reshape(-1, 6)
    data_dicts = merge_manual(yolo_data, x_ticks, y_ticks)
    response = make_prediction_manual(data_dicts, title, x_label, y_label)

    return response
