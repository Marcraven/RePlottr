from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response
import cv2
import numpy as np
import os

from donutplot.interface.predict import make_prediction

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


# reception, decoding, and processing of an image file.


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

    return {
        "prediction": response
    }  # Response(content=im.tobytes(), media_type="image/jpg")
