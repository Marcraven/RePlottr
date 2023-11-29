import numpy as np
import cv2
import os
import comet_ml


from ultralytics import YOLO


def train_model():
    comet_ml.init()
    # Load the pre-trained model
    model = YOLO("yolov8s-p2.yaml").load("ObjectRecognition/yolov8s.pt")
    currentdir = os.path.dirname(os.path.abspath(__file__))
    # Train the model
    model.train(
        data=currentdir + "/dataset.yaml",
        # data="dataset.yaml",
        epochs=10,
        imgsz=320,
        save=True,  # device="gpu"
    )  # Set imgsz to 320 for training on 320xsomething images

    # Export the model to ONNX format

    path = model.export()
    print(path)


train_model()
