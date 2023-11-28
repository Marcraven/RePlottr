import numpy as np
import cv2
import os


from ultralytics import YOLO


def train_model():
    # Load the pre-trained model
    model = YOLO("yolov8s-p2.yaml").load("yolov8s.pt")
    currentdir = os.path.dirname(os.path.abspath(__file__))
    # Train the model
    model.train(
        data=currentdir + "/dataset.yaml",
        epochs=200,
        imgsz=320,
        save=True,
        format="onnx",
    )  # Set imgsz to 320 for training on 320xsomething images

    # Export the model to ONNX format
    path = model.export()
    print(path)


train_model()
