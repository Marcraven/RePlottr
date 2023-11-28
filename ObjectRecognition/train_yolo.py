import numpy as np
import cv2
import os


from ultralytics import YOLO


def train_model():
    # Load the pre-trained model
    model = YOLO("yolov8s-p2.yaml").load("ObjectRecognition/yolov8s.pt")
    currentdir = os.path.dirname(os.path.abspath(__file__))
    pathdir = "ObjectRecognition/yolo/dataset/results"
    # Train the model
    model.train(
        data=currentdir + "/dataset.yaml",
        epochs=10,
        imgsz=320,
        save=True,
        format="onnx",
    )  # Set imgsz to 320 for training on 320xsomething images

    # Export the model to ONNX format
    export_path = pathdir + "/my_trained_model.onnx"
    path = model.export(export_path)
    print(path)


train_model()
