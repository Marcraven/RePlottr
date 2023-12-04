import numpy as np
import comet_ml
import os
import sys
from comet_ml import API
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import save_one_box
import comet_ml
import torch
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(__file__)) + "/"
workspace = os.environ["WORKSPACE"]
model_name = os.environ["MODEL_NAME"]
project = os.environ["COMET_PROJECT_NAME"]


class YoloModel:
    def __init__(self, initial_weights_path=currentdir + "best.pt") -> None:
        self.weights = initial_weights_path

    def train(self):
        """This loads a model and then trains it, the results are saved into Comet"""
        yolo = self.load()
        comet_ml.init()
        hyper_params = {
            "patience": 10,
            "epochs": 20,
            "batch_size": 16,
            "imgsz": 320,
        }

        # Train the model
        yolo.train(
            data=currentdir + "dataset.yaml",
            name=model_name,
            project=project,
            amp=False,
            epochs=hyper_params["epochs"],
            patience=hyper_params["patience"],
            batch=hyper_params["batch_size"],
            imgsz=hyper_params["imgsz"],
            save=True,
        )
        self.save(yolo)

    def predict(self, scatterpath):
        """This gives a prediction of the image found in scatterpath"""
        yolo = self.load()
        results = yolo.predict(
            scatterpath,
            save=False,
            imgsz=320,
            # save_txt=True,
            # save_conf=True,
            save_frames=True,
            save_crop=True,
        )

        x_tick_box = []
        y_tick_box = []

        data = results[0].boxes.data
        xywhn = torch.cat(
            (
                results[0].boxes.data[:, -1].unsqueeze(1),
                results[0].boxes.data[:, -2].unsqueeze(1),
                results[0].boxes.xywhn,
            ),
            axis=1,
        )

        sort_column_index = 0

        x_tick_data = data[data[:, -1] == 0, :]
        x_tick_indices = torch.argsort(x_tick_data[:, sort_column_index], dim=0)
        sorted_x_ticks = x_tick_data[x_tick_indices]

        sort_column_index = 1

        y_tick_data = data[data[:, -1] == 1, :]
        y_tick_indices = torch.argsort(y_tick_data[:, sort_column_index], dim=0)
        sorted_y_ticks = y_tick_data[y_tick_indices]

        for box in sorted_x_ticks:
            x_tick_box.append(
                save_one_box(box[:4], results[0].orig_img, save=False, gain=0.6)
            )

        for box in sorted_y_ticks:
            y_tick_box.append(
                save_one_box(box[:4], results[0].orig_img, save=False, gain=0.65)
            )

        results[0].boxes.data
        return xywhn, x_tick_box, y_tick_box

    def load(self) -> YOLO:
        """This function loads the YOLO model given by the path initialized
        Otherwise it will download the weights from the latest version on Comet"""
        if self.weights is not None:
            return YOLO(self.weights)
        else:
            api = API()
            models = api.get_model(workspace=workspace, model_name=model_name)
            last_version = models.find_versions()[0]
            version_path = currentdir + "weights/" + last_version.replace(".", "_")
            if os.path.exists(version_path) == False:
                os.makedirs(version_path)
                print("Downloading latest version...")
                models.download(
                    version=last_version,
                    output_folder=version_path,
                    expand=True,
                )
            return YOLO(version_path + "/best.pt")

    def save(self, yolo) -> YOLO:
        """This function saves the YOLO model locally and in Comet"""
        yolo.export()
        api = API()
        experiments = api.get(workspace=workspace, project_name=project)
        experiment = api.get(
            workspace=workspace,
            project_name=project,
            experiment=experiments[-1]._name,
        )
        experiment.register_model(model_name)


if __name__ == "__main__":
    """Take the argument after .py as an image to be predicted or train the model otherwise"""
    if len(sys.argv) == 2:
        model = YoloModel()
        model.predict(sys.argv[1])
    else:
        model = YoloModel()
        model.train()
