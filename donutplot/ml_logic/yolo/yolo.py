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
import datetime
from donutplot.params import *

currentdir = os.path.dirname(os.path.abspath(__file__)) + "/"


class YoloModel:
    def __init__(self, initial_weights_path=currentdir + "best.pt") -> None:
        self.weights = initial_weights_path

    def train(self):
        """This loads a model and then trains it, the results are saved into Comet"""
        self.load()
        comet_ml.init()
        hyper_params = {
            "patience": PATIENCE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "imgsz": IMGSZ,
        }
        workspace, model_name, project = self.load_environ()
        # Train the model
        self.yolo.train(
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
        self.save()

    def predict(self, scatterpath):
        """This gives a prediction of the image found in scatterpath"""
        yolo = self.load()
        now = datetime.datetime.now()
        results = yolo.predict(
            scatterpath,
            save=False,
            imgsz=320,
            # save_txt=True,
            # save_conf=True,
            # save_frames=True,
            # save_crop=True,
        )
        print(datetime.datetime.now() - now)
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
                save_one_box(box[:4], results[0].orig_img, save=False, gain=1)
            )

        for box in sorted_y_ticks:
            y_tick_box.append(
                save_one_box(box[:4], results[0].orig_img, save=False, gain=1)
            )

        results[0].boxes.data

        return xywhn.cpu().numpy(), x_tick_box, y_tick_box

    def load(self) -> YOLO:
        """This function loads the YOLO model given by the path initialized
        Otherwise it will download the weights from the latest version on Comet"""
        print("Loading models...")
        if self.weights is not None:
            self.local = YOLO(self.weights)
        # else:
        api = API()
        workspace, model_name, project = self.load_environ()
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
        self.comet = YOLO(version_path + "/best.pt")
        mp50prod = self.comet.val(
            data=currentdir + "dataset.yaml", split="test"
        ).results_dict["metrics/mAP50(B)"]
        mp50local = self.local.val(
            data=currentdir + "dataset.yaml", split="test"
        ).results_dict["metrics/mAP50(B)"]
        if mp50prod < mp50local:
            self.yolo = self.local
            self.chosen = "local"
            self.mp50 = mp50prod
            print("Local model was best and was selected.")
        else:
            self.yolo = self.comet
            self.chosen = "comet"
            self.mp50 = mp50local
            print("Comet model was best and was selected.")

    def save(self) -> YOLO:
        """This function saves the YOLO model locally and in Comet"""
        result = self.local.val(
            data=currentdir + "dataset.yaml", split="test"
        ).results_dict["metrics/mAP50(B)"]
        if result > self.mp50:
            print("Model is better than previous. Registering in Comet.")
            workspace, model_name, project = self.load_environ()
            self.yolo.export()
            api = API()
            experiments = api.get(workspace=workspace, project_name=project)
            experiment = api.get(
                workspace=workspace,
                project_name=project,
                experiment=experiments[-1]._name,
            )
            experiment.register_model(
                model_name,
                metric=result,
                status="Production",
                description="mp50 = " + str(result),
            )
        else:
            print("Previous model was better. Results not saved in Comet.")

    def load_environ(self):
        workspace = os.environ["WORKSPACE"]
        model_name = os.environ["MODEL_NAME"]
        project = os.environ["COMET_PROJECT_NAME"]
        return workspace, model_name, project


if __name__ == "__main__":
    """Take the argument after .py as an image to be predicted or train the model otherwise"""
    if len(sys.argv) == 2:
        model = YoloModel()
        model.predict(sys.argv[1])
    else:
        model = YoloModel()
        model.train()
