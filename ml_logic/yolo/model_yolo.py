import numpy as np
import comet_ml
import os
import sys
from comet_ml import API
import ultralytics
from ultralytics import YOLO
from ultralytics.engine.results import save_one_box
import comet_ml

currentdir = os.path.dirname(os.path.abspath(__file__)) + "/"
workspace = os.environ["WORKSPACE"]
model_name = os.environ["MODEL_NAME"]
project = os.environ["COMET_PROJECT_NAME"]


class yolo_model:
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
            save=True,
            imgsz=320,
            # save_txt=True,
            # save_conf=True,
            # save_frames=True,
            # save_crop=True,
        )

        x_tick_box = []
        y_tick_box = []
        for d in results[0].boxes:
            if d.data[:, -1] == 0:
                x_tick_box.append(
                    save_one_box(
                        d.xyxy,
                        results[0].orig_img,
                        save=False,
                    )
                )
            if d.data[:, -1] == 1:
                y_tick_box.append(
                    save_one_box(
                        d.xyxy,
                        results[0].orig_img,
                        save=False,
                    )
                )
        results[0].boxes.data
        breakpoint()
        return results[0].boxes.data.cpu().numpy(), x_tick_box, y_tick_box

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
        model = yolo_model()
        model.predict(sys.argv[1])
    else:
        model = yolo_model()
        model.train()
