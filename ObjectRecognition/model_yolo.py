import numpy as np
import comet_ml
import os
from comet_ml import API
from ultralytics import YOLO


def train_model():
    comet_ml.init()

    hyper_params = {
        "patience": 5,
        "epochs": 1,
        "batch_size": 16,
        "imgsz": 320,
    }

    # Train the model
    model.train(
        data=currentdir + "dataset.yaml",
        name=model_name,
        project=project,
        amp=False,
        epochs=hyper_params["epochs"],
        patience=hyper_params["patience"],
        batch=hyper_params["batch_size"],
        imgsz=hyper_params["imgsz"],
        save=False,  # device="gpu"
    )  # Set imgsz to 320 for training on 320xsomething images

    experiments = api.get(workspace=workspace, project_name=project)
    experiment = api.get(
        workspace=workspace,
        project_name=project,
        experiment=experiments[-1]._name,
    )
    experiment.register_model(model_name)


if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.abspath(__file__)) + "/"

    workspace = os.environ["WORKSPACE"]
    model_name = os.environ["MODEL_NAME"]
    project = os.environ["PROJECT"]

    api = API()
    models = api.get_model(workspace=workspace, model_name=model_name)
    last_version = models.find_versions()[0]
    version_path = currentdir + "/weights/" + last_version.replace(".", "_")
    if os.path.exists(version_path) == False:
        os.mkdir(version_path)
        print("Downloading latest version...")
        models.download(
            version=last_version,
            output_folder=version_path,
            expand=True,
        )

    model = YOLO("yolov8s-p2.yaml").load(version_path + "/best.pt")
    train_model()
