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

    api = API()
    models = api.get_model(workspace="marcraven", model_name="yolov8")
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

    # temp = last_version.split(".")
    # temp[-1] = str(int(temp[-1]) + 1)
    # new_version = ".".join(temp)
    # new_version_path = currentdir + "/weights/" + new_version.replace(".", "_")

    # Train the model
    model.train(
        data=currentdir + "dataset.yaml",
        name="yolov8",
        project="yolo-donutplot-marcraven",
        amp=False,
        epochs=hyper_params["epochs"],
        patience=hyper_params["patience"],
        batch=hyper_params["batch_size"],
        imgsz=hyper_params["imgsz"],
        save=True,  # device="gpu"
        # save_dir=new_version_path,
    )  # Set imgsz to 320 for training on 320xsomething images

    experiments = api.get(
        workspace="marcraven", project_name="yolo-donutplot-marcraven"
    )
    experiment = api.get(
        workspace="marcraven",
        project_name="yolo-donutplot-marcraven",
        experiment=experiments[-1]._name,
    )
    experiment.register_model("yolov8")
    # path = model.export()
    # print(path)


if __name__ == "__main__":
    currentdir = os.path.dirname(os.path.abspath(__file__)) + "/"
    api = API()
    models = api.get_model(workspace="marcraven", model_name="yolov8")
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
