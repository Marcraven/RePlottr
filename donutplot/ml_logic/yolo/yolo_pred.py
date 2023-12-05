from ultralytics import YOLO
from ultralytics.engine.results import save_one_box
import torch
from donutplot.params import *

best_pt_path = BEST_PT_PATH


class YoloModel:
    def __init__(self, initial_weights_path=best_pt_path + "best.pt") -> None:
        self.weights = initial_weights_path

    def predict(self, scatterpath):
        """This gives a prediction of the image found in scatterpath"""
        yolo = YOLO(self.weights)

        results = yolo.predict(
            scatterpath,
            save=False,
            # imgsz=320,
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
                save_one_box(box[:4], results[0].orig_img, save=False, gain=1)
            )

        for box in sorted_y_ticks:
            y_tick_box.append(
                save_one_box(box[:4], results[0].orig_img, save=False, gain=1)
            )

        return xywhn.cpu().numpy(), x_tick_box, y_tick_box
