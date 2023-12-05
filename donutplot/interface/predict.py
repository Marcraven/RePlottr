from donutplot.ml_logic.yolo.yolo_pred import YoloModel
from donutplot.ml_logic.ocr.ocr import (
    read_title,
    read_x_axis_label,
    read_y_axis_label,
    read_ticks,
)
from donutplot.ml_logic.merge import merge
import json

model = YoloModel()


def make_prediction(image):
    yolo_data, yolo_xticks, yolo_yticks = model.predict(image)
    columns = max(len(yolo_xticks), len(yolo_yticks))

    x_ticks_values = []
    y_ticks_values = []

    for i, box in enumerate(yolo_xticks):
        x_ticks_values.append(read_ticks(box))
        conf = round(float(yolo_data[yolo_data[:, 0] == 0, :][i, 1]), 2)

    for j, box in enumerate(yolo_yticks):
        y_ticks_values.append(read_ticks(box))
        conf = round(float(yolo_data[yolo_data[:, 0] == 1, :][j, 1]), 2)

    data_dicts = merge(yolo_data, x_ticks_values, y_ticks_values)

    title = read_title(image)
    x_label = read_x_axis_label(image)
    y_label = read_y_axis_label(image)

    # File path for the JSONL file
    # file_path = './api/temp/'' + "metadata.json"

    # Writing data to the JSONL file
    # with open(file_path, "w") as file:
    #     json.dump(output, file_path, default=str)  # Use str() for non-serializable objects
    #     file.write("\n")  # Add a newline character to separate JSON objects

    response = {
        "data_dicts": data_dicts,
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
    }

    return response


if __name__ == "__main__":
    make_prediction()
