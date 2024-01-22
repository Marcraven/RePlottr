from donutplot.ml_logic.yolo.yolo_pred import YoloModel
from donutplot.ml_logic.ocr.ocr import (
    read_title,
    read_x_axis_label,
    read_y_axis_label,
    read_ticks,
)
from donutplot.ml_logic.merge import merge



model = YoloModel()


def make_prediction(image):
    yolo_data, yolo_xticks, yolo_yticks = model.predict(image)
    columns = max(len(yolo_xticks), len(yolo_yticks))

    x_ticks_values = []
    y_ticks_values = []

    title = read_title(image)
    x_label = read_x_axis_label(image)
    y_label = read_y_axis_label(image)

    for i, box in enumerate(yolo_xticks):
        x_ticks_values.append(read_ticks(box))
        # conf = round(float(yolo_data[yolo_data[:, 0] == 0, :][i, 1]), 2)

    for j, box in enumerate(yolo_yticks):
        y_ticks_values.append(read_ticks(box))
        # conf = round(float(yolo_data[yolo_data[:, 0] == 1, :][j, 1]), 2)

    if (
        len([item for item in x_ticks_values if item != ""]) < 2
        or len([item for item in y_ticks_values if item != ""]) < 2
    ):
        yolo_data = yolo_data.reshape(1, -1).tolist()
        return {
            "status": "input_required",
            "message": "Unable to Read X and y Ticks. Please provide two X and two y ticks.",
            "yolo": yolo_data,
            "title": title,
            "x_label": x_label,
            "y_label": y_label,
        }
    else:
        data_dicts = merge(yolo_data, x_ticks_values, y_ticks_values)

    response = {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data_dicts": data_dicts,
    }

    return {"status": "success", "prediction": response}


def make_prediction_manual(data_dicts, title, x_label, y_label):
    data_dicts = data_dicts
    title = title
    x_label = x_label
    y_label = y_label

    response = {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "data_dicts": data_dicts,
    }

    return {"status": "success", "prediction": response}


if __name__ == "__main__":
    make_prediction()
