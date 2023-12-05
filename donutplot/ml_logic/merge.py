from sklearn.linear_model import LinearRegression
import numpy as np

markers = [
    ".",
    "o",
    "v",
    "^",
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "*",
    "h",
    "H",
    "+",
    "x",
    "D",
    "d",
]


def fit_the_scale(tick_coordinates, tick_values):
    tick_Xy = np.ndarray((0, 2))
    for i, x in enumerate(tick_values):
        if x != "":
            tick_Xy = np.vstack(
                (tick_Xy, np.array((float(tick_coordinates[i]), float(x))))
            )
    model = LinearRegression()
    model.fit(X=tick_Xy[:, 0].reshape(-1, 1), y=tick_Xy[:, 1].reshape(-1, 1))
    return model


def merge(yolo_output, x_tick_values, y_tick_values):
    x_tick_coords = np.sort(yolo_output[yolo_output[:, 0] == 0, 2])
    y_tick_coords = np.sort(yolo_output[yolo_output[:, 0] == 1, 3])
    scatterpoints = yolo_output[yolo_output[:, 0] > 1, :]

    x_model = fit_the_scale(x_tick_coords, x_tick_values)
    y_model = fit_the_scale(y_tick_coords, y_tick_values)

    series = list(set(scatterpoints[:, 0]))
    series_list = []

    for serie in series:
        serie_dict = {}
        serie_dict["marc"] = markers[int(serie) - 2]
        points = scatterpoints[scatterpoints[:, 0] == serie, :]
        serie_dict["x_values"] = x_model.predict(points[:, 2].reshape(-1, 1))
        serie_dict["y_values"] = y_model.predict(points[:, 3].reshape(-1, 1))
        series_list.append(serie_dict)

    return series_list
