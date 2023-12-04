from sklearn.linear_model import LinearRegression

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


def merge(yolo_output, x_tick_values, y_tick_values):
    x_tick_coords = yolo_output[yolo_output[:, 0] == 0, 2].sort()
    y_tick_coords = yolo_output[yolo_output[:, 0] == 1, 3].sort()
    scatterpoints = yolo_output[yolo_output[:, 0] > 1, :4].sort()

    x_dict = {}
    y_dict = {}

    for i, x in enumerate(x_tick_values):
        if x != "":
            x_dict[float(x_tick_coords[0][i])] = float(x)

    for j, y in enumerate(y_tick_values):
        if y != "":
            y_dict[float(y_tick_coords[0][j])] = float(y)

    x_model = LinearRegression()
    y_model = LinearRegression()

    x_model.fit(X=x_dict.keys(), y=x_dict.values())
    y_model.fit(X=y_dict.keys(), y=y_dict.values())

    series = set(scatterpoints[:, 0])
    series_list = []

    for serie in series:
        serie_dict = {}
        serie_dict["number"] = markers[int(serie) - 2]
        serie_dict["x_values"] = x_model.predict(serie[:, 1])
        serie_dict["y_values"] = y_model.predict(serie[:, 2])
        series_list.append(serie_dict)

    return series_list
