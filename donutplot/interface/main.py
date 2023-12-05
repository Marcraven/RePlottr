from donutplot.ml_logic.yolo.yolo_pred import YoloModel
from donutplot.ml_logic.ocr.ocr import (
    read_title,
    read_x_axis_label,
    read_y_axis_label,
    read_ticks,
)
from donutplot.ml_logic.merge import merge
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


model = YoloModel()

image = "data/train/0000.jpg"
yolo_data, yolo_xticks, yolo_yticks = model.predict(image)
columns = max(len(yolo_xticks), len(yolo_yticks))

x_ticks_values = []
y_ticks_values = []
plt.figure(1)

for i, box in enumerate(yolo_xticks):
    x_ticks_values.append(read_ticks(box))
    plt.subplot(2, columns, i + 1)
    plt.imshow(box)
    conf = round(float(yolo_data[yolo_data[:, 0] == 0, :][i, 1]), 2)
    plt.xlabel(f"value: {x_ticks_values[i]} \n conf_box:{conf}")
    plt.xticks([])
    plt.yticks([])

for j, box in enumerate(yolo_yticks):
    y_ticks_values.append(read_ticks(box))
    plt.subplot(2, columns, j + i + 2)
    plt.imshow(box)
    conf = round(float(yolo_data[yolo_data[:, 0] == 1, :][j, 1]), 2)
    plt.xlabel(f"value: {y_ticks_values[j]} \n conf_box:{conf}")
    plt.xticks([])
    plt.yticks([])


data_dicts = merge(yolo_data, x_ticks_values, y_ticks_values)

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

plt.figure(2)
for series in data_dicts:
    plt.scatter(series["x_values"], series["y_values"], marker=series["marc"])


title = read_title(image)
x_label = read_x_axis_label(image)
y_label = read_y_axis_label(image)

print(f"Title: {title}")
print(f"X Label: {x_label}")
print(f"y Label: {y_label}")
print(data_dicts)
plt.figure(3)
plt.imshow(mpimg.imread(image))
plt.show()