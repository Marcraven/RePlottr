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

image = "data/train/0004.jpg"
yolo_data, yolo_xticks, yolo_yticks = model.predict(image)
columns = max(len(yolo_xticks), len(yolo_yticks))


plt.figure(1)

nticks = 0
confidence = 100
crop = 1
k = 0
while nticks < 2 and confidence >= 0:
    if k % 2 == 0:
        confidence -= 20
        crop = 1
    else:
        crop -= 0.50
    print(f"Trying xticks with confidence {confidence} and crop {crop}")
    x_ticks_values = []
    for i, box in enumerate(yolo_xticks):
        x_ticks_values.append(read_ticks(box, confidence=confidence, crop=crop))
    nticks = len([item for item in x_ticks_values if item != ""])
    k += 1


for i, box in enumerate(yolo_xticks):
    plt.subplot(2, columns, i + 1)
    plt.imshow(box)
    conf = round(float(yolo_data[yolo_data[:, 0] == 0, :][i, 1]), 2)
    plt.xlabel(
        f"value: {x_ticks_values[i]} \nconf_box: {conf} \nconf_ocr: {confidence}"
    )
    plt.xticks([])
    plt.yticks([])

nticks = 0
confidence = 100
crop = 1
k = 0
while nticks < 2 and confidence >= 0:
    if k % 2 == 0:
        confidence -= 20
        crop = 1
    else:
        crop -= 0.50
    print(f"Trying yticks with confidence {confidence} and crop {crop}")
    y_ticks_values = []
    for j, box in enumerate(yolo_yticks):
        y_ticks_values.append(read_ticks(box, confidence=confidence, crop=crop))
    nticks = len([item for item in y_ticks_values if item != ""])
    k += 1


for j, box in enumerate(yolo_yticks):
    plt.subplot(2, columns, j + i + 2)
    plt.imshow(box)
    conf = round(float(yolo_data[yolo_data[:, 0] == 1, :][j, 1]), 2)
    plt.xlabel(
        f"value: {y_ticks_values[j]} \nconf_box: {conf} \nconf_ocr: {confidence}"
    )
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
