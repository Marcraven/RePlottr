from ml_logic.yolo.yolo import YoloModel
from ml_logic.ocr.ocr import image_read
from ml_logic.merge import merge
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.image as mpimg

model = YoloModel()

image = "data/train/0006.jpg"
output = model.predict(image)
columns = max(len(output[1]), len(output[2]))

x_ticks_values = []
y_ticks_values = []
plt.figure(1)
for i, box in enumerate(output[1]):
    x_ticks_values.append(image_read(box))
    plt.subplot(2, columns, i + 1)
    plt.imshow(box)
    plt.xlabel(x_ticks_values[i])
    plt.xticks([])
    plt.yticks([])

for j, box in enumerate(output[2]):
    y_ticks_values.append(image_read(box))
    plt.subplot(2, columns, j + i + 2)
    plt.imshow(box)
    plt.xlabel(y_ticks_values[j])
    plt.xticks([])
    plt.yticks([])

data_dicts = merge(output[0], x_ticks_values, y_ticks_values)
print(data_dicts)
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
    # plt.subplot(1, 2, 2)
    plt.scatter(series["x_values"], series["y_values"], marker=series["marc"])
# plt.subplot(1, 2, 1)

plt.figure(3)
plt.imshow(mpimg.imread(image))
plt.show()
