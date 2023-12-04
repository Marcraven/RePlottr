from ml_logic.yolo.yolo import YoloModel
from ml_logic.ocr.ocr import image_read
from ml_logic.merge import merge
import matplotlib.pyplot as plt

model = YoloModel()

image = "data/train/0002.jpg"
output = model.predict(image)
columns = max(len(output[1]), len(output[2]))

x_ticks_values = []
y_ticks_values = []

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
merge(output[0], x_ticks_values, y_ticks_values)

print(x_ticks_values)
print(y_ticks_values)

plt.show()
