import numpy as np
import cv2

colors = [
    (255, 0, 0),  # Red
    (0, 255, 0),  # Green
    (0, 0, 255),  # Blue
    (255, 255, 0),  # Yellow
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Cyan
    (128, 0, 0),  # Maroon
    (0, 128, 0),  # Green (dark)
    (0, 0, 128),  # Navy
    (128, 128, 0),  # Olive
    (128, 0, 128),  # Purple
    (0, 128, 128),  # Teal
    (255, 165, 0),  # Orange
    (128, 128, 128),  # Gray
    (255, 255, 255),  # White
    (0, 0, 0),  # Black
    (255, 192, 203),  # Pink
    (255, 140, 0),  # Dark Orange
    (0, 128, 128),  # Dark Teal
    (255, 20, 147),  # Deep Pink
    (0, 255, 0),  # Lime
    (173, 255, 47),  # Green Yellow
    (135, 206, 250),  # Sky Blue
    (128, 0, 0),  # Maroon (dark)
    (255, 215, 0),  # Gold
]


def draw_boxes(path: str = "ObjectRecognition/yolo/dataset/train/0000"):
    with open(path + ".txt", "r") as file:
        file_contents = file.read().splitlines()
    img = cv2.imread(path + ".jpg")
    x_pix = img.shape[1]
    y_pix = img.shape[0]
    for rowraw in file_contents:
        row = np.array([float(x) for x in rowraw.split()])
        cat = int(row[0])
        x1 = int(row[1] * x_pix)
        y1 = int(row[2] * y_pix)
        x2 = x1 + int(row[3] * x_pix)
        y2 = y1 + int(row[4] * y_pix)
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=colors[cat],
            thickness=1,
        )
    cv2.imwrite("test.jpg", img)


if __name__ == "__main__":
    draw_boxes("ObjectRecognition/yolo/dataset/train/0000")
