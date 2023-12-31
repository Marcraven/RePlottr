import numpy as np
import cv2
import os, os.path
from donutplot.params import SOURCE_PATH, BOX_PATH

##### Define and import constants #####
file_name = "0000"

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


##### Draw boxes on one single image #####
def draw_boxes(
    file_name: str = file_name,
    source_path: str = SOURCE_PATH,
    save_path: str = BOX_PATH,
):
    # Read TXT files and split the lines
    txt_file_name = os.path.join(source_path, f"{file_name}.txt")
    with open(txt_file_name, "r") as file:
        file_contents = file.read().splitlines()

    # Read JPG files and save contents and pixels
    img_file_name = os.path.join(source_path, f"{file_name}.jpg")
    img = cv2.imread(img_file_name)
    x_pix = img.shape[1]
    y_pix = img.shape[0]

    # Draw rectangles
    for rowraw in file_contents:
        row = np.array([float(x) for x in rowraw.split()])
        cat = int(row[0])
        x1 = int(row[1] * x_pix) - int(row[3] * x_pix * 0.5)
        y1 = int(row[2] * y_pix) - int(row[4] * y_pix * 0.5)
        x2 = x1 + int(row[3] * x_pix)
        y2 = y1 + int(row[4] * y_pix)
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color=colors[cat],
            thickness=1,
        )

    # Write output image
    output_name = os.path.join(save_path, f"{file_name}_boxed.jpg")
    cv2.imwrite(output_name, img)


##### If name = main #####
if __name__ == "__main__":
    print("Creating boxed files")
    os.makedirs(BOX_PATH, exist_ok=True)

    draw_boxes()
