import os
import pytesseract
from pytesseract import Output
import cv2
import re

import ml_logic.yolo.model_yolo as yolo

model = yolo()
image = "data/train/0000.jpg"

output = model.predict(image)


def image_to_text(boxes_input):
    labels = boxes_input[1]

    text_data = {}

    for i in os.listdir(labels):
        if i.endswith(".jpg") or i.endswith(".png"):
            file_path = os.path.join(labels, i)
            img = cv2.imread(file_path)
            text = pytesseract.image_to_data(
                img, lang="eng", config="--psm 3", output_type=Output.DATAFRAME
            )
            filtered_text = text[text.conf > 85]["text"]
            extracted_numbers = " ".join(
                " ".join(re.findall(r"\b\d+\.?\d*\b", str(item).strip()))
                for item in filtered_text
            )
            text_data[i] = extracted_numbers.strip()
    return text_data
