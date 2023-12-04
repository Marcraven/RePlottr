import os
import pytesseract
from pytesseract import Output
import cv2
import re


def image_to_text(boxes_input):
    labels = boxes_input[1]

    text_data = {}

    for i in os.listdir(labels):
        if i.endswith(".jpg") or i.endswith(".png"):
            file_path = os.path.join(labels, i)
            img = cv2.imread(file_path)
            text_data[i] = image_read(img)
    return text_data


def image_read(img):
    text = pytesseract.image_to_data(img, output_type=Output.DATAFRAME)
    filtered_text = text[text.conf > 0]["text"]
    extracted_numbers = " ".join(
        " ".join(re.findall(r"\b\d+\.?\d*\b", str(item).strip()))
        for item in filtered_text
    )
    return extracted_numbers.strip()
