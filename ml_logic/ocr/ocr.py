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
    filtered_text = text[text.conf > 80]["text"]
    extracted_numbers = " ".join(
        " ".join(re.findall(r"\b\d+\.?\d*\b", str(item).strip()))
        for item in filtered_text
    )
    return extracted_numbers.strip()


# def image_read(image, digits_only=1):
#     # image = cv2.imread(img)
#     # if image is None:
#     #     raise FileNotFoundError(f"Unable to read the image at path: {img}")
#     rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     options = ""
#     if digits_only:
#         options = "outputbase digits"

#     text = pytesseract.image_to_string(
#         rgb,
#         config=options,
#     )

#     return text.strip()
