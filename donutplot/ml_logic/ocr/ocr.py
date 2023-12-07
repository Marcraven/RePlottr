import pytesseract
import cv2
import numpy as np
import re
from pytesseract import Output


def preprocess_image(image):
    # resize image
    res = cv2.resize(image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    # clean image
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    clean = cv2.bitwise_not(gray)

    # noise reduction
    kernel = np.ones((1, 1), np.uint8)
    nrd = cv2.dilate(clean, kernel, iterations=1)
    nrd = cv2.erode(nrd, kernel, iterations=1)

    # filter color
    processed_image = cv2.bilateralFilter(nrd, 9, 75, 75)

    thr1 = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    return thr1


def clean_text(s):
    s = s.strip()
    # Remove trailing hyphens
    if s.endswith("-"):
        s = s.rstrip("-")

    # Remove trailing "-4"
    if s.endswith("-4"):
        s = s.rstrip("-4")

    # Remove initial dot
    if s.startswith("."):
        s = s.lstrip(".")

    # Remove trailing dot
    if s.endswith("."):
        s = s.rstrip(".")

    if "\n" in s:
        parts = s.split("\n")
        s = parts[1] if len(parts) >= 2 and parts[1].strip() else parts[0]

    s = s.strip()

    return s


def read_title(image):
    image = cv2.imread(image)
    image = preprocess_image(image)

    (height, width) = image.shape
    x = 0
    y = 0
    w = width
    h = round(height * 0.10)

    img_mrz = image[y : y + h, x : x + w]
    img_mrz = cv2.GaussianBlur(img_mrz, (3, 3), 0)
    ret, img_mrz = cv2.threshold(img_mrz, 127, 255, cv2.THRESH_TOZERO)

    text = pytesseract.image_to_string(img_mrz, config="--psm 6 --oem 3")
    text = text.strip()
    return text


def read_x_axis_label(image):
    image = cv2.imread(image)
    image = preprocess_image(image)

    (height, width) = image.shape
    w = width
    h = round(height * 0.15)
    x = 0
    y = height - h

    img_mrz = image[y : y + h, x : x + w]
    img_mrz = cv2.GaussianBlur(img_mrz, (3, 3), 0)
    ret, img_mrz = cv2.threshold(img_mrz, 127, 255, cv2.THRESH_TOZERO)

    text = pytesseract.image_to_string(img_mrz, config="--psm 6 --oem 3")

    # pattern = r"(\w+)\s*\[([^\]]+)\]"
    # label = re.findall(pattern, text)
    return text


def read_y_axis_label(image):
    image = cv2.imread(image)
    image = preprocess_image(image)

    (height, width) = image.shape
    w = round(width * 0.10)
    h = height
    x = 0
    y = 0

    img_mrz = image[y : y + h, x : x + w]
    img_mrz = cv2.GaussianBlur(img_mrz, (3, 3), 0)
    ret, img_mrz = cv2.threshold(img_mrz, 127, 255, cv2.THRESH_TOZERO)

    rotated_image = cv2.rotate(img_mrz, cv2.ROTATE_90_CLOCKWISE)

    text = pytesseract.image_to_string(rotated_image, config="--psm 6 --oem 3")

    # pattern = r"(\w+)\s*\[([^\]]+)\]"
    # label = re.findall(pattern, text)
    return text


def read_ticks(image, digits_only=1):
    image = preprocess_image(image)
    options = ""
    if digits_only:
        options = "--psm 6 --oem 3 outputbase digits"

    text = pytesseract.image_to_data(
        image, config=options, output_type=Output.DATAFRAME
    )

    filtered_text = text.loc[text["conf"] > 75, "text"]
    extracted_numbers = " ".join(
        " ".join(re.findall(r"-?\b\d+\b(?:\.\d+)?", str(item).strip()))
        for item in filtered_text
    )

    text_data = clean_text(extracted_numbers)

    return text_data


def read_ticks_string(image, digits_only=1):
    image = preprocess_image(image)

    options = ""
    if digits_only:
        options = "--psm 6 --oem 3 outputbase digits"

    text = pytesseract.image_to_string(
        image,
        config=options,
    )

    text = clean_text(text)

    return text
