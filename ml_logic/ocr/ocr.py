import os
import pytesseract
import cv2
import numpy as np


def read_title(image):
    image = cv2.imread(image)
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

    text = pytesseract.image_to_string(thr1, config="--psm 3 --oem 3")


def image_read(image, digits_only=1):
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

    options = ""
    if digits_only:
        options = "--psm 6 --oem 3 outputbase digits"

    thr1 = cv2.threshold(
        processed_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        thr1,
        config=options,
    )

    def clean_text(s):
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

    text = clean_text(text)

    return text
