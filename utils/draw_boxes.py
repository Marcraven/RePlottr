import numpy as np
from utils.draw_box import draw_boxes
import os, os.path


##### Draw boxes on all  images found in source folder and save them in save folder #####
def draw_boxes_all_files(
    source_path: str = "ObjectRecognition/yolo/dataset/train/",
    save_path: str = "ObjectRecognition/yolo/dataset/train/boxed/",
):
    # Create list with JPG files in source folder
    jpg_file_list = [file for file in os.listdir(source_path) if file.endswith(".jpg")]

    # Create more general list of file names and sort alphabetically
    file_list = [file.strip(".jpg") for file in jpg_file_list]
    file_list_sorted = sorted(file_list)

    # For each file, draw boxes around dots and ticks
    for file in file_list_sorted:
        draw_boxes(file_name=file, source_path=source_path, save_path=save_path)


##### If name = main #####
if __name__ == "__main__":
    train_folder = "ObjectRecognition/yolo/dataset/train/"
    boxed_folder = train_folder + "boxed/"

    print("Creating boxed files")
    os.makedirs(boxed_folder, exist_ok=True) if not os.path.exists(
        boxed_folder
    ) else None

    draw_boxes_all_files(train_folder, boxed_folder)
