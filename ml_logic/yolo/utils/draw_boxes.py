import numpy as np
from ml_logic.yolo.utils.draw_box import draw_boxes
import os, os.path


##### Define and import constants #####
source_path = os.environ.get("SOURCE_PATH")
save_path = os.environ.get("SAVE_PATH")


##### Draw boxes on all  images found in source folder and save them in save folder #####
def draw_boxes_all_files(
    source_path: str = source_path,
    save_path: str = save_path,
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
    boxed_folder = os.environ.get("SAVE_PATH")

    print("Creating boxed files")
    os.makedirs(boxed_folder, exist_ok=True) if not os.path.exists(
        boxed_folder
    ) else None

    draw_boxes_all_files()
