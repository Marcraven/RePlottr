import os


##### Train data ######
TRAIN_SIZE = 10
VAL_SPLIT = 0.125
TEST_SPLIT = 0.125

XLIM_LOW = 0
XLIM_HIGH = 2_000
YLIM_LOW = 0
YLIM_HIGH = 2_000

NUM_SERIES_MIN = 4
NUM_SERIES_MAX = 6

NUM_POINTS_MIN = 20
NUM_POINTS_MAX = 40

START_INDEX = 0

FIGSIZE_WIDTH = 3.2
FIGSIZE_HEIGHT = 2.4
FIGSIZE_DPI = 100


##### Yolo target draw boxes #####
SOURCE_PATH = "data/train/"
SAVE_PATH = "data/train/boxed/"
