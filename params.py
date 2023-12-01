import os


##### Imports from .env ######
# COMET_API = os.environ["COMET_API"]
# COMET_API_KEY = os.environ["COMET_API_KEY"]
# WORKSPACE = os.environ["WORKSPACE"]
# MODEL_NAME = os.environ["MODEL_NAME"]
# COMET_PROJECT_NAME = os.environ["COMET_PROJECT_NAME"]


##### Train data ######
TRAIN_SIZE = 20
VAL_SPLIT = 0.125
TEST_SPLIT = 0.125

XLIM_LOW = 0
XLIM_HIGH = 1_000
YLIM_LOW = 0
YLIM_HIGH = 1_000

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
