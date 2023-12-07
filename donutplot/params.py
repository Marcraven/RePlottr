import os

##### Train mode (data generation) ######
TRAINING_MODE = True
FIGSIZE_WIDTH_TRAINING_MODE = 3.2
FIGSIZE_HEIGHT_TRAINING_MODE = 2.4
DPI_TRAINING_MODE = 300

EPOCHS = 20
BATCH_SIZE = 2
IMGSZ = 960
PATIENCE = 10


##### Data generation ######
TRAIN_SIZE = 5
VAL_SPLIT = 0.2
TEST_SPLIT = 0.2

XLIM_LOW = -1000
XLIM_HIGH = 2_000
YLIM_LOW = -1000
YLIM_HIGH = 2_000

NUM_SERIES_MIN = 4
NUM_SERIES_MAX = 6

NUM_POINTS_MIN = 40
NUM_POINTS_MAX = 60

START_INDEX = 0

##### Paths #####
DATA_PATH = os.path.expanduser("~/.donutplot/data/")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VALIDATE_PATH = os.path.join(DATA_PATH, "validate")
TEST_PATH = os.path.join(DATA_PATH, "test")


BEST_PT_PATH = os.path.expanduser(
    "~/donutplot/ml_logic/yolo/"
)  # "donutplot/ml_logic/yolo/"


SOURCE_PATH = TRAIN_PATH
BOX_PATH = os.path.join(DATA_PATH, "boxed")
