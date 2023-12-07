##### Train mode (data generation) ######
TRAINING_MODE = True
FIGSIZE_WIDTH_TRAINING_MODE = 3.2
FIGSIZE_HEIGHT_TRAINING_MODE = 2.4
DPI_TRAINING_MODE = 300

EPOCHS = 1
BATCH_SIZE = 3
IMGSZ = 960
PATIENCE = 10


##### Data generation ######
TRAIN_SIZE = 100
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

##### Yolo target draw boxes #####
SOURCE_PATH = "./data/train/"
SAVE_PATH = "./data/train/boxed/"
TEST_PATH = "./data/test/"
BEST_PT_PATH = "donutplot/ml_logic/yolo/"
