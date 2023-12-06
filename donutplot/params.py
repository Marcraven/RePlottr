##### Train data ######
TRAINING_MODE = True

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

EPOCHS = 50
BATCH_SIZE = 16
IMGSZ = 1920
PATIENCE = 10
##### Yolo target draw boxes #####
SOURCE_PATH = "./data/train/"
SAVE_PATH = "./data/train/boxed/"
BEST_PT_PATH = "donutplot/ml_logic/yolo/"
