from yacs.config import CfgNode as CN
_C = CN()
_C.MODEL = CN()
_C.MODEL.DEVICE_ID = '0,1'
_C.MODEL.NAME = 'efficientnet-b5'
_C.MODEL.LOSS_WAY = 'all'
_C.MODEL.MODEL_PATH = '/kaggle/input/leaf-diy-1/weights/'
_C.MODEL.WEIGHT_FROM = 'kaggle'
_C.MODEL.CLASSES = 5

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [386, 386]
# Size of the image during test
_C.INPUT.SIZE_TEST = [386, 386]
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Random probability for random erasing
_C.INPUT.RE_PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.MIXUP = False
# Value of padding size
_C.INPUT.PADDING = 10
_C.INPUT.GRID_PRO = 0.0
_C.INPUT.GRAY_RPO = 0.05

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = ('../data/cassava-leaf-disease-classification')
_C.DATASETS.ROOT_TEST = '/kaggle/input/cassava-leaf-disease-classification'
_C.DATASETS.HARD_AUG = 'auto'
_C.DATASETS.SPLIT = 0.3
_C.DATASETS.BATCH_SIZE = 32
_C.DATASETS.WORKERS = 2
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
# Name of optimizer
_C.SOLVER.OPTIMIZER_NAME = "SGD"
# Number of max epoches
_C.SOLVER.MAX_EPOCHS = 50
# Base learning rate
_C.SOLVER.BASE_LR = 0.001
# Whether using larger learning rate for fc layer
_C.SOLVER.LARGE_FC_LR = False
# Factor of learning bias
_C.SOLVER.BIAS_LR_FACTOR = 1
# Momentum
_C.SOLVER.MOMENTUM = 0.9
# Margin of triplet loss
_C.SOLVER.CENTER_LR = 0.2
# Balanced weight of center loss
_C.SOLVER.CENTER_LOSS_WEIGHT = 0.0003

_C.SOLVER.T_MAX = 5
_C.SOLVER.ETA_MIN = 0.001
_C.SOLVER.SWA = False
_C.SOLVER.SWA_START = [50, 60, 70, 80]
_C.SOLVER.SWA_ITER = 10
_C.SOLVER.SWA_MAX = 0.0045
_C.SOLVER.SWA_MIN = 0.001
_C.SOLVER.GRADUAL_UNLOCK = False
# Settings of weight decay
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
_C.SOLVER.GAMMA = 0.1
# decay step of learning rate
_C.SOLVER.STEPS = (25, 40)

_C.SOLVER.TYPE = 'warmup'
# warm up factor
_C.SOLVER.WARMUP_FACTOR = 0.01
#  warm up epochs
_C.SOLVER.WARMUP_EPOCHS = 10
# method of warm up, option: 'constant','linear'
_C.SOLVER.WARMUP_METHOD = "linear"


_C.SOLVER.COSINE_MARGIN = 0.1
_C.SOLVER.COSINE_SCALE = 40


# epoch number of saving checkpoints
_C.SOLVER.CHECKPOINT_PERIOD = 10
# iteration of display training log
_C.SOLVER.LOG_PERIOD = 100
# epoch number of validation
_C.SOLVER.EVAL_PERIOD = 10
_C.SOLVER.FP16 = False
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.SEED = 1234
_C.SOLVER.GRADCENTER = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Path to checkpoint and saved log of trained model
_C.OUTPUT_DIR = "./log/"
_C.TBOUTPUT_DIR = "./log/tensorboard"
_C.IF_VAL = False
