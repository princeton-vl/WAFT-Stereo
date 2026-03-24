from .config import CfgNode as CN

# NOTE: given the new config system
# we will stop adding new functionalities to default CfgNode

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# ------------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# ------------------------------------------------------------------------------

_CN = CN()

# The version number, to upgrade from old config to new ones if any
# changes happen. It's recommended to keep a VERSION in your config file.
_CN.VERSION = 2

# ---------------------------------------------------------------------------- #
# Model
# ---------------------------------------------------------------------------- #
_CN.ALGORITHM = "waft"

_CN.WAFT = CN()
_CN.WAFT.FEATURE_ENCODER = CN()
_CN.WAFT.FEATURE_ENCODER.TYPE = "dinov3"
_CN.WAFT.FEATURE_ENCODER.ARCH = "vits"
_CN.WAFT.FEATURE_ENCODER.LORA_RANK = None
_CN.WAFT.FEATURE_ENCODER.LORA_ALPHA = None
_CN.WAFT.ITERATIVE_MODULE = CN()
_CN.WAFT.MAX_DISP = 320
_CN.WAFT.LOSS = None

_CN.WAFT.ITERATIVE_MODULE.TASK = []

_CN.WAFT.ITERATIVE_MODULE.PROP_ITER = CN()
_CN.WAFT.ITERATIVE_MODULE.PROP_ITER.TYPE = None
_CN.WAFT.ITERATIVE_MODULE.PROP_ITER.ARCH = None
_CN.WAFT.ITERATIVE_MODULE.PROP_ITER.LORA_RANK = None
_CN.WAFT.ITERATIVE_MODULE.PROP_ITER.LORA_ALPHA = None
_CN.WAFT.ITERATIVE_MODULE.PROP_ITER.PATCH_SIZE = None

_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER = CN()
_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER.TYPE = None
_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER.ARCH = None
_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER.LORA_RANK = None
_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER.LORA_ALPHA = None
_CN.WAFT.ITERATIVE_MODULE.DELTA_ITER.PATCH_SIZE = None

# ---------------------------------------------------------------------------- #
# Dataset and data augmentation
# ---------------------------------------------------------------------------- #
_CN.DATASETS = CN()
# List of dataset name for training
_CN.DATASETS.TRAIN = ["sceneflow"]
_CN.DATASETS.MUL = [-1]
# List of dataset name for testing
_CN.DATASETS.TEST = ["things"]
# Image gamma
_CN.DATASETS.IMG_GAMMA = None
# Color saturation
_CN.DATASETS.SATURATION_RANGE = [0, 1.4]
# Flip the images horizontally or vertically, valid choice [False, 'h', 'v']
_CN.DATASETS.DO_FLIP = False
# Re-scale the image randomly
_CN.DATASETS.SPATIAL_SCALE = [-0.2, 0.4]
# Simulate imperfect rectification
_CN.DATASETS.YJITTER = False
# Image size for training
_CN.DATASETS.CROP_SIZE = [368, 784]
_CN.DATASETS.DIVIS_BY = 16

# ---------------------------------------------------------------------------- #
# Dataset and data augmentation
# ---------------------------------------------------------------------------- #
_CN.DATALOADER = CN()
# Number of data loading threads
_CN.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_CN.SOLVER = CN()

_CN.SOLVER.MIX_PRECISION = False
_CN.SOLVER.MAX_ITER = 300000

_CN.SOLVER.BASE_LR = 0.0005

# Save a checkpoint after every this number of iterations
_CN.SOLVER.CHECKPOINT_PERIOD = 100000
_CN.SOLVER.LATEST_CHECKPOINT_PERIOD = 1000

# Number of images per batch across all machines. This is also the number
# of training images per step (i.e. per iteration). If we use 16 GPUs
# and IMS_PER_BATCH = 32, each GPU will see 2 images per batch.
_CN.SOLVER.IMS_PER_BATCH = 8

# Gradient clipping
_CN.SOLVER.GRAD_CLIP = 1.0

# resume from pretrain model for finetuning or resuming from terminated training
_CN.SOLVER.RESUME = None
_CN.SOLVER.NO_RESUME_OPTIMIZER = False

# Maximum disparity for training,
# ground truth disparities exceed than this threshold will be ignored for loss computation
_CN.SOLVER.MAX_DISP = 192

# Loss type used in cost aggregation and refinement
_CN.SOLVER.LOSS_TYPE = "L1"

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_CN.TEST = CN()
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_CN.TEST.EVAL_PERIOD = 20000
# Threshold for metric computation for testing
_CN.TEST.EVAL_THRESH = [['1.0', '3.0']]
# Maximum disparity for metric computation mask
_CN.TEST.EVAL_MAX_DISP = [1000]
# Whether use only valid pixels in evaluation
_CN.TEST.EVAL_ONLY_VALID = [True]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same of similar sizes, benchmark is often helpful.
_CN.CUDNN_BENCHMARK = True


def export_model_config(cfg: CN) -> CN:
    model_config = CN()
    fields = ["ALGORITHM"]
    for fld in fields:
        setattr(model_config, fld, getattr(cfg, fld))
    return model_config.clone()
