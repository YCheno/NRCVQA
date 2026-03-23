# --------------------------------------------------------
# Swin Transformer
# Code Link: https://github.com/microsoft/Swin-Transformer
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
# _C.DATA.BATCH_SIZE = 48
# _C.DATA.BATCH_SIZE = 12
_C.DATA.BATCH_SIZE = 8
_C.DATA.BATCH_TEST = 8

# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = ''
_C.DATA.DATASET_TEST = ''
# Input image size
_C.DATA.IMG_SIZE = 224
# Number of data loading threads
_C.DATA.NUM_WORKERS = 6

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'vb_cnn_transformer'

_C.FINE_TUNE = False
_C.FIVE = False
_C.IDX = -1

_C.VQA = CN()
_C.VQA.FRAMEWANT = 64

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LOSS = "plcc"
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-5
_C.TRAIN.WARMUP_LR =5e-7 #5e-6
_C.TRAIN.MIN_LR = 5e-7#5e-6
# Clip gradient norm
# _C.TRAIN.CLIP_GRAD = 5.0
_C.TRAIN.CLIP_GRAD = 10.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 10
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
_C.TRAIN.PERCENT = 0.8
_C.TRAIN.PRE_TRAINED = 'pre_trained/swin_small_patch4_window7_224.pth'
_C.TRAIN.USEPRETRAINED = False


_C.TRAIN.lambda1 = 1
_C.TRAIN.lambda2 = 0.5
_C.TRAIN.lambda3 = 0.5

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
_C.TEST.BESTSROCC = 0.7
# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 50
# Fixed random seed
_C.SEED = 1024
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0



def update_config(config, args):

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.base_lr:
        config.TRAIN.BASE_LR = args.base_lr
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.batch_test:
        config.DATA.BATCH_TEST = args.batch_test
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.dataset:
        config.DATA.DATASET = args.dataset
    if args.dataset_test:
        config.DATA.DATASET_TEST = args.dataset_test
    if args.model:
        config.MODEL.TYPE = args.model
    if args.frame:
        config.VQA.FRAMEWANT = args.frame
    if args.loss:
        config.TRAIN.LOSS = args.loss
    if args.best:
        config.TEST.BESTSROCC = args.best
    if args.epoch:
        config.TRAIN.EPOCHS = args.epoch
    if args.warm_up_epochs:
        config.TRAIN.WARMUP_EPOCHS = args.warm_up_epochs
    if args.pre_trained_path:
        config.TRAIN.PRE_TRAINED = args.pre_trained_path
    if args.fine_tune:
        config.FINE_TUNE = args.fine_tune
        config.TRAIN.USEPRETRAINED = True
    if args.five:
        config.FIVE = args.five
    if args.idx != -1:
        config.IDX = args.idx

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
