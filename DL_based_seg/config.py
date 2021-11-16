import numpy as np

NUM_CLASSES = 2
NUM_WORKERS = 1

RANDOM_SEED = 9314
BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
LR = 1e-3

BASE_DIR = "../dataset/dset-s2"
NORMALISE_PARAMS = [
    np.array([0.08563971, 0.114898555, 0.11390314, 0.2771133, 0.22948432, 0.1595726]).reshape((1, 1, 6)),  # MEAN
    np.array([0.07971889, 0.0962829, 0.12481037, 0.20359233, 0.20531356, 0.17086406]).reshape((1, 1, 6)),  # STD
]
RESIZE = 200
CROP_SIZE = [200, 200]

CKPT_PATH = "../saved_model/rs_checkpoint.pth"




