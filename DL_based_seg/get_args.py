import argparse
from config import *

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="NAS Search")
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=RESIZE,
        help="Resize side transformation.",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default=BASE_DIR,
        help="the root of database",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        nargs="+",
        default=CROP_SIZE,
        help="Crop size for training,",
    )
    parser.add_argument(
        "--normalise-params",
        type=list,
        default=NORMALISE_PARAMS,
        help="Normalisation parameters [scale, mean, std],",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=BATCH_SIZE,
        help="Batch size to train the segmenter model.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of workers for pytorch's dataloader.",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="+",
        default=NUM_CLASSES,
        help="Number of output classes for each task.",
    )

    parser.add_argument(
        "--val-batch-size",
        type=int,
        default=VAL_BATCH_SIZE,
        help="Batch size to validate the segmenter model.",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed to provide (near-)reproducibility.",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default=CKPT_PATH, help="Path to the checkpoint file."
    )
    # Optimisers
    parser.add_argument(
        "--lr",
        type=float,
        nargs="+",
        default=LR,
        help="Learning rate for encoder.",
    )


    return parser.parse_args()