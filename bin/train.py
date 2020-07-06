import os

from src.utils import utils
from src.lightning.model import HatefulMemesModel

LOGGER = utils.get_logger(__name__)

TRAIN_PATH = os.environ.get("TRAIN_PATH")
DEV_PATH = os.environ.get("DEV_PATH")
DATA_DIR = os.environ.get("DATA_DIR")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH")

hparams = {
    # Required hparams
    "train_path": TRAIN_PATH,
    "dev_path": DEV_PATH,
    "img_dir": DATA_DIR,
    # Optional hparams
    "embedding_dim": 150,
    "language_feature_dim": 300,
    "vision_feature_dim": 300,
    "fusion_output_size": 256,
    "output_path": OUTPUT_PATH,
    "dev_limit": None,
    "lr": 0.00005,
    "max_epochs": 10,
    "n_gpu": 1,
    "batch_size": 4,
    # allows us to "simulate" having larger batches
    "accumulate_grad_batches": 16,
    "early_stop_patience": 3,
}


@utils.timing
def main():
    LOGGER.info(f"Starting training with params: {hparams}")
    hateful_memes_model = HatefulMemesModel(hparams=hparams)
    hateful_memes_model.fit()
    LOGGER.info("Training complete!")


if __name__ == "__main__":
    main()
