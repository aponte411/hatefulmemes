import os

from hydra.experimental import initialize, compose

from src.utils import utils
from src.lightning.model import HatefulMemesModel

LOGGER = utils.get_logger(__name__)


@utils.timing
def train():
    # initialize hydra config
    initialize(config_dir="conf")
    cfg = compose(config_file="config.yaml")
    hparams = {
        # Required hparams
        "train_path": cfg.data.train_path,
        "dev_path": cfg.data.valid_path,
        "img_dir": cfg.data.image_dir,
        # Optional hparams
        "embedding_dim": cfg.model.embedding_dim,
        "language_feature_dim": cfg.model.language_feature_dim,
        "vision_feature_dim": cfg.model.vision_feature_dim,
        "fusion_output_size": cfg.model.fusion_output,
        "output_path": cfg.model.output_path,
        "dev_limit": cfg.experiment.dev_limit,
        "lr": cfg.experiment.learning_rate,
        "max_epochs": cfg.experiment.max_epochs,
        "n_gpu": cfg.experiment.n_gpu,
        "batch_size": cfg.experiment.batch_size,
        # allows us to "simulate" having larger batches
        "accumulate_grad_batches": cfg.experiment.accumulate_grad_batches,
        "early_stop_patience": cfg.experiment.early_stop_patience,
    }
    LOGGER.info(f"Training HatefulMemesModel with params: {hparams}")
    hateful_memes_model = HatefulMemesModel(hparams=hparams)
    hateful_memes_model.fit()
    LOGGER.info("Training complete!")


if __name__ == "__main__":
    train()
