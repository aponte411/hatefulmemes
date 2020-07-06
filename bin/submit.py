import os

from src.utils import utils
from src.lightning.model import HatefulMemesModel

LOGGER = utils.get_logger(__name__)

CHECKPOINT = os.environ.get("CHECKPOINT")
TEST_PATH = os.environ.get("TEST_PATH")


def main():
    hateful_memes_model = HatefulMemesModel.load_from_checkpoint(CHECKPOINT)
    submission = hateful_memes_model.make_submission_frame(TEST_PATH)
    LOGGER.info(submission.head())


if __name__ == "__main__":
    main()
