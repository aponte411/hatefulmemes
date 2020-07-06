import logging
from typing import Any, Callable
from time import time

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name: str, level: int = logging.INFO) -> Any:
    """Returns logger object with given name"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


LOGGER = get_logger(__name__)


def timing(func: Callable) -> Callable:
    """Get execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start = time()
        result = func(*args, **kwargs)
        end = time()
        LOGGER.info(f"Elapsed time: {end - start:.3f} seconds")
        return result

    return wrapper
