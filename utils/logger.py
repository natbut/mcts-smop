import logging
import os
from datetime import datetime

import numpy as np


def init_logger(name: str, log_level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig()
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    return logger


class FileLogger:
    """Manages logging data to a file."""

    def __init__(self, filename: str = None):
        log_dir = os.path.join(os.getcwd(), "logs")

        if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

        if filename is None:
            filename = os.path.join(
                log_dir, f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.csv"
            )
        else:
            filename = os.path.join(log_dir, filename)

        self.log_filename = filename
        self.log_file = open(filename, "w")  # type: ignore

        self.log_file.write(
            "trial,best,"
            "frontEndOnly_rew,frontEndOnly_potent,frontEndOnly_percDead,"
            "distrOnly_rew,distrOnly_potent,distrOnly_percDead,"
            "twoStep_rew,twoStep_potent,twoStep_percDead,"
            "hybrid_rew,hybrid_potent,hybrid_percDead\n"
        )

    def __call__(
        self,
        trial: int,
        best_results: np.ndarray,
        frontEnd_results: np.ndarray,
        distrOnly_results: np.ndarray,
        twoPart_results: np.ndarray,
        hybrid_results: np.ndarray,
    ):
        self.log_file.write(
            f"{trial},{best_results[0]},"
            f"{frontEnd_results[0]},{frontEnd_results[1]},{frontEnd_results[2]},"
            f"{distrOnly_results[0]},{distrOnly_results[1]},{distrOnly_results[2]},"
            f"{twoPart_results[0]},{twoPart_results[1]},{twoPart_results[2]},"
            f"{hybrid_results[0]},{hybrid_results[1]},{hybrid_results[2]}\n"
        )
