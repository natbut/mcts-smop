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
        self.log_file = open(filename, 'w')  # type: ignore

        self.log_file.write(
            "trial,test,"  # best,"
            "frontEndOnly_rew,frontEndOnly_potent,frontEndOnly_percDead,"
            "distrOnly_rew,distrOnly_potent,distrOnly_percDead,"
            "twoStep_rew,twoStep_potent,twoStep_percDead,"
            "dist_hybrid_rew,dist_hybrid_rew,dist_hybrid_rew,"
            "full_hybrid_rew,full_hybrid_rew,full_hybrid_rew\n"
        )

        self.log_file.close()

    def __call__(
        self,
        trial: int,
        test: int,
        frontEnd_results: np.ndarray,
        distrOnly_results: np.ndarray,
        twoPart_results: np.ndarray,
        dist_hybrid_results: np.ndarray,
        full_hybrid_results: np.ndarray,
    ):
        self.log_file = open(self.log_filename, 'a')  # type: ignore

        self.log_file.write(
            f"{trial[0]},{test[0]},"
            f"{frontEnd_results[0]},{frontEnd_results[1]},{frontEnd_results[2]},"
            f"{distrOnly_results[0]},{distrOnly_results[1]},{distrOnly_results[2]},"
            f"{twoPart_results[0]},{twoPart_results[1]},{twoPart_results[2]},"
            f"{dist_hybrid_results[0]},{dist_hybrid_results[1]},{dist_hybrid_results[2]},"
            f"{full_hybrid_results[0]},{full_hybrid_results[1]},{full_hybrid_results[2]}\n"
        )

        self.log_file.close()
