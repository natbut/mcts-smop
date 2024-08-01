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

    def __init__(self, filename: str | None = None):
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
            "trial,best,frontEndOnly,distrOnly,twoStep,hybrid,position6\n"
            # "velocity1,velocity2,velocity3,velocity4,velocity5,velocity6,control_input1,"
            # "control_input2,control_input3,control_input4,control_input5,control_input6,"
            # "end_effector_x,end_effector_y,end_effector_z\n"
        )

    def __call__(
        self,
        trials_arr: np.ndarray,
        best_results: np.ndarray,
        frontEnd_results: np.ndarray,
        distrOnly_results: np.ndarray,
        twoPart_results: np.ndarray,
        hybrid_results: np.ndarray,
    ):
        self.log_file.write(
            f"{timestamp},{thetas[0]},{thetas[1]},{thetas[2]},{thetas[3]},{thetas[4]},"
            f"{thetas[5]},{theta_dots[0]},{theta_dots[1]},{theta_dots[2]},"
            f"{theta_dots[3]},{theta_dots[4]},{theta_dots[5]},{taus[0]},{taus[1]},"
            f"{taus[2]},{taus[3]},{taus[4]},{taus[5]},{ee_pos[0]},{ee_pos[1]},"
            f"{ee_pos[2]}\n"
        )
