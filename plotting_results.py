
from argparse import ArgumentParser, Namespace

from utils.plotter import plot_results_from_log


def get_args() -> Namespace:
    """Parse the script arguments.

    Returns:
        The parsed argument namespace.
    """
    parser = ArgumentParser()

    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the data file",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    filepath = args.data_file
    plot_results_from_log(filepath)
