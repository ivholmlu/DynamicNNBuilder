"""Parser for the neural network."""
from argparse import ArgumentParser


class NnParserFactory:
    """Factory for creating the parser for the neural network"""

    def __init__(self, _config_path):
        self._parser = ArgumentParser(
            description="ANN made for solving the MNIST dataset using torch.")
        self._parser.add_argument(
            '-f', '--file', type=str,
            default=_config_path,
            help="File path to config file")
        self._parser.add_argument(
            '-s', '--save', type=str, default="parameters",
            help='Save the parameters with the provided filename')
        self._parser.add_argument(
            '-l', '--load', type=str,
            help='Load the parameters with the provided filename')
        self._parser.add_argument(
            '-k', '--kaggle',
            type=bool, default=False,
            help="Use network to create csv prediction for Kaggle comp")
        self._parser.add_argument(
            '-cd', '--conf_dir',
            type=str,
            help='Path to folder with dir with .toml files for configurations')
        self._parser.add_argument(
            '-r', '--report',
            type=bool, default=False, help='Flag for creating report')
        self._parser.add_argument(
            '-rd', '--report_dir', type=str, default="report",
            help='Path to folder where report will be saved')
        self.args = self._parser.parse_args()

    def __call__(self):
        return self.args

    @property
    def parser(self):
        """Return the parser"""
        return self._parser
