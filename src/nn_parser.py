from argparse import ArgumentParser
from pathlib import Path

class NN_parser_factory:

    def __init__(self, _config_path):
        self.parser = ArgumentParser(description="ANN made for solving the MNIST dataset using torch.")

        self.parser.add_argument('-f', '--file', type=str, default=_config_path, help="File path to config file")
        self.parser.add_argument('-s', '--save', type=str, help='Save the parameters with the provided filename')
        self.parser.add_argument('-l', '--load', type=str, help='Load the parameters with the provided filename')
        self.parser.add_argument('-k', '--kaggle', type=bool, default=False, help="Use network to create csv prediction for Kaggle comp")
        self.args = self.parser.parse_args()


    def __call__(self):
        return self.args