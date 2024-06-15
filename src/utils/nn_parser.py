from argparse import ArgumentParser


class NN_parser_factory:

    def __init__(self, _config_path):
        self.parser = ArgumentParser(
            description="ANN made for solving the MNIST dataset using torch.")
        self.parser.add_argument(
            '-f', '--file', type=str,
            default=_config_path,
            help="File path to config file")
        self.parser.add_argument(
            '-s', '--save', type=str, default="parameters",
            help='Save the parameters with the provided filename')
        self.parser.add_argument(
            '-l', '--load', type=str,
            help='Load the parameters with the provided filename')
        self.parser.add_argument(
            '-k', '--kaggle',
            type=bool, default=False, 
            help="Use network to create csv prediction for Kaggle comp")
        self.parser.add_argument(
            '-cd', '--conf_dir',
            type=str,
            help='Path to folder with dir with .toml files for configurations')
        self.parser.add_argument(
            '-r', '--report', 
            type=bool, default=False, help='Flag for creating report')
        self.parser.add_argument(
            '-rd', '--report_dir', type=str, default="report", 
            help='Path to folder where report will be saved')
        self.args = self.parser.parse_args()

    def __call__(self):
        return self.args