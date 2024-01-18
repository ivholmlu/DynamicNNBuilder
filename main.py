from src.network.trainer import Trainer
from src.utils.nn_parser import NN_parser_factory
from pathlib import Path
import glob

import logging
logging.basicConfig(filename="logs.log", level=10)
# Default, can be defined by user
_config_path = "config.toml"
_parameter_dir = "parameters"


def main():

    args = NN_parser_factory(_config_path)()

    if args.load:  # Run if
        network = Trainer(create_net=False)
        par_path_load = _parameter_dir / args.load
        network.load_params(par_path_load)
        network.load_test()

    else:
        # Creating list of toml files
        if Path(args.file).is_dir():
            # Path is a directory, find all .toml files
            config_files = glob.glob(f"{args.file}/*.toml")
        else:
            # Path is a specific file
            config_files = [args.file]

        # Training on the provided toml files
        for file in config_files:
            file = Path(file)

            network = Trainer(str(file))  # Training using config file.
            network.show_arcitechture()
            network.train()
            if args.save:
                parameter_dir = Path(_parameter_dir)
                par_path_save = parameter_dir / (file.stem + ".pth")
                network.save(par_path_save)


if __name__ == "__main__":
    main()
