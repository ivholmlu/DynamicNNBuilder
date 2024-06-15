from src.network.trainer import Trainer
from src.utils.nn_parser import NN_parser_factory
from pathlib import Path
import glob
import logging

_config_path = "configs/config.toml"
_parameter_dir = "parameters"

logging.basicConfig(filename="report/logs.log", level=40)


def main() -> None:

    # Log on highest level that program starting
    logging.log(40, "Program started")

    parameter_dir = Path(_parameter_dir)  # Used for saving parameters

    args = NN_parser_factory(_config_path)()

    if args.load:  # Run if weights should be loaded
        network = Trainer(create_net=False, parameter_path=args.load)
        par_path_load = parameter_dir / args.load
        network.load_params(par_path_load)
        network.load_test()  # Add args.r here as TRUE/FALSE #TODO

    else:
        # Creating list of toml files, either from a directory or a single file
        if Path(args.file).is_dir():
            config_files = glob.glob(f"{args.file}/*.toml")
        else:
            config_files = [args.file]

        logo = True 
        wm = "w"  # Setting write mode to overwrite
        for file in config_files:
            file = Path(file)
            network = Trainer(str(file))  # Training using config file.
            network.show_arcitechture(logo)
            network.train(report=args.report, writemode=wm, save=args.save)
            wm = "a" 
            logo = False

    logging.log(40, "Program ended")


if __name__ == "__main__":
    main()
