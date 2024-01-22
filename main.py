from src.network.trainer import Trainer
from src.utils.nn_parser import NN_parser_factory
from pathlib import Path
import glob
import psutil


import logging
logging.basicConfig(filename="logs.log", level=10)
# Default, can be defined by user
_config_path = "config.toml"
_parameter_dir = "parameters"


def print_memory_usage():
    memory = psutil.virtual_memory()
    total_memory = memory.total / (1024 * 1024)  # Convert to MB
    used_memory = memory.used / (1024 * 1024)  # Convert to MB
    memory_percentage = memory.percent

    print(f"Total Memory: {total_memory:.2f} MB")
    print(f"Used Memory: {used_memory:.2f} MB")
    print(f"Memory Usage: {memory_percentage}%")

def main():

    parameter_dir = Path(_parameter_dir) #Used for saving parameters

    args = NN_parser_factory(_config_path)()

    if args.load:  # Run if weights should be loaded
        network = Trainer(create_net=False)
        par_path_load = _parameter_dir / args.load
        network.load_params(par_path_load)
        network.load_test()  #Add args.r here as TRUE/FALSE #TODO

    else:
        # Creating list of toml files, either from a directory or a single file
        if Path(args.file).is_dir():
            config_files = glob.glob(f"{args.file}/*.toml")
        else:
            config_files = [args.file]

        logo = True 
        wm = "w" #Setting write mode to overwrite
        for file in config_files:
            file = Path(file)
            network = Trainer(str(file))  # Training using config file.
            print_memory_usage()
            network.show_arcitechture(logo)
            network.train(report = args.report, writemode=wm, save = args.save)
            wm = "a" 
            if args.save: #NEEDS TO BE IN TRAIN
                par_path_save = parameter_dir / (file.stem + ".pth")
                network.save(par_path_save)
            logo = False
        


if __name__ == "__main__":
    main()
