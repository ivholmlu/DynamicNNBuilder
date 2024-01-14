from src.trainer import Trainer
from argparse import ArgumentParser
from pathlib import Path

_config_path = "config.toml"
_parameter_dir = "parameters"

def main():
    network = Trainer(_config_path)
    network.show_arcitechture()
    network.train()
    parser = ArgumentParser(description="TODO")

    parser.add_argument('-s', '--save', type=str, help='Save the parameters with the provided filename')
    args = parser.parse_args()

    if args.save:
        path = Path(_parameter_dir) / args.save
        network.save(path)
        

if __name__ == "__main__":
    main()




