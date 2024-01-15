from src.trainer import Trainer
from argparse import ArgumentParser
from pathlib import Path

_config_path = "config.toml"
_parameter_dir = "parameters"

def main():
    
    parser = ArgumentParser(description="ANN made for solving the MNIST dataset using torch.")

    parser.add_argument('-f', '--file', type=str, default=_config_path, help="File path to config file")
    parser.add_argument('-s', '--save', type=str, help='Save the parameters with the provided filename')
    parser.add_argument('-l', '--load', type=str, help='Load the parameters with the provided filename')
    args = parser.parse_args()

    par_path = Path(_parameter_dir)


    ##################################### RUNNNIG NETWORK ###############################################################
    if not args.load:
        network = Trainer(args.file) #Training using config file.
        network.show_arcitechture()
        network.train()

    else:   #TODO
        par_path_load = par_path / args.load
        #network.load(par_path_load) #TODO maybe using new object?

    ##################################### Saving #########################################################################
    if args.save:
        par_path_save = par_path / args.save
        network.save(par_path_save)
    
    ##################################### Upload weights to network ######################################################
    if args.load:
        par_path_load = par_path / args.load
        network.load(par_path_load)

    ##################################### KAGGLE MODE ####################################################################
        #TODO
        

if __name__ == "__main__":
    main()


