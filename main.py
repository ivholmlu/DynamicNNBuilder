from src.trainer import Trainer
from src.nn_parser import NN_parser_factory
from argparse import ArgumentParser
from pathlib import Path

_config_path = "config.toml"
#_config_path = "config_examples/config_4.toml"
#_config_path = "config_examples/vanillalowrank.toml"
_parameter_dir = "parameters"

def main():
    
    args = NN_parser_factory(_config_path)()

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
        network = Trainer(create_net=False)
        par_path_load = par_path / args.load
        network.load_params(par_path_load)
        network.load_test()
        

        

    ##################################### KAGGLE MODE ####################################################################
        #TODO
        

if __name__ == "__main__":
    main()


