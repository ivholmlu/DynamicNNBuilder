from src.network.trainer import Trainer
from src.utils.nn_parser import NN_parser_factory
from argparse import ArgumentParser
from pathlib import Path
import glob


_config_path = "config.toml"
#_config_path = "config_examples/config_4.toml"
#_config_path = "config_examples/vanillalowrank.toml"
_parameter_dir = "parameters"

def main():
    
    args = NN_parser_factory(_config_path)()

    #s_dir = 
    if Path(_config_path).is_dir():
        # Path is a directory, find all .toml files
        config_files = glob.glob(f"{args.file}/*.toml")
        for file in config_files:
            file = Path(file)
            if not args.load:
                network = Trainer(str(file)) #Training using config file.
                network.show_arcitechture()
                network.train()

            else: #Pretrained weights
                network = Trainer(create_net=False)
                par_path_load = _parameter_dir / args.load
                network.load_params(par_path_load)
                network.load_test()

            ##################################### Saving #########################################################################
            if args.save:
                par_path_save = _parameter_dir / file.stem +"pth"
                network.save(par_path_save)
    else:
        # Path is a specific file
        config_files = [args.file]  
    ##################################### RUNNNIG NETWORK ###############################################################

    #if args.cd:
    #    network = Trainer()
    for file in config_files:

        if not args.load:
            network = Trainer(file) #Training using config file.
            network.show_arcitechture()
            network.train()

        else:   #TODO
            network = Trainer(create_net=False)
            par_path_load = _parameter_dir / args.load
            network.load_params(par_path_load)
            network.load_test()
            #network.load(par_path_load) #TODO maybe using new object?

        ##################################### Saving #########################################################################
        if args.save:
            par_path_save = _parameter_dir / args.save
            network.save(par_path_save)
        
    ##################################### Upload weights to network ######################################################

        

    ##################################### KAGGLE MODE ####################################################################
        #TODO
        

if __name__ == "__main__":
    main()


