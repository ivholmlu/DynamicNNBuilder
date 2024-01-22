import time
import torch
import toml
import torch.nn as nn
import logging
from pathlib import Path

from .network import NeuralNetwork
from ..utils.loader import Loader
from ..utils.conf_handler import ConfigHandler




def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Model executed in: {end_time - start_time:.4f} seconds".upper())
        return result
    return wrapper
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'  # Resets the color

class Trainer:
    """Object for training a neural network.
    """
    def __init__(self, conf_path="config.toml", create_net =True) -> None:

        self._config_path = conf_path
        self._config = toml.load(self._config_path)
        conf_handler = ConfigHandler(self._config)
        conf_handler.check_config()
        if create_net:
            self.net = NeuralNetwork(self._config)
            self._iterations = self._config["settings"]["iterations"]
            self._criterion = nn.CrossEntropyLoss() #TODO Create a criterion factory
        
        loader = Loader()
        self._trainloader, self._testloader = loader.load_dataset(self._config)
        self.device = torch.device("cpu")
        self.best_accuracy = 0
        self.best_epoch = None

    def test(self, epoch=1, report=False, wm="w", save=False):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(self._testloader):
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Epoch [{epoch+1}/{self._iterations}], Validation Accuracy: {100 * accuracy:.2f}%')
        if self.best_accuracy < accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            if save:
                conf_path = Path(self._config_path)
                self.save(Path(f"parameters/{conf_path.stem}.pth"))
        if report:
            #Write result to report/report.txt
            with open("report/report.txt", wm) as f:
                f.write(f'Epoch [{epoch+1}/{self._iterations}], Validation Accuracy: {100 * accuracy:.2f}%\n')
                if epoch+1 == self._iterations:
                    f.write(f"Best accuracy: {100 * self.best_accuracy:.2f}% at epoch {self.best_epoch+1}\n")
                    f.write("----------------------------------------\n")

    @time_it
    def train(self, show_progress=True, report=False, writemode="w", save=False):

        if report:
            self.report_header(writemode)
            writemode = "a"
            
        logging.debug("Start_training")
        for epoch in range(self._iterations):
            self.net.train()
            for batch, (images, labels) in enumerate(self._trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                #Forward pass
                out = self.net(images)
                loss = self._criterion(out, labels)

                ### Update loss for network
                loss.backward()

                #Update network(Not s in lowrank)
                self.net.step(s=False)
                
                #Update S in lowrank if network contains lowrank layers.
                if self.net._contains_lowrank:
                    #Forward pass
                    out = self.net(images)

                    #Calculating loss based on criterion
                    loss = self._criterion(out, labels)
                    
                    #Calculate gradients
                    loss.backward()

                    #Update coefficients
                    self.net.step(s=True)
                
                if (batch + 1) % 100 == 0:
                    print(f"Epoch [{epoch+1}/{self._iterations}] "
                            f"Step [{batch+1}/{len(self._trainloader)}] "
                            f"Loss: {Colors.CYAN}{loss.item():.4f}{Colors.ENDC}")
                
            self.test(epoch, report=report, wm=writemode, save=save)
        
        print(f"Best accuracy: {100 * self.best_accuracy:.2f}% at epoch {self.best_epoch+1}\n")
        if report:
            self.show_best_accuracy()

            
    def show_best_accuracy(self):
        print(f"Best accuracy: {100 * self.best_accuracy:.2f}% at epoch {self.best_epoch+1}\n")
        with open("report/report.txt", "a") as f: #TODO Generalise this
            f.write(f"{Colors.CYAN}Best accuracy: {100 * self.best_accuracy:.2f}%")
            f.write("at epoch {self.best_epoch+1}{Colors.ENDC}\n")
            f.write("----------------------------------------\n")

    def load_params(self, path):
        """Loading parameters from path into self.net to be used on predictions"""
        #load_dict =  torch.load(path)

        #Parsing load_dict into the network
        network_dict = torch.load(path)
        layer_dict = {}
        for param in network_dict: #TODO Rewrite to function.
            
            param_list = param.split('.')
            layer_idx = str(param_list[1].split('_')[1])
            layer_type = param_list[1].split('_')[2]
            
            activation = param_list[1].split('_')[3]
            
            attribute = param_list[2]
            
            
            #Dictionary containing list with each layer information
            if layer_idx not in layer_dict:
                layer_dict[layer_idx] = {"type":layer_type, "activation": activation, "attributes": {}}
                layer_dict[layer_idx]["attributes"][attribute] = network_dict[param]

            else:
                
                layer_dict[layer_idx]["attributes"][attribute] = network_dict[param]
            
        
        layer_dict = dict(sorted(layer_dict.items()))
        self.net = NeuralNetwork(layer_dict, create_net=False)

        "Layer dict contains keys for each layer." 
        "Key contains dictionary with key for type, activatioon and attributes which contains key for attribute and value"

    def load_test(self, epoch=1):
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(self._testloader):
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        print(f'Validation Accuracy: {100 * accuracy:.2f}%')

    def save(self, path):
        #print(self.net.state_dict())
        #print(self.net.state_dict().keys())
        torch.save(self.net.state_dict(), path)
        print(f"Parameters saved from {path.stem} to {path}")

    def show_arcitechture(self, logo):

        art = """
        ______                             _      _   _ _   _______       _ _     _           
        |  _  \                           (_)    | \ | | \ | | ___ \     (_| |   | |          
        | | | |_   _ _ __   __ _ _ __ ___  _  ___|  \| |  \| | |_/ /_   _ _| | __| | ___ _ __ 
        | | | | | | | '_ \ / _` | '_ ` _ \| |/ __| . ` | . ` | ___ | | | | | |/ _` |/ _ | '__|
        | |/ /| |_| | | | | (_| | | | | | | | (__| |\  | |\  | |_/ | |_| | | | (_| |  __| |   
        |___/  \__, |_| |_|\__,_|_| |_| |_|_|\___\_| \_\_| \_\____/ \__,_|_|_|\__,_|\___|_|   
                __/ |                                                                         
                |___/                                                                          """
        if logo:
            print(art)
        # Calculating maximum lengths for alignment

        settings = self._config["settings"]
        print(f"{Colors.HEADER}Network Settings for: {Colors.ENDC}{self._config_path}")
        for key, value in settings.items():
            print(f"{Colors.BLUE}{key.capitalize()}: {Colors.ENDC}{value}")
        max_type_length = max(len(layer['type']) for layer in self._config["layer"])
        max_input_length = max(len(str(layer['dim_in'])) 
                            for layer in self._config["layer"])
        max_output_length = max(len(str(layer['dim_out'])) 
                                for layer in self._config["layer"])
        max_activation_length = max(len(layer['activation']) if layer['activation'] 
                                    else 0 for layer in self._config["layer"])

        print(f"{Colors.HEADER}Network Architecture:{Colors.ENDC}")

        # Iterating over layers to print details
        for i, layer in enumerate(self._config["layer"], 1):
            layer_type = layer['type'].ljust(max_type_length)
            layer_input = str(layer['dim_in']).rjust(max_input_length)
            layer_output = str(layer['dim_out']).rjust(max_output_length)
            activation = (layer['activation'] if layer['activation'] else 'None')
            activation = activation.ljust(max_activation_length)

            layer_str = (f"{Colors.BLUE}Layer {i}: {Colors.ENDC}{layer_type}, "
                        f"{Colors.GREEN}Input: {Colors.ENDC}{layer_input}, "
                        f"{Colors.YELLOW}Output: {Colors.ENDC}{layer_output}, "
                        f"{Colors.RED}Activation: {Colors.ENDC}{activation}")
            print(layer_str)
    

    def report_header(self, writemode):
        #Check if report folder exists and create if not.
            Path("report").mkdir(parents=True, exist_ok=True)

            settings = self._config["settings"]
            header = f"Network Settings for: {self._config_path}"
            for key, value in settings.items():
                print(f"{key.capitalize()} : {value}")
            max_type_length = max(len(layer['type']) for layer in self._config["layer"])
            max_input_length = max(len(str(layer['dim_in'])) 
                                for layer in self._config["layer"])
            max_output_length = max(len(str(layer['dim_out'])) 
                                    for layer in self._config["layer"])
            max_activation_length = max(len(layer['activation']) if layer['activation'] 
                                        else 0 for layer in self._config["layer"])
            # Iterating over layers to print details
            with open("report/report.txt", writemode) as f:
                if not writemode == "w":
                    f.write("\n\n")
                f.write(header+"\n")
                f.write(f"Report for {self._config_path}\n")
                for i, layer in enumerate(self._config["layer"], 1):
                    layer_type = layer['type'].ljust(max_type_length)
                    layer_input = str(layer['dim_in']).rjust(max_input_length)
                    layer_output = str(layer['dim_out']).rjust(max_output_length)
                    activation = (layer['activation'] if layer['activation'] else 'None')
                    activation = activation.ljust(max_activation_length)
                    layer_str = (f" Input: {layer_input} -  "
                                f"Output: {layer_output}\n"
                                f" Activation: {activation}"
                                f"Layer {i}: {layer_type}\n")
                    f.write(layer_str)
                f.write("----------------------------------------\n")



