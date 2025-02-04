"""Module for training the network."""
import time
import torch
import toml
from torch import nn
import logging
from pathlib import Path
from .network import NeuralNetwork
from src.utils.loader import Loader
from src.utils.conf_handler import ConfigHandler

_REPORT_PATH = "report/report.txt"


def time_it(func):
    """
    Wrapper for timing methods or functions

    Args:
        func (function): Function to time. @time_it can be used.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Model executed in: {end_time - start_time:.4f} seconds".upper())
        return result
    return wrapper


COLORS = {
    "HEADER": '\033[95m', # Bright magenta
    "BLUE": '\033[94m',
    "CYAN": '\033[96m',
    "GREEN": '\033[92m',
    "YELLOW": '\033[93m',
    "RED": '\033[91m',
    "ENDC": '\033[0m'
}


class Trainer:
    """
    Object for training a neural network.
    """
    def __init__(
            self, conf_path="config.toml",
            create_net=True, parameter_path=None) -> None:

        self._config_path = conf_path
        self._config = toml.load(self._config_path)
        conf_handler = ConfigHandler(self._config)
        conf_handler.check_config()
        if create_net:
            self.net = NeuralNetwork(self._config)
            self._iterations = self._config["settings"]["iterations"]
            self._criterion = nn.CrossEntropyLoss()
            # TODO Create a criterion factory

        if not create_net:
            self.parameter_path = Path(parameter_path)
        loader = Loader()
        self._trainloader, self._testloader = loader.load_dataset(self._config)
        self.device = torch.device("cpu")
        self.best_accuracy = 0
        self.best_epoch = None

    def test(self, epoch=1, report=False, wm="w", save=False) -> None:
        """Test the network on the test set"""
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(self._testloader):
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
        epoch_text = f'Epoch [{epoch+1}/{self._iterations}],'
        validation_text = f'Validation Accuracy: {100 * accuracy:.2f}%'
        print(epoch_text + validation_text)
        if self.best_accuracy < accuracy:
            self.best_accuracy = accuracy
            self.best_epoch = epoch
            if save:
                conf_path = Path(self._config_path)
                self.save(Path(f"parameters/{conf_path.stem}.pth"))
        if report:
            # Write result to report/report.txt
            with open(_REPORT_PATH, wm) as f:
                f.write(epoch_text + validation_text)
                if epoch+1 == self._iterations:
                    f.write(f"""Best accuracy: {100 * self.best_accuracy:.2f}%
                            at epoch {self.best_epoch+1}\n""")
                    f.write("----------------------------------------\n")

    @time_it
    def train(self, show_progress=True, report=False, writemode="w", save=False) -> None:
        """Train the network"""

        if report:
            self.report_header(writemode)
            writemode = "a"

        logging.debug("Start_training")
        for epoch in range(self._iterations):
            self.net.train()
            for batch, (images, labels) in enumerate(self._trainloader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                out = self.net(images)
                loss = self._criterion(out, labels)

                # Update loss for network
                loss.backward()

                # Update network(Not s in lowrank)
                self.net.step(s=False)

                # Update S in lowrank if network contains lowrank layers.
                if self.net.contains_lowrank:
                    # Forward pass
                    out = self.net(images)

                    # Calculating loss based on criterion
                    loss = self._criterion(out, labels)

                    # Calculate gradients
                    loss.backward()

                    # Update coefficients
                    self.net.step(s=True)

                if (batch + 1) % 100 == 0:
                    print(
                        f"Epoch [{epoch+1}/{self._iterations}] ",
                        f"Step [{batch+1}/{len(self._trainloader)}] ",
                        f"Loss: {COLORS.CYAN}{loss.item():.4f}{COLORS.ENDC}")

            self.test(epoch, report=report, wm=writemode, save=save)

        print(f"""{COLORS.CYAN}Best accuracy: {100 * self.best_accuracy:.2f}%
            at epoch {COLORS.ENDC}{self.best_epoch+1}\n""")
        if report:
            self.show_best_accuracy()

        if show_progress:
            pass

    def show_best_accuracy(self) -> None:
        """Prints the best accuracy and epoch to the console"""
        best_accuracy = f'Best accuracy: {100 * self.best_accuracy:.2f}%'
        print(best_accuracy + f" at epoch {self.best_epoch+1}\n")
        with open("report/report.txt", "a") as f:  # TODO Generalise this
            f.write(f"{COLORS.CYAN}" + best_accuracy)
            f.write("at epoch {self.best_epoch+1}{COLORS.ENDC}\n")
            f.write("----------------------------------------\n")

    def load_params(self, path) -> None:
        """Loading parameters from path into self.net to be used on predictions"""
        # load_dict =  torch.load(path)

        # Parsing load_dict into the network
        network_dict = torch.load(path)
        layer_dict = {}
        for param in network_dict:  # TODO Rewrite to function.

            param_list = param.split('.')
            layer_idx = str(param_list[1].split('_')[1])
            layer_type = param_list[1].split('_')[2]
            activation = param_list[1].split('_')[3]
            attribute = param_list[2]
            # Dictionary containing list with each layer information
            if layer_idx not in layer_dict:
                layer_dict[layer_idx] = {"type":layer_type
                                        , "activation": activation,
                                        "attributes": {}}
                layer_dict[layer_idx]["attributes"][attribute] = network_dict[param]

            else:
                layer_dict[layer_idx]["attributes"][attribute] = network_dict[param]

        layer_dict = dict(sorted(layer_dict.items()))
        self.net = NeuralNetwork(layer_dict, create_net=False)

        #"Layer dict contains keys for each layer."
        #"Key contains dictionary with key for type, activatioon and attributes
        # which contains key for attribute and value"

    def load_test(self, epoch=1) -> None:
        """Testing the network on the test set"""
        total = 0
        correct = 0
        for i, (images, labels) in enumerate(self._testloader):
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Testing parameters from {self.parameter_path.stem}")
        print(f'Validation Accuracy: {100 * accuracy:.2f}%')
        print(f"Tested on {total} images")

    def save(self, path) -> None:
        """Saving the parameters of the network to a file"""
        # print(self.net.state_dict())
        # print(self.net.state_dict().keys())
        torch.save(self.net.state_dict(), path)
        print(f"Parameters saved from {path.stem} to {path}")

    def show_arcitechture(self, logo) -> None:
        # pylint: disable=W1401

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
        print(f"{COLORS.HEADER}Network Settings for: {COLORS.ENDC}{self._config_path}")
        for key, value in settings.items():
            print(f"{COLORS.BLUE}{key.capitalize()}: {COLORS.ENDC}{value}")
        max_type_length = max(len(layer['type']) for layer in self._config["layer"])
        max_input_length = max(len(str(layer['dim_in']))
                            for layer in self._config["layer"])
        max_output_length = max(len(str(layer['dim_out']))
                                for layer in self._config["layer"])
        max_activation_length = max(len(layer['activation']) if layer['activation'] 
                                    else 0 for layer in self._config["layer"])

        print(f"{COLORS.HEADER}Network Architecture:{COLORS.ENDC}")

        # Iterating over layers to print details
        for i, layer in enumerate(self._config["layer"], 1):
            layer_type = layer['type'].ljust(max_type_length)
            layer_input = str(layer['dim_in']).rjust(max_input_length)
            layer_output = str(layer['dim_out']).rjust(max_output_length)
            activation = (layer['activation'] if layer['activation'] else 'None')
            activation = activation.ljust(max_activation_length)

            layer_str = (f"{COLORS.BLUE}Layer {i}: {COLORS.ENDC}{layer_type}, "
                        f"{COLORS.GREEN}Input: {COLORS.ENDC}{layer_input}, "
                        f"{COLORS.YELLOW}Output: {COLORS.ENDC}{layer_output}, "
                        f"{COLORS.RED}Activation: {COLORS.ENDC}{activation}")
            if layer['type'] == "lowrank":
                layer_str += f"{COLORS.CYAN}Rank: {COLORS.ENDC}{layer['rank']}"
            print(layer_str)


    def report_header(self, writemode) -> None:
        """Prints the header for the report"""
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
