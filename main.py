from src.trainer import Trainer

_config_path = "config.toml"
network = Trainer(_config_path)
network.train()
