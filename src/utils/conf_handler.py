class ConfigError(Exception):
    def __init__(self, missing_key, expected_format=None, additional_info=None):
        self.missing_key = missing_key
        self.expected_format = expected_format
        self.additional_info = additional_info
        message = f"Configuration Error: Missing or incorrect key '{missing_key}'."
        if expected_format:
            message += f" Expected format: {expected_format}."
        if additional_info:
            message += f" Additional information: {additional_info}."
        super().__init__(message)

class ConfigHandler:

    def __init__(self, config):
        self.config = config

    def check_config(self):
        try:
            self._validate_key(self.config, "settings", expected_format="dictionary", additional_info="Should contain batch_size, iterations, and learning_rate.")
            settings = self.config["settings"]
            self._validate_key(settings, "batch_size", expected_format="integer", additional_info="Number of items to process at once.")
            self._validate_key(settings, "iterations", expected_format="integer", additional_info="Number of times to repeat the process.")
            self._validate_key(settings, "learning_rate", expected_format="float", additional_info="Rate at which learning occurs.")
        except KeyError as e:
            print(f"{e}: Error in settings from toml file. Settings must contain batch_size, iterations, and learning_rate"
                f"Provided keys: {self.config['settings'].keys()}")

    @staticmethod
    def _validate_key(dictionary, key, expected_format=None, additional_info=None):
        if key not in dictionary:
            raise ConfigError(key, expected_format, additional_info)

