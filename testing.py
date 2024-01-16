import toml

config = toml.load("config.toml")

print(config)

print(config["settings"]["learning_rate"])

print(config["layer"])

for layer in config["layer"]:
    print(layer)