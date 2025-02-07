# DynamicNNBuilder

The DynamicNNbuilder is meant to easily compare different metrics for several different networks. 
By the use of factories, scaling up the project with more layer type is possible.

The user can provide a directory of folder, to compare in the generated report.

In this README there is a set up guide for how to set up a correct python environment.
Furthermore there is a guide to show the usecase of the project.

# Project Setup Guide

### Prerequisites

- Ensure you have >=Python 3.9.12 installed on your system.

### Creating and Activating the Virtual Environment

1. **Navigate to Your Project Directory**:
   
   Open a terminal and change to your project's directory.

   ```bash
   cd path/to/your/project
2. **Set up the venv**

    ```bash
    python3.9 -m venv venv
3. **Activate the environment**
    ```bash
    venv\Scripts\activate
4. **Install the requirementes**
    ```bash
    pip install -r requirements.txt
    ```


To run the default config file in configs folder use:

`python main.py`

<details>
  <summary>Arguments</summary>

The `DynamicNNBuilder` uses an `ArgumentParser` to manage command-line options. Here's a breakdown of all available options:

| Option        | Short | Type   | Default        | Description                                                                                                                                                                                                                                                                                               |
|---------------|-------|--------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--file`      | `-f`  | string | `_config_path` | Path to the main configuration file (TOML).  This can be a single `.toml` file or a directory containing multiple `.toml` files. If a directory is provided, the builder will process all `.toml` files within it.                                                                                               |
| `--save`      | `-s`  | string | `"parameters"`  | Filename (without extension) to save the trained model's parameters. The saved file will be placed in a directory (likely a "parameters" folder, but this might depend on other configurations).                                                                                                      |
| `--load`      | `-l`  | string | `None`         | Filename (without extension) to load pre-trained model parameters.  The code expects the file to be in the same location where parameters are saved. If this option is provided, the model will load these parameters instead of training from scratch.                                                  |
| `--kaggle`    | `-k`  | bool   | `False`        | If set to `True`, the network will be used to generate a CSV prediction file suitable for Kaggle competitions. This assumes the code is set up to handle the Kaggle data format and prediction generation.                                                                                             |
| `--conf_dir`  | `-cd` | string | `None`         | Path to a directory containing configuration files (TOML). This is an alternative way to provide configurations, especially useful when comparing multiple network configurations. If this is provided it will overwrite the `-f` flag.                                                                   |
| `--report`    | `-r`  | bool   | `False`        | If set to `True`, a report summarizing the training results and other relevant metrics will be generated.                                                                                                                                                                                            |
| `--report_dir`| `-rd` | string | `"report"`     | Path to the directory where the generated report will be saved.                                                                                                                                                                                                                                         |
</details>

<details>
<summary>Example Usage</summary>

* **Running with a specific TOML file:**
  ```bash
  python main.py -f path/to/my_config.toml
  ```
</summary>

**!!! UNDER DEVELOPMENT !!!**

To easily(well not yet) create and then run toml file, use:

`python check.py`

This uses the tkinter library to make the toml file you want to run.


