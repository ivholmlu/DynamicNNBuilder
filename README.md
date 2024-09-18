# DynamicNNBuilder

The DynamicNNbuilder is meant to easily compare different metrics for several different networks. 
By the use of factories, scaling up the project with more layer type is possible.

The user can provide a directory of folder, to compare in the generated report.

In this README there is a set up guide for how to set up a correct python environment.
Furthermore there is a guide to show the usecase of the project.

# Project Setup Guide

## Setting Up a Python Virtual Environment

This guide will walk you through the process of setting up a virtual environment for this project using Python >= 3.9.12. During developement Python 3.9.12 was used.

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


### How to use the DynamicNNBuilder

To run the default config file in main folder use:

`python main.py`

To define a specific toml or folder with toml files use:

`python main.py -f path_to_file_or_folder.toml`

To save at a specific location, use:

`python main.py -l path_to_save_location`

You can specify where the report should be saved with:

`python main.py -rd report_save_path`

**!!! UNDER DEVELOPMENT !!!**

To easily(well not yet) create and then run toml file, use:

`python check.py`

This uses the tkinter library to make the toml file you want to run.


