# DynamicNNBuilder

The DynamicNNbuilder is meant to easily compare different metrics for several different networks. 
By the use of factories, scaling up the project with more layer type is possible.

The user can provide a directory of folder, to compare in the generated report.

In this README there is a set up guide for how to set up a correct python environment.
Furthermore there is a guide to show the usecase of the project.


# Project Setup Guide

## Setting Up a Python Virtual Environment

This guide will walk you through the process of setting up a virtual environment for this project using Python 3.9.12.

### Prerequisites

- Ensure you have Python 3.9.12 installed on your system.

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