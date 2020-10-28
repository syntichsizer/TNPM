# TNPM

## About 

Project for the Advanced programming in Python course @ TVZ Zagreb.

## Setup

First, we need to create virtual environment for our python project with [`venv`](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) module to avoid installing Python packages globally which could break systems tools or other projects.

You can install virtual environment using pip with following command in your project folder.

On macOS and Linux:

`$ python3.8 -m venv env`

On Windows:

`$ py -3.8 -m venv env`


### Activating a virtual environment

Before you can start installing or using packages in venv you will need to activate it.

On macOS and Linux: 

`$ source env/bin/activate`

On Windows:

`$ .\env\Scripts\activate`


### Installing required packages

Now that you are in virtual environment you can install required packages for this project. 

`(env) $ pip install -r modules.txt`