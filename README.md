# Project Setup

#### 1. Installing Python 3.11.9

#### 2. Setting up a Virtual Environment (optional)

#### 3. Installing Required Modules

#### 4. Installing the Dataset

#### 5. Training on a Different Dataset

## 1. Installing Python 3.11.9

### Windows

Download Python 3.11.9: Visit the official Python website (https://www.python.org/downloads/) and download the latest version of Python 3.11.9 for Windows.

Run the downloaded installer and follow the prompts. Make sure to check the box that says "Add Python 3.11 to PATH" during the installation.

You can check that the installation was successful by opening the command prompt (cmd) as an administrator and running the following command:

```bash
python --version
```

Pip is the default package manager, and is usually installed by default with Python. However, you can check if its installed by running the following command in the command prompt as an administrator:

```bash
python -m pip --version
```

If pip is installed, it will display the version. If not, follow the instructions on this page: https://pip.pypa.io/en/stable/installation/

### Linux

Open a terminal and run the following commands to update your package lists:

```bash
sudo apt-get update
sudo apt-get upgrade
```

Run the following command to install Python 3.11:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install python3.11
```

Pip is the default package manager is usually installed by default with Python. However, you can check if it's installed by running the following command in your terminal:

```bash
python -m pip --version
```

If pip is installed, it will display the version. If not, follow the instructions on this page: https://pip.pypa.io/en/stable/installation/

## 2. Setting up a Virtual Environment in the Workspace Directory (Optional)

A virtual environment is a tool that helps to keep dependencies required by different projects separate by creating isolated python environments for them. It is good practice to install modules in a virtual environment, as it allows for instances of different versions of the modules, ensuring the projects do not break when modules are updated.

### Windows

Use the cd command in the command prompt to navigate to your workspace directory:

```bash
cd\<working-directory-file-path>
```

Run the following command to create a virtual environment:

```bash
python -m venv myenv
```

This will create a folder in your current directory which will contain the Python executable files, and a copy of the pip library which you can use to install other packages. The name of the virtual environment in this example is `myenv`.

Run the following command to activate the virtual environment:

```bash
myenv\Scripts\activate.bat
```

You will see the name of the current virtual environment in your command prompt when it's activated.

To deactivate the virtual environment when you are finished, run the following command in the command prompt where the virtual environment is activated:

```bash
deactivate
```

The name of the current virtual environment will disappear from your terminal prompt when it's deactivated.

### Linux

Use the cd command in the terminal to navigate to your workspace directory:

```bash
cd /path/to/your/workspace
```

Run the following command to create a virtual environment:

```bash
python3 -m venv myenv
```

This will create a folder in your current directory which will contain the Python executable files, and a copy of the pip library which you can use to install other packages. The name of the virtual environment in this example is `myenv`.

Run the following command to activate the virtual environment:

```bash
source myenv/bin/activate
```

You will see the name of the current virtual environment in your terminal prompt when it's activated.

To deactivate the virtual environment when you are finished, run the following command in the terminal where the virtual environment is activated:

```bash
deactivate
```

The name of the current virtual environment will disappear from your terminal prompt when it's deactivated.

## 3. Installing Modules

### Windows

When navigated to the workspace directory (and activated the virtual environment), run the following command:

```bash
pip install -r requirements.txt
```

### Linux

When navigated to the workspace directory (and activated the virtual environment), run the following command:

```bash
pip3 install -r requirements.txt
```

## 4. Installing the Dataset

The dataset used for the project was provided from Xiao, H., Wu, J.-L., Laizet, S., and Duan, L., "Flows Over Periodic Hills of Parameterized Geometries: A Dataset for Data-Driven Turbulence Modeling from Direct Simulations," Computers & Fluids, Vol. 200, 104431, 2020, https://doi.org/10.1016/j.compfluid.2020.104431.

The dataset can be found here: https://turbmodels.larc.nasa.gov/Other_DNS_Data/parameterized_periodic_hills.html

The downloaded zip folder should be extracted into the working directory, so the relative path to the files should be for example
```bash
DNS_29_Periodic_Hills\alph05-4071-2024.dat
```

## 5. Training on a Different Dataset
The logic transforming the data can be found in the data processing folder. The data is fed into the neural network as a zipped numpy `.npz` format.