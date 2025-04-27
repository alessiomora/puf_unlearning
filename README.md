# Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients
This is the official repository for Federated Unlearning Made Practical: Seamless Integration via Negated Pseudo-Gradients

## Preliminaries
The simulation code in this repository mainly leverages TensorFlow (TF). 
Python virtual env is managed via Poetry.
See `puf_unlearning/pyproject.toml`. To reproduce our virtual env,
follow the instructions in the Environment Setup section of this readme.

The code in this repository has been tested on Ubuntu 22.04.3,
and with Python version `3.10.13`.

## Environment Setup
By default, Poetry will use the Python version in your system. 
In some settings, you might want to specify a particular version of Python 
to use inside your Poetry environment. You can do so with `pyenv`. 
Check the documentation for the different ways of installing `pyenv`,
but one easy way is using the automatic installer:

```bash
curl https://pyenv.run | bash
```
You can then install any Python version with `pyenv install <python-version>`
(e.g. `pyenv install 3.9.17`) and set that version as the one to be used. 
```bash
# cd to your puf_unlearning directory (i.e. where the `pyproject.toml` is)
pyenv install 3.10.12

pyenv local 3.10.12

# set that version for poetry
poetry env use 3.10.12
```
To build the Python environment as specified in the `pyproject.toml`, use the following commands:
```bash
# cd to your puf_unlearning directory (i.e. where the `pyproject.toml` is)

# install the base Poetry environment
poetry install

# activate the environment
poetry shell

# intall tensorflow + cuda gpu support (retry again if error appears)
pip install tensorflow[and-cuda]=="2.15.0.post1"

# installed for transformer experiments via pip
pip install ml_collections=="0.1.1"
pip install tensorflow-hub=="0.14.0"
pip install torchvision=="0.13.0"
pip install transformers=="4.34.0"
# for charts
pip install seaborn=="0.13.0"
```

## Creating Client Datasets
To exactly reproduce the label distribution we used in the paper run the following lines of code.
Note that we use the txt files in `client_data` folder.

```bash
python -m puf_unlearning.dataset_preparation dataset="cifar100" alpha=0.1 total_clients=10

python -m puf_unlearning.dataset_preparation dataset="cifar10" alpha=0.3 total_clients=10

# alpha=-1 creates IID partitions
python -m puf_unlearning.dataset_preparation dataset="cifar100" alpha=-1 total_clients=10
```

## Running the Simulations
We prepared two scripts to run the experiments reported in the paper.
Note that this scripts save model checkpoints on disk (it can occupy 20-30 GB in total).
```bash
bash ./puf_unlearning/simulation_manager_cifar10.sh

bash ./puf_unlearning/simulation_manager_cifar100.sh
```

## Generating CSV Results
We prepared two scripts to generate csv files that report individual and aggregated results.
```bash
python -m puf_unlearning.generate_csv_results dataset="cifar10" alpha=0.3

python -m puf_unlearning.generate_csv_results dataset="cifar100" alpha=0.1

```