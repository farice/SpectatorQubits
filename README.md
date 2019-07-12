# SpectatorQubits

## Setup

1.  Install pip and virtualenv

On MacOS

```bash
sudo easy_install pip
sudo pip install --upgrade virtualenv
```

2. Clone the repository and install the requirements

```bash
git clone https://github.com/farice/SpectatorQubits
cd SpectatorQubits
mkdir ~/qec-env
virtualenv ~/qec-env
source ~/qec-env/bin/activate
python3 -m pip install -r requirements.txt
```

> `qec-env` is an arbitrary directory. You can replace `qec-env` with a directory of your choice.

## View the notebooks

We recommend jupyter lab for viewing.

```bash
jupyter lab
```

## Layout

- `simple_classification.ipynb`: coherent vs. incoherent noise classification
- `mixed_coherent_classification.ipynb`: mixed incoherent and coherent error channel classification
- `asym_depolarizing.ipynb`: asymmetric depolarizing channel classification

## Deactivate

Once finished deactivate the virtual environment

```bash
(rl-env)$ deactivate
```
