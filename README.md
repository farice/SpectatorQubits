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
git clone git@github.com:farice/SpectatorQubits # SSH pub key must be linked to your GH account
cd SpectatorQubits
python3 -m venv ~/qec-env
source ~/qec-env/bin/activate
python3 -m pip install --user -U -r requirements.txt
```

> `qec-env` is an arbitrary directory. You can replace `qec-env` with a directory of your choice.

## View the notebooks

We recommend jupyter lab for viewing.

```bash
jupyter lab # remote VM
ssh -N -f -M -S /tmp/session -L 8000:localhost:8888 USER@dcc-slogin-01.oit.duke.edu # local machine
# open localhost:8000 in web browser on local machine
```

Remember to close the ssh session from the local machine, when finished:

```bash
ssh -S /tmp/session -O exit USER@dcc-slogin-01.oit.duke.edu
```

in addition to shutting down the jupyter lab server on the remote VM.

## Layout

- `simple_classification.ipynb`: coherent vs. incoherent noise classification
- `mixed_coherent_classification.ipynb`: mixed incoherent and coherent error channel classification
- `asym_depolarizing.ipynb`: asymmetric depolarizing channel classification

## Deactivate

Once finished deactivate the virtual environment

```bash
(rl-env)$ deactivate
```
