# SpectatorQubits

## Setup (DCC Cluster)

1.  Install homebrew

Homebrew allows one to easily install dependencies at the user level. We will use this dependency manager to install the latest version of gcc, python3, open-mpi, and node:

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/Linuxbrew/install/master/install.sh)"
```

and follow the instructions to update your PATH. Now, we install the dependencies:

```
brew install gcc python3 open-mpi node
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

## Configure Jupyter Lab

```bash
jupyter notebook --generate-config
```

We will create a password for remote access:

```bash
jupyter notebook password
```

Then, edit `~/.jupyter/jupyter_notebook_config.py`. Modify the following flags:

```python
c.NotebookApp.open_browser = False
c.NotebookApp.password_required = True
c.NotebookApp.port = 8888
```

## View the notebooks

We recommend jupyter lab for viewing our experiments. The first time, one should rebuild the assets:

```bash
jupyter lab build
```

Now, we can start the remote server and access the web interface from our local machine:

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

# FAQ

## Deactivate virtual environment

Simply,

```bash
(rl-env)$ deactivate
```
