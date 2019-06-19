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
git clone https://github.com/farice/spectatorqubits
cd spectatorqubits
mkdir ~/qec-env
virtualenv ~/qec-env
source ~/qec-env/bin/activate
pip install -r requirements.txt
```

> `rl-env` is an arbitrary directory. You can replace `rl-env` with a directory of your choice.

## View the notebooks

```bash
jupyter notebook
```

## Deactivate

Once finished deactivate the virtual environment

```bash
(rl-env)$ deactivate
```
