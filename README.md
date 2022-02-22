# Notebooks

This repository contains Jupyter notebooks.

# Development

```sh
# Create a virtualenv and activate it
virtualenv -p python3 venv
source venv/bin/activate

# Install requirements
pip install - requirements.txt

# Install the package
python setup.py develop

jupyterlab
```

# JupyterLab extensions

```sh
# tracking notebooks
jupytext --set-formats .ipynb,.md <notebook>.ipynb

# install matplotlib widgets
pip install ipympl
pip install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension


```