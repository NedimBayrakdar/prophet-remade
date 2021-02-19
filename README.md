prophet-remade
==============================

Minimal version of Prophet remade in Pyro

# Getting started

Before running any commands/notebooks, we first need to create our conda environment
and install our code into the environment. This can be done as follows::

    make env
    conda activate prophet-remade
    python -m pip install -e .


# Usage



## Running the CI steps (black, pylint, etc.)

Pre-commit (https://pre-commit.com/) is used to run all CI steps aside from the
unit tests (which typically take longer to run). You can run the CI steps using:

    make ci

To automatically run the CI steps on each commit you need to install pre-commit's
git hooks. This only needs to be done once using:

    pre-commit install


Credits
-------

Nedim Bayrakdar (bayrakdar.nedim@gmail.com)

References
----------

## Project structure

```

├── notebooks           <- Jupyter notebooks.
├── src
│   └── prophet_remade  <- Directory containing your Python modules/sub-packages.
├── tests               <- Tests for your Python modules/sub-packages.
├── azure-pipelines.yml <- File describing the CI steps for Azure DevOps.
├── environment.yml     <- File describing the conda environment.
├── Makefile            <- Makefile with commands like `make env` or `make test`.
├── README.md           <- The top-level README for developers using this project.
└── setup.py            <- Setup file for installing your code as a package.
```
