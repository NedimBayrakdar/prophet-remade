prophet-remade
==============================

Minimal version of Prophet remade in Pyro

# Getting started

Before running any commands/notebooks, we first need to create our conda environment
and install our code into the environment. This can be done as follows::

    make env
    conda activate prophet-remade
    python -m pip install -e .

Note that this command installs our code as an editable package so that changes in
your code are directly reflected in the environment. This allows you to import your
code using `import prophet_remade`. This can be combined with the
autoreload extension to automatically reload your modules on any change.

For development purposes, you can also set-up pre-commit to run CI steps locally
before each commit::

    pre-commit install

This will ensure that black, pylint, etc. are run whenever you commit your code,
speeding up the CI process as you don't have to wait for any build pipelines, etc.

# Usage



## Running the CI steps (black, pylint, etc.)

Pre-commit (https://pre-commit.com/) is used to run all CI steps aside from the
unit tests (which typically take longer to run). You can run the CI steps using:

    make ci

To automatically run the CI steps on each commit you need to install pre-commit's
git hooks. This only needs to be done once using:

    pre-commit install



**Although we provide default setup for the different tools, you can edit their config files to fit your personal preferences. For example, for pylint you can edit the .pylintrc file to fit your preferences.**

## Running (unit) tests

Tests can be run locally using the command ``make test``.
or
```pip install -e .[dev] ```

```pytest```


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
