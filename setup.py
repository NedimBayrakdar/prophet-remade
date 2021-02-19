#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools

with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements-dev.txt") as f:
    test_requirements = f.read().splitlines()

requirements = []
setup_requirements = []
extra_requirements = {"dev": test_requirements}

setuptools.setup(
    name="prophet_remade",
    author="Nedim Bayrakdar",
    author_email="bayrakdar.nedim@gmail.com",
    description="Minimal version of Prophet remade in Pyro",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    version="0.1.0",
    install_requires=requirements,
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
)
