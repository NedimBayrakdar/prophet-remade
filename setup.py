#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import setuptools

with open("README.md") as readme_file:
    readme = readme_file.read()


requirements = []
setup_requirements = []

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
)
