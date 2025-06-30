#!/usr/bin/env python
# -*- coding: utf-8 -*-


from setuptools import setup, find_packages 


setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
    "numpy<2",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

data_requirements = [
    "quilt3",
    "pooch",
    "matplotlib",
    "notebook"
]

logger_requirements = [
    "tensorboard",
]



setup(
    packages=["mmv_im2im"],
    package_dir={"mmv_im2im": "mmv_im2im"},
   
)