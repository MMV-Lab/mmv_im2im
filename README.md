# MMV Im2Im Transformation

[![Build Status](https://github.com/MMV-Lab/mmv_im2im/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/mmv_im2im/actions)
[![Documentation](https://github.com/MMV-Lab/mmv_im2im/workflows/Documentation/badge.svg)](https://MMV-Lab.github.io/mmv_im2im/)
[![Code Coverage](https://codecov.io/gh/MMV-Lab/mmv_im2im/branch/main/graph/badge.svg)](https://codecov.io/gh/MMV-Lab/mmv_im2im)

A python package for deep learing based image to image transformation

---

## Features

-   Store values and retain the prior value in memory
-   ... some other functionality

## Quick Start

Here, we use a [3D labelfree determination](https://www.allencell.org/label-free-determination.html#:~:text=The%20Label-Free%20Determination%20model%20can%20leverage%20the%20specificity,structures.%20How%20does%20the%20label-free%20determination%20model%20work%3F) as a test case.

First, we pull 100 examples from the AllenCell quilt bucket by running the following. Make sure you change the `parent_path` before running.
```bash
python  scripts/generate_synthetic_data.py
```

Then, we can train a labelfree model like this:
```base
run_im2im --config ./mmv_im2im/example_configs/example_labelfree.yaml  --mode train --debug
```
Make sure you double check all the file paths in the yaml file.


## Installation

**Stable Release:** `pip install mmv_im2im`<br>
**Development Head:** `pip install git+https://github.com/MMV-Lab/mmv_im2im.git`

## Documentation

For full package documentation please visit [MMV-Lab.github.io/mmv_im2im](https://MMV-Lab.github.io/mmv_im2im).

## Notes on the package design

There are three levels of abstraction: 
- `proj_trainer.py` (main entry point to define data module, pytorch-lightning module, and trainer)
    -  `./data_modules/data_loaders.py` (Currently, all 2D/3D paired/unpaired data loader can share the same universal dataloader. We may add other generic dataloaders when really needed, otherwise, the current one is general enough to cover current applications)
    -  `./models/basic_XYZ.py` (the middle level wrapper for different categories of models: FCN, pix2pix, GAN, embedseg. We can add more when needed, e.g., for denoising. This is the pytorch-lightning module specific for this category of models. The specific model backbone, loss function, etc. can be easily specified via parameters in yaml config)
        -  Other scripts under `./models/` and `preprocessing`, as well as `utils`, are low level functions to instantiate the pytorch-lightining module defined in `basic_XYZ.py` or `data_loader.py`


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## The Four Commands You Need To Know

1. `pip install -e .[dev]`

    This will install your package in editable mode with all the required development
    dependencies (i.e. `tox`).

2. `make build`

    This will run `tox` which will run all your tests in both Python 3.7
    and Python 3.8 as well as linting your code.

3. `make clean`

    This will clean up various Python and build generated files so that you can ensure
    that you are working in a clean environment.

4. `make docs`

    This will generate and launch a web browser to view the most up-to-date
    documentation for your Python package.

#### Additional Optional Setup Steps:

-   Turn your project into a GitHub repository:
    -   Make an account on [github.com](https://github.com)
    -   Go to [make a new repository](https://github.com/new)
    -   _Recommendations:_
        -   _It is strongly recommended to make the repository name the same as the Python
            package name_
        -   _A lot of the following optional steps are *free* if the repository is Public,
            plus open source is cool_
    -   After a GitHub repo has been created, run the commands listed under:
        "...or push an existing repository from the command line"
-   Register your project with Codecov:
    -   Make an account on [codecov.io](https://codecov.io)(Recommended to sign in with GitHub)
        everything else will be handled for you.
-   Ensure that you have set GitHub pages to build the `gh-pages` branch by selecting the
    `gh-pages` branch in the dropdown in the "GitHub Pages" section of the repository settings.
    ([Repo Settings](https://github.com/MMV-Lab/mmv_im2im/settings))
-   Register your project with PyPI:
    -   Make an account on [pypi.org](https://pypi.org)
    -   Go to your GitHub repository's settings and under the
        [Secrets tab](https://github.com/MMV-Lab/mmv_im2im/settings/secrets/actions),
        add a secret called `PYPI_TOKEN` with your password for your PyPI account.
        Don't worry, no one will see this password because it will be encrypted.
    -   Next time you push to the branch `main` after using `bump2version`, GitHub
        actions will build and deploy your Python package to PyPI.

#### Suggested Git Branch Strategy

1. `main` is for the most up-to-date development, very rarely should you directly
   commit to this branch. GitHub Actions will run on every push and on a CRON to this
   branch but still recommended to commit to your development branches and make pull
   requests to main. If you push a tagged commit with bumpversion, this will also release to PyPI.
2. Your day-to-day work should exist on branches separate from `main`. Even if it is
   just yourself working on the repository, make a PR from your working branch to `main`
   so that you can ensure your commits don't break the development head. GitHub Actions
   will run on every push to any branch or any pull request from any branch to any other
   branch.
3. It is recommended to use "Squash and Merge" commits when committing PR's. It makes
   each set of changes to `main` atomic and as a side effect naturally encourages small
   well defined PR's.


**MIT license**

