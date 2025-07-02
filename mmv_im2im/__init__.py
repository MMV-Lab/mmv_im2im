# -*- coding: utf-8 -*-

"""Top-level package for MMV Im2Im Transformation."""

__author__ = "Jianxu Chen"
__email__ = "jianxuchen.ai@gmail.com"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.5.6"


def get_module_version():
    return __version__


from .proj_tester import ProjectTester  # noqa F401
from .proj_trainer import ProjectTrainer  # noqa F401
