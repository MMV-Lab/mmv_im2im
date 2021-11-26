#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging


###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTester(object):
    """
    entry for training models

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, init_value: int = 10):
        # Check initial value
        self._check_value(init_value)

        # Set values
        self.current_value = init_value
        self.old_value = None

    def run_inference(self):
        pass
