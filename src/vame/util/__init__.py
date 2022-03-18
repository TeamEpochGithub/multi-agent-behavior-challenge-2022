#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational Animal Motion Embedding 0.1 Toolbox
© K. Luxem & P. Bauer, Department of Cellular Neuroscience
Leibniz Institute for Neurobiology, Magdeburg, Germany

https://github.com/LINCellularNeuroscience/VAME
Licensed under GNU General Public License v3.0
"""
import sys

from vame.util.auxiliary import create_config_template, read_config, update_config, write_config

sys.dont_write_bytecode = True

__all__ = [
    "create_config_template",
    "read_config",
    "write_config",
    "update_config",
]
