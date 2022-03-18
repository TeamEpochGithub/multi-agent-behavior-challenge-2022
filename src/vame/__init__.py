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

from vame.analysis import (
    community,
    community_videos,
    generative_model,
    gif,
    motif_videos,
    pose_segmentation,
    visualization,
)
from vame.initialize_project import init_new_project
from vame.model import create_trainset, evaluate_model, train_model
from vame.util import auxiliary
from vame.util.align_egocentrical import egocentric_alignment
from vame.util.auxiliary import update_config
from vame.util.csv_to_npy import csv_to_numpy

sys.dont_write_bytecode = True
