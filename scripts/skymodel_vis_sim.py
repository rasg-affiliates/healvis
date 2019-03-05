#!/bin/env python

#SBATCH -J healvis
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH -A jpober-condo
#SBATCH --qos=jpober-condo
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adam_lanman@brown.edu

"""
Visibility simulation
"""

import numpy as np
import argparse
import os
import sys
import yaml
import pyuvsim
from healvis.utils import comoving_voxel_volume
from itertools import izip

import healvis

parser = argparse.ArgumentParser()
parser.add_argument(dest='param', help='obsparam yaml file')
args = parser.parse_args()

param_file = args.param

if 'SLURM_CPUS_PER_TASK' in os.environ:
    Nprocs = int(os.environ['SLURM_CPUS_PER_TASK'])
elif 'Nprocs' in param_dict:
    Nprocs = int(param_dict['Nprocs'])
else:
    Nprocs = 1

sjob_id = None
if 'SLURM_JOB_ID' in os.environ:
    sjob_id = os.environ['SLURM_JOB_ID']


healvis.simulator.run_simulation(param_file, Nprocs=Nprocs, sjob_id=sjob_id)
