#!/bin/env python
# -*- mode: python; coding: utf-8 -*
# Copyright (c) 2019 Radio Astronomy Software Group
# Licensed under the 3-clause BSD License

#SBATCH -J healvis
#SBATCH -t 2-00:00:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=100G
#SBATCH -A jpober-condo
#SBATCH --qos=jpober-condo
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adam_lanman@brown.edu

from __future__ import absolute_import, division, print_function

# -----------------------
# Visibility simulation wrapper script.
# Designed for use in Simple Linux Utility for Resource Management (SLURM) clusters.
# -----------------------

import argparse
import os

import healvis

parser = argparse.ArgumentParser()
parser.add_argument(dest='param', help='obsparam yaml file')
parser.add_argument('-n', dest='Nproc', help='Number of processes (overrides SLURM Ncpus)', type=int)
args = parser.parse_args()

param_file = args.param

if args.Nproc is not None:
    print("Nprocs: ", args.Nproc)
    Nprocs = args.Nproc
elif 'SLURM_CPUS_PER_TASK' in os.environ:
    Nprocs = int(os.environ['SLURM_CPUS_PER_TASK'])
else:
    Nprocs = 1

sjob_id = None
if 'SLURM_JOB_ID' in os.environ:
    sjob_id = os.environ['SLURM_JOB_ID']


healvis.simulator.run_simulation(param_file, Nprocs=Nprocs, sjob_id=sjob_id)
