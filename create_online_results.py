import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import pylab as plt
import pandas as pd
import quantities as pq
import os, sys
from os.path import join as join
import seaborn as sns
import yaml
from plotting_convention import *

datatype = 'convolution'
# root = os.path.dirname(os.path.realpath(sys.argv[0]))
root = './'
folder = 'recordings/convolution/gtica'
probe=['SqMEA-10-15um']
avail_probes = os.listdir(folder)

all_recordings = []
results_df = []
results_files = ['results.csv']

# for av in avail_probes:
#     if any([p in av] for p in probe):


for p in probe:
    all_recordings.extend([join(root, folder, p, f) for f in os.listdir(join(root, folder, p))])

for rec in all_recordings:
    print rec
    with open(join(rec, 'rec_info.yaml'), 'r') as f:
        info = yaml.load(f)

    duration = float(info['Spikegen']['duration'].split()[0])
    seed = info['General']['seed']
    ncells = int(info['Spikegen']['n_cells'])
    noise_lev = float(info['Noise']['noise_level'])
    electrode = info['General']['electrode name']

    print duration, ncells, noise_lev

    for res in results_files:
        if os.path.isfile(join(rec, res)):
            with open(join(rec, res), 'r') as f:
                df = pd.read_csv(f)
            nobs = len(df)
            dur_vec = [duration] * nobs
            seed_vec = [seed] * nobs
            ncells_vec = [ncells] * nobs
            noise_vec = [noise_lev] * nobs
            elec_vec = [electrode] * nobs

            df['duration'] = dur_vec
            df['seed'] = seed_vec
            df['ncells'] = ncells_vec
            df['noise'] = noise_vec
            df['electrode'] = elec_vec

            if len(results_df) == 0:
                results_df = df
            else:
                results_df = pd.concat([results_df, df])

results_df.index = range(len(results_df))
results_df.to_csv('results_online.csv')