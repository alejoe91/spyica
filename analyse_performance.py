import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import quantities as pq
import os
from os.path import join as join
import seaborn as sns
import yaml

datatype = 'convolution'
spikesorters = ['ica', 'kilosort', 'klusta', 'spykingcircus', 'mountainsort', 'yass']

duration = 10.0
noise = 'all'
ncells = 10
probe = 'Neuronexus'

root = os.getcwd()

all_recordings = [join(root, 'recordings', datatype, f) for f in os.listdir(join(root, 'recordings', datatype))]
# print all_recordings

if noise == 'all':
    conditions = [str(duration)+'s', '_' + str(ncells) + '_', probe]
    noise_rec = all_recordings
    for cond in conditions:
        noise_rec = [f for f in noise_rec if cond in f]

    dur_vec, n_cells_vec, noise_vec, ss_vec, acc_vec, \
    sens_vec, prec_vec, false_vec, miss_vec, time_vec = [], [], [], [], [], [], [], [], [], []

    for rec in noise_rec:
        with open(join(rec, 'rec_info.yaml'), 'r') as f:
            info = yaml.load(f)

        duration = float(info['Spikegen']['duration'].split()[0])
        n_cells = int(info['Spikegen']['n_cells'])
        noise_lev = float(info['Noise']['noise_level'])

        sorters = [sort for sort in os.listdir(rec) if os.path.isdir(join(rec, sort))]
        for sort in sorters:
            if sort in spikesorters:
                print sort

                perf = np.load(join(rec, sort, 'results', 'performance.npy')).item()
                # time = np.load(join(rec, sort, 'results', 'time.npy'))

                dur_vec.append(duration)
                n_cells_vec.append(n_cells)
                noise_vec.append(noise_lev)
                ss_vec.append(sort)
                acc_vec.append(perf['accuracy'])
                sens_vec.append(perf['sensitivity'])
                prec_vec.append(perf['precision'])
                false_vec.append(perf['false_disc_rate'])
                miss_vec.append(perf['miss_rate'])
                # time_vec.append(float(time))

    # create dataframe
    data = {'duration': dur_vec, 'ncells': n_cells_vec, 'noise': noise_vec, 'spikesorter': ss_vec,
            'accuracy': acc_vec, 'sensitivity': sens_vec, 'precision': prec_vec, 'miss_rate': miss_vec,
            'false_rate': false_vec} #, 'time': time_vec}
    dset = pd.DataFrame(data)


