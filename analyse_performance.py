import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import quantities as pq
import os
from os.path import join as join
import seaborn as sns
import yaml

datatype = 'convolution'
spikesorters = ['ica', 'kilosort', 'klusta', 'spykingcircus', 'mountain', 'yass']

# duration = 10.0
# noise = 'all'
# ncells = 10
# probe = 'Neuronexus'

root = os.getcwd()
seed=2904
probe='SqMEA'

all_recordings = [join(root, 'recordings', datatype, f) for f in os.listdir(join(root, 'recordings', datatype))
                  if str(seed) in f and probe in f]

dur_vec, n_cells_vec, noise_vec, ss_vec, acc_vec, \
sens_vec, prec_vec, false_vec, miss_vec, time_vec = [], [], [], [], [], [], [], [], [], []

print "Creating dataframe from results: ", len(all_recordings),  " recordings"
for rec in all_recordings:
    with open(join(rec, 'rec_info.yaml'), 'r') as f:
        info = yaml.load(f)

    duration = float(info['Spikegen']['duration'].split()[0])
    n_cells = int(info['Spikegen']['n_cells'])
    noise_lev = float(info['Noise']['noise_level'])

    sorters = [sort for sort in os.listdir(rec) if os.path.isdir(join(rec, sort))]
    for sort in sorters:
        if sort in spikesorters:
            perf = np.load(join(rec, sort, 'results', 'performance.npy')).item()
            time = np.load(join(rec, sort, 'results', 'time.npy'))

            dur_vec.append(duration)
            n_cells_vec.append(n_cells)
            noise_vec.append(noise_lev)
            ss_vec.append(sort)
            acc_vec.append(perf['accuracy'])
            sens_vec.append(perf['sensitivity'])
            prec_vec.append(perf['precision'])
            false_vec.append(perf['false_disc_rate'])
            miss_vec.append(perf['miss_rate'])
            time_vec.append(float(time))

# create dataframe
data = {'duration': dur_vec, 'ncells': n_cells_vec, 'noise': noise_vec, 'spikesorter': ss_vec,
        'accuracy': acc_vec, 'sensitivity': sens_vec, 'precision': prec_vec, 'miss_rate': miss_vec,
        'false_rate': false_vec, 'time': time_vec}

dset = pd.DataFrame(data)

print "Complexity analysis"
duration = 5
dset_filt = dset[dset['duration']==duration]
noise = 5
dset_filt = dset_filt[dset_filt['noise']==noise]
fig1 = plt.figure()
ax11 = fig1.add_subplot(231)
sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11)
ax11.set_title('Accuracy')
ax12 = fig1.add_subplot(232)
sns.pointplot(x='ncells', y='precision', hue='spikesorter', data=dset_filt, ax=ax12)
ax12.set_title('Precision')
ax13 = fig1.add_subplot(233)
sns.pointplot(x='ncells', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax13)
ax13.set_title('Sensitivity')
ax21 = fig1.add_subplot(234)
sns.pointplot(x='ncells', y='miss_rate', hue='spikesorter', data=dset_filt, ax=ax21)
ax21.set_title('Miss Rate')
ax22 = fig1.add_subplot(235)
sns.pointplot(x='ncells', y='false_rate', hue='spikesorter', data=dset_filt, ax=ax22)
ax22.set_title('False Discovery Rate')
ax23 = fig1.add_subplot(236)
sns.pointplot(x='ncells', y='time', hue='spikesorter', data=dset_filt, ax=ax23)
ax23.set_title('Processing Time')

# print "Noise analysis"
# duration = 10
# dset_filt = dset[dset['duration']==duration]
# cells=20
# dset_filt = dset_filt[dset_filt['ncells']==cells]
# sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt)
#
# print "Time analysis"
# duration = 10
# dset_filt = dset[dset['duration']==duration]
# cells=20
# dset_filt = dset_filt[dset_filt['ncells']==cells]
# sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt)

plt.ion()
plt.show()






