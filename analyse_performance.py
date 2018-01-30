import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import quantities as pq
import os
from os.path import join as join
import seaborn as sns
import yaml
from plotting_convention import *

datatype = 'convolution'
# spikesorters = ['ica', 'kilosort', 'klusta', 'spykingcircus', 'mountain', 'yass']
spikesorters = ['ica', 'mountain', 'spykingcircus'] #, 'yass']

# duration = 10.0
# noise = 'all'
# ncells = 10
# probe = 'Neuronexus'

root = os.getcwd()
seed=2904
probe='SqMEA'

all_recordings = [join(root, 'recordings', datatype, probe, f) for f in os.listdir(join(root, 'recordings',
                                                                                        datatype, probe))
                  if str(seed) in f and probe in f]

print all_recordings

dur_vec, n_cells_vec, noise_vec, ss_vec, acc_vec, \
sens_vec, prec_vec, false_vec, miss_vec, misclass_vec, time_vec = [], [], [], [], [], [], [], [], [], [], []

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
            if os.path.isfile(join(rec, sort, 'results', 'performance.npy')):
                perf = np.load(join(rec, sort, 'results', 'performance.npy')).item()
                counts = np.load(join(rec, sort, 'results', 'counts.npy')).item()
                misclass = float(counts['CL']+counts['CLO']+counts['CLSO'])/float(counts['TOT_GT']) * 100
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
                misclass_vec.append(misclass)
                time_vec.append(float(time))

# create dataframe
data = {'duration': dur_vec, 'ncells': n_cells_vec, 'noise': noise_vec, 'spikesorter': ss_vec,
        'accuracy': acc_vec, 'sensitivity': sens_vec, 'precision': prec_vec, 'miss_rate': miss_vec,
        'false_rate': false_vec, 'misclassification': misclass_vec,'time': time_vec}

dset = pd.DataFrame(data)

# print "Complexity analysis"
duration = 10
dset_filt = dset[dset['duration']==duration]
noise = 10
dset_filt = dset_filt[dset_filt['noise']==noise]
fig_comp = plt.figure(figsize=(9, 7))
fig_comp.suptitle('Complexity', fontsize=30)

ax11 = fig_comp.add_subplot(221)
sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.set_title('Accuracy (%)', y=1.02)
ax11.set_ylim([-5, 105])
legend = ax11.legend()
legend.remove()

ax12 = fig_comp.add_subplot(222)
sns.pointplot(x='ncells', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.set_title('Sensitivity (%)', y=1.02)
ax12.set_ylim([-5, 105])
ax12.legend()

ax21 = fig_comp.add_subplot(223)
sns.pointplot(x='ncells', y='precision', hue='spikesorter', data=dset_filt, ax=ax21)
ax21.set_xlabel('')
ax21.set_ylabel('')
ax21.set_title('Precision (%)', y=1.02)
ax21.set_ylim([-5, 105])
legend = ax21.legend()
legend.remove()

ax22 = fig_comp.add_subplot(224)
sns.pointplot(x='ncells', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22)
ax22.set_title('Misclassification (%)', y=1.02)
ax22.set_ylabel('')
ax22.set_ylim([-5, 40])
legend = ax22.legend()
legend.remove()
# fig_comp.tight_layout()

mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
simplify_axes([ax11, ax12, ax21, ax22])
fig_comp.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)

print "Noise analysis"
duration = 10
dset_filt = dset[dset['duration']==duration]
ncells = 10
dset_filt = dset_filt[dset_filt['ncells']==ncells]
fig_noise = plt.figure(figsize=(9, 7))
fig_noise.suptitle('Noise', fontsize=30)

ax11 = fig_noise.add_subplot(221)
sns.pointplot(x='noise', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.set_ylim([-5, 105])
ax11.set_title('Accuracy (%)', y=1.02)
legend = ax11.legend()
legend.remove()

ax12 = fig_noise.add_subplot(222)
sns.pointplot(x='noise', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.set_ylim([-5, 105])
ax12.set_title('Sensitivity (%)', y=1.02)
ax12.legend()

ax21 = fig_noise.add_subplot(223)
sns.pointplot(x='noise', y='precision', hue='spikesorter', data=dset_filt, ax=ax21)
ax21.set_xlabel('')
ax21.set_ylabel('')
ax21.set_title('Precision (%)', y=1.02)
ax21.set_ylim([-5, 105])
legend = ax21.legend()
legend.remove()

ax22 = fig_noise.add_subplot(224)
sns.pointplot(x='noise', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22)
ax22.set_title('Misclassification (%)', y=1.02)
ax22.set_ylabel('')
ax22.set_ylim([-5, 40])
legend = ax22.legend()
legend.remove()

mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
simplify_axes([ax11, ax12, ax21, ax22])
fig_noise.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)

# fig_noise.tight_layout()

print "Duration analysis"
noise = 10
dset_filt = dset[dset['noise']==noise]
ncells = 20
dset_filt = dset_filt[dset_filt['ncells']==ncells]
fig_dur = plt.figure(figsize=(9, 7))
fig_dur.suptitle('Duration', fontsize=30)

ax11 = fig_dur.add_subplot(221)
sns.pointplot(x='duration', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11)
ax11.set_xlabel('')
ax11.set_ylabel('')
ax11.set_ylim([-5, 105])
ax11.set_title('Accuracy (%)', y=1.02)
legend = ax11.legend()
legend.remove()

ax12 = fig_dur.add_subplot(222)
sns.pointplot(x='duration', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12)
ax12.set_xlabel('')
ax12.set_ylabel('')
ax12.set_ylim([-5, 105])
ax12.set_title('Sensitivity (%)', y=1.02)
ax12.legend()

ax21 = fig_dur.add_subplot(223)
sns.pointplot(x='duration', y='precision', hue='spikesorter', data=dset_filt, ax=ax21)
ax21.set_xlabel('')
ax21.set_ylabel('')
ax21.set_title('Precision (%)', y=1.02)
ax21.set_ylim([-5, 105])
legend = ax21.legend()
legend.remove()

ax22 = fig_dur.add_subplot(224)
sns.pointplot(x='duration', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22)
ax22.set_title('Misclassification (%)', y=1.02)
ax22.set_ylabel('')
ax22.set_ylim([-5, 40])
legend = ax22.legend()
legend.remove()

mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
simplify_axes([ax11, ax12, ax21, ax22])
fig_dur.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)

plt.ion()
plt.show()






