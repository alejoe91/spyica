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
root = os.path.dirname(os.path.realpath(sys.argv[0]))
folder = 'recordings/convolution/gtica'
probe=['SqMEA', 'Neuronexus']
avail_probes = os.listdir(folder)

### ORICA ALGORITHM ###
block_analysis = False
npass_analysis = True
ff_analysis = True
lambda_analysis = False
reg_analysis = True

with open('results_all.csv', 'r') as f:
    results_df = pd.read_csv(f)

ncells = [30]
noise = [10]
duration = [10]
probe=['Neuronexus-32-cut-30']
probe=['SqMEA-15-10um']
results_df_ica = results_df[results_df['mod']=='ica']
results_df = results_df[results_df.oricamode.isin(['original', 'A_block', 'W_block'])]

df_ica = results_df_ica
df_ica = df_ica[df_ica.ncells.isin(ncells)]
df_ica = df_ica[df_ica.noise.isin(noise)]
df_ica = df_ica[df_ica.duration.isin(duration)]
df_ica = df_ica[df_ica.electrode.isin(probe)]

compare_with='CC_source'

if block_analysis:
    ff='cooling'
    lambda_n=0.995

    df_block = results_df
    df_block = df_block[df_block.ncells.isin(ncells)]
    df_block = df_block[df_block.noise.isin(noise)]
    df_block = df_block[df_block.duration.isin(duration)]
    df_block = df_block[df_block.electrode.isin(probe)]
    df_block = df_block[df_block.ff == ff]
    df_block = df_block[df_block['lambda']==str(lambda_n)]

    fig_block = plt.figure()
    ax_bl = fig_block.add_subplot(121)
    ax_time = fig_block.add_subplot(122)

    ax_bl = sns.pointplot(x='block',y=compare_with, hue='oricamode', data=df_block, ax=ax_bl)
    ax_time = sns.pointplot(x='block', y='time', hue='oricamode', data=df_block, ax=ax_time)

    # sns.tsplot(data=[np.mean(df_ica.CC_mix)]*len(np.unique(df_block.block)), ci=np.std(df_ica.CC_mix), color='grey', ax=ax_bl)

    ax_bl.axhline(np.mean(df_ica[compare_with]), color='grey', alpha=0.8)
    ax_bl.axhline(np.mean(df_ica[compare_with]) + np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)
    ax_bl.axhline(np.mean(df_ica[compare_with]) - np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)

    ax_time.axhline(np.mean(df_ica.time), color='grey', alpha=0.8)
    ax_time.axhline(np.mean(df_ica.time) + np.std(df_ica.time), color='grey', ls='--', alpha=0.3)
    ax_time.axhline(np.mean(df_ica.time) - np.std(df_ica.time), color='grey', ls='--', alpha=0.3)



if ff_analysis:

    block = 50
    mu = 0
    df_ff = results_df
    df_ff = df_ff[df_ff.ncells.isin(ncells)]
    df_ff = df_ff[df_ff.noise.isin(noise)]
    df_ff = df_ff[df_ff.duration.isin(duration)]
    df_ff = df_ff[df_ff.electrode.isin(probe)]
    df_ff = df_ff[df_ff.block==block]
    df_ff = df_ff[df_ff.mu==mu]

    df_cooling = df_ff[df_ff.ff=='cooling']
    df_constant = df_ff[df_ff.ff=='constant']

    fig_ff = plt.figure()
    ax_cool = fig_ff.add_subplot(121)
    ax_cool.set_title('COOLING')
    sns.pointplot(x='lambda',y=compare_with, hue='oricamode', data=df_cooling, ax=ax_cool)

    ax_cool.axhline(np.mean(df_ica[compare_with]), color='grey', alpha=0.8)
    ax_cool.axhline(np.mean(df_ica[compare_with]) + np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)
    ax_cool.axhline(np.mean(df_ica[compare_with]) - np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)

    ax_const = fig_ff.add_subplot(122)
    ax_const.set_title('CONSTANT')
    sns.pointplot(x='lambda', y=compare_with, hue='oricamode', data=df_constant, ax=ax_const)

    ax_const.axhline(np.mean(df_ica[compare_with]), color='grey', alpha=0.7)
    ax_const.axhline(np.mean(df_ica[compare_with]) + np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)
    ax_const.axhline(np.mean(df_ica[compare_with]) - np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)

if reg_analysis:
    block = 50
    ff='constant'
    lambda_n = 'N'
    df_reg = results_df
    df_reg = df_reg[df_reg.ncells.isin(ncells)]
    df_reg = df_reg[df_reg.noise.isin(noise)]
    df_reg = df_reg[df_reg.duration.isin(duration)]
    df_reg = df_reg[df_reg.electrode.isin(probe)]
    df_reg = df_reg[df_reg.block==block]
    df_reg = df_reg[df_reg.ff==ff]
    df_reg = df_reg[df_reg['lambda']==str(lambda_n)]

    df_A_block = df_reg[df_reg.oricamode=='A_block']
    df_W_block = df_reg[df_reg.oricamode=='W_block']

    fig_ff = plt.figure()
    ax_A = fig_ff.add_subplot(121)
    ax_A.set_title('A_block')
    sns.pointplot(x='mu',y=compare_with, hue='reg', data=df_A_block, ax=ax_A)

    ax_A.axhline(np.mean(df_ica[compare_with]), color='grey', alpha=0.8)
    ax_A.axhline(np.mean(df_ica[compare_with]) + np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)
    ax_A.axhline(np.mean(df_ica[compare_with]) - np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)

    ax_W = fig_ff.add_subplot(122)
    ax_W.set_title('W_block')
    sns.pointplot(x='mu', y=compare_with, hue='reg', data=df_W_block, ax=ax_W)

    ax_W.axhline(np.mean(df_ica[compare_with]), color='grey', alpha=0.8)
    ax_W.axhline(np.mean(df_ica[compare_with]) + np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)
    ax_W.axhline(np.mean(df_ica[compare_with]) - np.std(df_ica[compare_with]), color='grey', ls='--', alpha=0.3)


plt.ion()
plt.show()




# dur_vec, n_cells_vec, noise_vec, ss_vec, acc_vec, \
# sens_vec, prec_vec, false_vec, miss_vec, misclass_vec, time_vec = [], [], [], [], [], [], [], [], [], [], []
#
# print "Creating dataframe from results: ", len(all_recordings),  " recordings"
# for rec in all_recordings:
#     with open(join(rec, 'rec_info.yaml'), 'r') as f:
#         info = yaml.load(f)
#
#     duration = float(info['Spikegen']['duration'].split()[0])
#     n_cells = int(info['Spikegen']['n_cells'])
#     noise_lev = float(info['Noise']['noise_level'])
#
#     sorters = [sort for sort in os.listdir(rec) if os.path.isdir(join(rec, sort))]
#     for sort in sorters:
#         if sort in spikesorters:
#             if os.path.isfile(join(rec, sort, 'results', 'performance.npy')):
#                 perf = np.load(join(rec, sort, 'results', 'performance.npy')).item()
#                 counts = np.load(join(rec, sort, 'results', 'counts.npy')).item()
#                 misclass = float(counts['CL']+counts['CLO']+counts['CLSO'])/float(counts['TOT_GT']) * 100
#                 time = np.load(join(rec, sort, 'results', 'time.npy'))
#
#                 dur_vec.append(duration)
#                 n_cells_vec.append(n_cells)
#                 noise_vec.append(noise_lev)
#                 ss_vec.append(sort)
#                 acc_vec.append(perf['accuracy'])
#                 sens_vec.append(perf['sensitivity'])
#                 prec_vec.append(perf['precision'])
#                 false_vec.append(perf['false_disc_rate'])
#                 miss_vec.append(perf['miss_rate'])
#                 misclass_vec.append(misclass)
#                 time_vec.append(float(time))
#
# # create dataframe
# data = {'duration': dur_vec, 'ncells': n_cells_vec, 'noise': noise_vec, 'spikesorter': ss_vec,
#         'accuracy': acc_vec, 'sensitivity': sens_vec, 'precision': prec_vec, 'miss_rate': miss_vec,
#         'false_rate': false_vec, 'misclassification': misclass_vec,'time': time_vec}
#
# dset_original = pd.DataFrame(data)
#
# if embc_analysis:
#     dset = dset_original[dset_original.spikesorter != 'orica']
#     # print "Complexity analysis"
#     duration = 10
#     dset_filt = dset[dset['duration']==duration]
#     noise = 10
#     dset_filt = dset_filt[dset_filt['noise']==noise]
#     fig_comp = plt.figure(figsize=(9, 7))
#     fig_comp.suptitle('Complexity', fontsize=30)
#
#     ax11 = fig_comp.add_subplot(221)
#     sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_title('Accuracy (%)', y=1.02)
#     ax11.set_ylim([-5, 105])
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_comp.add_subplot(222)
#     sns.pointplot(x='ncells', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.set_ylim([-5, 105])
#     ax12.legend()
#
#     ax21 = fig_comp.add_subplot(223)
#     sns.pointplot(x='ncells', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Neurons', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_comp.add_subplot(224)
#     sns.pointplot(x='ncells', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'], lw=1)
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_xlabel('Neurons', fontsize=20)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     legend = ax22.legend()
#     legend.remove()
#     # fig_comp.tight_layout()
#
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_comp.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
#     print "Noise analysis"
#     duration = 10
#     dset_filt = dset[dset['duration']==duration]
#     ncells = 20
#     dset_filt = dset_filt[dset_filt['ncells']==ncells]
#     fig_noise = plt.figure(figsize=(9, 7))
#     fig_noise.suptitle('Noise', fontsize=30)
#
#     ax11 = fig_noise.add_subplot(221)
#     sns.pointplot(x='noise', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_ylim([-5, 105])
#     ax11.set_title('Accuracy (%)', y=1.02)
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_noise.add_subplot(222)
#     sns.pointplot(x='noise', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_ylim([-5, 105])
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.legend()
#
#     ax21 = fig_noise.add_subplot(223)
#     sns.pointplot(x='noise', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Noise ($\mu V$)', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_noise.add_subplot(224)
#     sns.pointplot(x='noise', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     ax22.set_xlabel('Noise ($\mu V$)', fontsize=20)
#     legend = ax22.legend()
#     legend.remove()
#
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_noise.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
#     # fig_noise.tight_layout()
#
#     print "Duration analysis"
#     noise = 10
#     dset_filt = dset[dset['noise']==noise]
#     ncells = 20
#     dset_filt = dset_filt[dset_filt['ncells']==ncells]
#     fig_dur = plt.figure(figsize=(9, 7))
#     fig_dur.suptitle('Duration', fontsize=30)
#
#     ax11 = fig_dur.add_subplot(221)
#     sns.pointplot(x='duration', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_ylim([-5, 105])
#     ax11.set_title('Accuracy (%)', y=1.02)
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_dur.add_subplot(222)
#     sns.pointplot(x='duration', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_ylim([-5, 105])
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.legend(loc='lower right')
#
#     ax21 = fig_dur.add_subplot(223)
#     sns.pointplot(x='duration', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^','o','d'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Duration ($s$)', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_dur.add_subplot(224)
#     sns.pointplot(x='duration', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22,
#                   linestyles=['--', '-.', '-'], markers=['^','o','d'], lw=0.5)
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     ax22.set_xlabel('Duration ($s$)', fontsize=20)
#     legend = ax22.legend()
#     legend.remove()
#
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_dur.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
# if orica_analysis:
#     dset = dset_original[dset_original.spikesorter.isin(['ica', 'orica'])]
#     # print "Complexity analysis"
#     duration = 10
#     dset_filt = dset[dset['duration'] == duration]
#     noise = 10
#     dset_filt = dset_filt[dset_filt['noise'] == noise]
#     fig_comp = plt.figure(figsize=(9, 7))
#     fig_comp.suptitle('Complexity', fontsize=30)
#
#     ax11 = fig_comp.add_subplot(221)
#     sns.pointplot(x='ncells', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_title('Accuracy (%)', y=1.02)
#     ax11.set_ylim([-5, 105])
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_comp.add_subplot(222)
#     sns.pointplot(x='ncells', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.set_ylim([-5, 105])
#     ax12.legend()
#
#     ax21 = fig_comp.add_subplot(223)
#     sns.pointplot(x='ncells', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Neurons', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_comp.add_subplot(224)
#     sns.pointplot(x='ncells', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22,
#                   markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'], lw=1)
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_xlabel('Neurons', fontsize=20)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     legend = ax22.legend()
#     legend.remove()
#     # fig_comp.tight_layout()
#
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_comp.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
#     print "Noise analysis"
#     duration = 10
#     dset_filt = dset[dset['duration'] == duration]
#     ncells = 20
#     dset_filt = dset_filt[dset_filt['ncells'] == ncells]
#     fig_noise = plt.figure(figsize=(9, 7))
#     fig_noise.suptitle('Noise', fontsize=30)
#
#     ax11 = fig_noise.add_subplot(221)
#     sns.pointplot(x='noise', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_ylim([-5, 105])
#     ax11.set_title('Accuracy (%)', y=1.02)
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_noise.add_subplot(222)
#     sns.pointplot(x='noise', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_ylim([-5, 105])
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.legend()
#
#     ax21 = fig_noise.add_subplot(223)
#     sns.pointplot(x='noise', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Noise ($\mu V$)', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_noise.add_subplot(224)
#     sns.pointplot(x='noise', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     ax22.set_xlabel('Noise ($\mu V$)', fontsize=20)
#     legend = ax22.legend()
#     legend.remove()
#
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_noise.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
#     # fig_noise.tight_layout()
#
#     print "Duration analysis"
#     noise = 10
#     dset_filt = dset[dset['noise'] == noise]
#     ncells = 20
#     dset_filt = dset_filt[dset_filt['ncells'] == ncells]
#     fig_dur = plt.figure(figsize=(9, 7))
#     fig_dur.suptitle('Duration', fontsize=30)
#
#     ax11 = fig_dur.add_subplot(221)
#     sns.pointplot(x='duration', y='accuracy', hue='spikesorter', data=dset_filt, ax=ax11, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax11.set_xlabel('')
#     ax11.set_ylabel('')
#     ax11.set_ylim([-5, 105])
#     ax11.set_title('Accuracy (%)', y=1.02)
#     legend = ax11.legend()
#     legend.remove()
#
#     ax12 = fig_dur.add_subplot(222)
#     sns.pointplot(x='duration', y='sensitivity', hue='spikesorter', data=dset_filt, ax=ax12, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax12.set_xlabel('')
#     ax12.set_ylabel('')
#     ax12.set_ylim([-5, 105])
#     ax12.set_title('Sensitivity (%)', y=1.02)
#     ax12.legend(loc='lower right')
#
#     ax21 = fig_dur.add_subplot(223)
#     sns.pointplot(x='duration', y='precision', hue='spikesorter', data=dset_filt, ax=ax21, markers=['^', 'o', 'd'],
#                   linestyles=['--', '-.', '-'])
#     ax21.set_xlabel('')
#     ax21.set_ylabel('')
#     ax21.set_title('Precision (%)', y=1.02)
#     ax21.set_ylim([-5, 105])
#     ax21.set_xlabel('Duration ($s$)', fontsize=20)
#     legend = ax21.legend()
#     legend.remove()
#
#     ax22 = fig_dur.add_subplot(224)
#     sns.pointplot(x='duration', y='misclassification', hue='spikesorter', data=dset_filt, ax=ax22,
#                   linestyles=['--', '-.', '-'], markers=['^', 'o', 'd'], lw=0.5)
#     ax22.set_title('Misclassification (%)', y=1.02)
#     ax22.set_ylabel('')
#     ax22.set_ylim([-5, 40])
#     ax22.set_xlabel('Duration ($s$)', fontsize=20)
#     legend = ax22.legend()
#     legend.remove()
#     mark_subplots([ax11, ax12, ax21, ax22], xpos=-0.2, ypos=1.05, fs=25)
#     simplify_axes([ax11, ax12, ax21, ax22])
#     fig_dur.subplots_adjust(left=0.1, right=0.98, bottom=0.1, top=0.85, hspace=0.4, wspace=0.2)
#
#     print "Time analysis"
#     duration = 10
#     dset_filt = dset[dset['duration']==duration]
#     fig_dur = plt.figure(figsize=(9, 7))
#
#     ax111 = fig_dur.add_subplot(111)
#     sns.boxplot(x='spikesorter', y='time', data=dset_filt, ax=ax111)
#     ax111.set_title('Time (%)', y=1.02)
#     legend = ax11.legend()
#     legend.remove()
#
#     simplify_axes([ax111])
#
#
# plt.ion()
# plt.show()
#
#
