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
root = './'
folder = 'recordings/convolution/gtica'
probe=['SqMEA', 'Neuronexus']
avail_probes = os.listdir(folder)

### ORICA ALGORITHM ###
dimred_analysis = False
sorting_analysis = True

with open('results_online.csv', 'r') as f:
    results_df = pd.read_csv(f)

noise = [5.]
probe=['SqMEA-15-10um']
fig_size = (8, 15)
fs_title = 25
fs_labels = 20
fs_legend = 15

compare_with='CC_source'

colors=plt.rcParams['axes.color_cycle']

if dimred_analysis:

    df_dim = results_df
    df_dim = df_dim[df_dim.noise.isin(noise)]
    df_dim.M = df_dim.M.astype('int')

    Ms = np.unique(df_dim.M)

    fig_dim = plt.figure(figsize=fig_size)
    ax_dim_10 = fig_dim.add_subplot(311)
    ax_time_10 = ax_dim_10.twinx()
    ax_dim_20 = fig_dim.add_subplot(312)
    ax_time_20 = ax_dim_20.twinx()
    ax_dim_30 = fig_dim.add_subplot(313)
    ax_time_30 = ax_dim_30.twinx()

    df_10 = df_dim[df_dim.ncells.isin([10])]
    df_20 = df_dim[df_dim.ncells.isin([20])]
    df_30 = df_dim[df_dim.ncells.isin([30])]

    l1, = ax_dim_10.plot(df_10.M, df_10.C_gt, '--', marker='o', color=colors[0], label='$C_{gt}$', lw=2)
    l2, = ax_dim_10.plot(df_10.M, df_10.C_id, '-.', marker='d', color=colors[1], label='$C_{id}$', lw=2)
    l3, = ax_time_10.plot(df_10.M, df_10.time, '-', marker='.', color=colors[2], label='time', lw=2)
    lines = [l1, l2]
    ax_dim_10.legend(lines, [l.get_label() for l in lines], fontsize=fs_legend, loc='lower right')

    ax_time_10.spines["right"].set_edgecolor(l3.get_color())
    ax_time_10.tick_params(axis='y', colors=l3.get_color())

    ax_dim_10.set_xticks(Ms)
    ax_dim_10.set_xlabel('')
    ax_dim_10.set_ylabel('correlation', fontsize=fs_labels)
    ax_dim_10.set_ylim([0, 1])
    ax_time_10.set_ylim([0, 220])
    ax_time_10.set_ylabel('time (s)', fontsize=fs_labels, color=l3.get_color())
    ax_dim_10.spines['top'].set_visible(False)
    ax_time_10.spines['top'].set_visible(False)
    ax_dim_10.set_title('10 cells', fontsize=fs_title)

    l1, = ax_dim_20.plot(df_20.M, df_20.C_gt, '--', marker='o', color=colors[0], label='$C_{gt}$', lw=2)
    l2, = ax_dim_20.plot(df_20.M, df_20.C_id, '-.', marker='d', color=colors[1], label='$C_{id}$', lw=2)
    l3, = ax_time_20.plot(df_20.M, df_20.time, '-', marker='.', color=colors[2], label='time', lw=2)
    lines = [l1, l2]
    ax_dim_20.legend(lines, [l.get_label() for l in lines], fontsize=fs_legend)

    ax_time_20.spines["right"].set_edgecolor(l3.get_color())
    ax_time_20.tick_params(axis='y', colors=l3.get_color())

    ax_dim_20.set_xticks(Ms)
    ax_dim_20.set_xlabel('')
    ax_dim_20.set_ylabel('correlation', fontsize=fs_labels)
    ax_dim_20.set_ylim([0, 1])
    ax_time_20.set_ylim([0, 220])
    ax_time_20.set_ylabel('time (s)', fontsize=fs_labels, color=l3.get_color())
    ax_dim_20.spines['top'].set_visible(False)
    ax_time_20.spines['top'].set_visible(False)
    ax_dim_20.set_title('20 cells', fontsize=fs_title)


    l1, = ax_dim_30.plot(df_30.M, df_30.C_gt, '--', marker='o', color=colors[0], label='$C_{gt}$', lw=2)
    l2, = ax_dim_30.plot(df_30.M, df_30.C_id, '-.', marker='d', color=colors[1], label='$C_{id}$', lw=2)
    l3, = ax_time_30.plot(df_30.M, df_30.time, '-', marker='.', color=colors[2], label='time', lw=2)
    lines = [l1, l2]
    ax_dim_30.legend(lines, [l.get_label() for l in lines], fontsize=fs_legend)

    ax_time_30.spines["right"].set_edgecolor(l3.get_color())
    ax_time_30.tick_params(axis='y', colors=l3.get_color())

    ax_dim_30.set_xticks(Ms)
    ax_dim_30.set_xlabel('M - reduced dimension', fontsize=fs_labels)
    ax_dim_30.set_ylabel('correlation', fontsize=fs_labels)
    ax_dim_30.set_ylim([0, 1])
    ax_time_30.set_ylim([0, 220])
    ax_time_30.set_ylabel('time (s)', fontsize=fs_labels, color=l3.get_color())
    ax_dim_30.spines['top'].set_visible(False)
    ax_time_30.spines['top'].set_visible(False)
    ax_dim_30.set_title('30 cells', fontsize=fs_title)


    ax_time_10.axhline(120, color=l3.get_color(), alpha=0.8, lw=1.5, ls='--')
    ax_time_10.text(80, 100, 'real-time', fontsize=fs_legend, color=l3.get_color())
    ax_time_20.axhline(120, color=l3.get_color(), alpha=0.8, lw=1.5, ls='--')
    ax_time_20.text(80, 100, 'real-time', fontsize=fs_legend, color=l3.get_color())
    ax_time_30.axhline(120, color=l3.get_color(), alpha=0.8, lw=1.5, ls='--')
    ax_time_30.text(80, 100, 'real-time', fontsize=fs_legend, color=l3.get_color())

    fig_dim.subplots_adjust(left=0.1, right=0.9, bottom=0.08, top=0.95, hspace=0.4)
    mark_subplots([ax_dim_10, ax_dim_20, ax_dim_30], xpos=-0.08, ypos=1.05, fs=35)

if sorting_analysis:
    probe = ['SqMEA-10-15um']
    avail_probes = os.listdir(folder)

    all_recordings = []

    for p in probe:
        all_recordings.extend([join(root, folder, p, f) for f in os.listdir(join(root, folder, p))])

    ncell_vec = []
    acc_vec, sens_vec, prec_vec, false_vec, miss_vec, misclass_vec = [], [], [], [], [], []

    for rec in all_recordings:
        print rec
        with open(join(rec, 'rec_info.yaml'), 'r') as f:
            info = yaml.load(f)
        if os.path.isdir(join(rec, 'results')):
            perf = np.load(join(rec, 'results', 'performance.npy')).item()

            ncells = int(info['Spikegen']['n_cells'])
            noise_lev = float(info['Noise']['noise_level'])

            ncell_vec.append(ncells)
            acc_vec.append(np.round(perf['accuracy'],1))
            sens_vec.append(np.round(perf['sensitivity'],1))
            prec_vec.append(np.round(perf['precision'],1))
            false_vec.append(np.round(perf['false_disc_rate'],1))
            miss_vec.append(np.round(perf['miss_rate'],1))
            misclass_vec.append(np.round(perf['tot_cl'],1))


    sorting_df = pd.DataFrame(data={'ncells': ncell_vec, 'accuracy': acc_vec, 'sensitivity': sens_vec,
                                    'precision': prec_vec, 'miss_rate': miss_vec, 'false_rate': false_vec,
                                    'misclassification': misclass_vec})
    sorting_df = sorting_df.sort_values('ncells')

    print(sorting_df.to_latex(columns=['ncells', 'accuracy', 'sensitivity', 'precision',
                                       'misclassification'], index=False))
