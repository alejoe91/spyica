import numpy as np
import os, sys
from os.path import join
import matplotlib.pyplot as plt
import neo
import elephant
import scipy.signal as ss
import scipy.stats as stats
import quantities as pq
import json
import yaml
import time
import h5py
import ipdb

from tools import *
from neuroplot import *
import MEAutility as MEA
from spike_sorting import plot_mixing, plot_templates, templates_weights
import ICA as ica
from orICA import ORICA

root_folder = os.getcwd()




if __name__ == '__main__':

    if '-r' in sys.argv:
        pos = sys.argv.index('-r')
        folder = sys.argv[pos + 1]
    if '-mod' in sys.argv:
        pos = sys.argv.index('-mod')
        mod = sys.argv[pos + 1]
    else:
        mod = 'orica'
    if '-block' in sys.argv:
        pos = sys.argv.index('-block')
        block_size = int(sys.argv[pos + 1])
    else:
        block_size = 500
    if '-ff' in sys.argv:
        pos = sys.argv.index('-ff')
        ff = sys.argv[pos + 1]
    else:
        ff = 'constant'
    if '-mu' in sys.argv:
        pos = sys.argv.index('-mu')
        tstart = sys.argv[pos + 1]
    else:
        tstart = None
    if '-lambda' in sys.argv:
        pos = sys.argv.index('-lambda')
        lambda_val = float(sys.argv[pos + 1])
    else:
        lambda_val = 0.995
    if '-oricamod' in sys.argv:
        pos = sys.argv.index('-oricamod')
        oricamod = sys.argv[pos + 1]
    else:
        oricamod = 'original'
    if '-npass' in sys.argv:
        pos = sys.argv.index('-npass')
        npass = int(sys.argv[pos + 1])
    else:
        npass = 1

    if len(sys.argv) == 1:
        print 'Evaluate ICA for spike sorting:\n   -r recording folder\n   -mod orica-ica\n   block block size' \
              '\n   -ff constant-cooling\n   -mu smoothing\n   -lambda lambda_0' \
              '\n   -oricamod  original - A - W - A_block - W_block\n   -npass numpass'
        raise Exception('Indicate recording folder -r')

    else:
        recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)

        electrode_name = info['General']['electrode name']
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        templates = np.load(join(folder, 'templates.npy'))
        mixing = np.load(join(folder, 'mixing.npy'))

    orica_type = oricamod # original - W - A -  W_block - A_block
    if ff == 'constant':
        lambda_val = 1. / recordings.shape[1] # 0.995
    else:
        lambda_val = lambda_val

    adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)

    t_start = time.time()
    if mod == 'orica':
        if orica_type == 'original':
            ori = ORICA(recordings, sphering='offline', forgetfac=ff, lambda_0=lambda_val,
                        mu=0, verbose=True, numpass=npass, block_white=block_size, block_ica=block_size)
        elif orica_type == 'W':
            ori = ORICA_W(recordings, sphering='offline', forgetfac=ff, lambda_0=lambda_val,
                              mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
        elif orica_type == 'A':
            ori = ORICA_A(recordings, sphering='offline', forgetfac=ff, lambda_0=lambda_val,
                        mu=0, verbose=True, numpass=npass, block_white=block_size, block_ica=block_size)
        elif orica_type == 'W_block':
            ori = ORICA_W_block(recordings, sphering='offline', forgetfac=ff, lambda_0=lambda_val,
                        mu=0, verbose=True, numpass=npass, block_white=block_size, block_ica=block_size)
        elif orica_type == 'A_block':
            ori = ORICA_A_block(recordings, sphering='offline', forgetfac=ff, lambda_0=lambda_val,
                        mu=0, verbose=True, numpass=npass, block_white=block_size, block_ica=block_size)
        else:
            raise Exception('ORICA type not understood')

        y = ori.y
        w = ori.unmixing
        m = ori.sphere
        a = ori.mixing

    elif mod == 'ica':
        y, a, w = ica.instICA(recordings)

    proc_time = time.time() - t_start
    print 'Processing time: ', proc_time

    # Skewness
    skew_thresh = 0.1
    sk = stats.skew(y, axis=1)
    high_sk = np.where(np.abs(sk) >= skew_thresh)

    # Kurtosis
    ku_thresh = 0.8
    ku = stats.kurtosis(y, axis=1)
    high_ku = np.where(np.abs(ku) >= ku_thresh)

    a_t = np.zeros(a.shape)
    for i, a_ in enumerate(a):
        if np.abs(np.min(a_)) > np.abs(np.max(a_)):
            a_t[i] = -a_ / np.max(np.abs(a_))
        else:
            a_t[i] = a_ / np.max(np.abs(a_))

    # Smoothness
    smooth = np.zeros(a.shape[0])
    for i in range(len(smooth)):
       smooth[i] = (np.mean([1. / len(adj) * np.sum(a_t[i, j] - a_t[i, adj]) ** 2
                                             for j, adj in enumerate(adj_graph)]))


    print 'High skewness: ', np.abs(sk[high_sk])
    print 'Average high skewness: ', np.mean(np.abs(sk[high_sk]))
    print 'Number high skewness: ', len(sk[high_sk])

    print 'High kurtosis: ', ku[high_ku]
    print 'Average high kurtosis: ', np.mean(ku[high_ku])
    print 'Number high kurtosis: ', len(ku[high_ku])

    print 'Smoothing: ', smooth[high_sk]
    print 'Average smoothing: ', np.mean(smooth[high_sk])

    plot_mixing(mixing, mea_pos, mea_dim)
    plot_mixing(a_t[high_sk], mea_pos, mea_dim)

    PI, C = evaluate_PI(a_t[high_sk], mixing)

    print 'Performance index: ', PI

    # f = plot_mixing(a[high_sk], mea_pos, mea_dim)
    # f.suptitle('ORICA ' + orica_type + ' ' + str(block_size))
    # plt.figure();
    # plt.plot(y[high_sk].T)
    # plt.title('ORICA ' + orica_type + ' ' + str(block_size))
