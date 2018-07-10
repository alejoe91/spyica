'''
Evaluates the spike sprting performance between 2 spike trains by:
- matching spike trains using correlations
- computing accuracy, precision, sensitivity, misclasification rate, false discovery rate
- computing the confusion matrix on spike counts for matched and unmatched spikes
'''
import numpy as np
import os, sys
from os.path import join
import matplotlib.pyplot as plt
import neo
import elephant
import scipy.signal as ss
import quantities as pq
import json
import yaml
import time
import ipdb
import MEAutility as MEA

from tools import *
from neuroplot import *

root_folder = os.getcwd()


if __name__ == '__main__':
    '''
        COMMAND-LINE 

    '''

    if '-gtst' in sys.argv:
        pos = sys.argv.index('-gtst')
        gtst_path = sys.argv[pos + 1]
    if '-sst' in sys.argv:
        pos = sys.argv.index('-sst')
        sst_path = sys.argv[pos + 1]
    if '-nosave' in sys.argv:
        save = False
    else:
        save = True
    if '-noplot' in sys.argv:
        plot_fig = False
    else:
        plot_fig = True

    if len(sys.argv) == 1:
        print 'Arguments: \n   -gtst ground truth spike trains npy path' \
              '\n   -sst sorted spike trains npy path' \
              '\n   -nosave do not save results'
    elif '-gtst' not in sys.argv or '-sst' not in sys.argv:
        raise Exception('Provide path to GTST and SST npy files')

    gtst = np.load(gtst_path)
    sst = np.load(sst_path)

    if 'spiketrains.npy' in gtst_path:
        rec_folder = os.path.dirname(gtst_path)
    elif 'sst.npy' in gtst_path:
        rec_folder = os.path.dirname(os.path.dirname(os.path.dirname(gtst_path)))

    if 'orica-online' in sst_path:
        orica_folder = os.path.dirname(os.path.dirname(sst_path))
        with open(join(orica_folder, 'orica_params.yaml'), 'r') as f:
            orica_par = yaml.load(f)

        t_start = (orica_par['pca_window'] + orica_par['ica_window']) * pq.s
    else:
        t_start = 0 * pq.s

    with open(join(rec_folder, 'rec_info.yaml'), 'r') as f:
        info = yaml.load(f)
    electrode_name = info['General']['electrode name']

    mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name, sortlist=None)

    print 'Matching spike trains'
    gtst_red = [gt.time_slice(t_start=t_start, t_stop=gtst[0].t_stop) for gt in gtst]
    counts, pairs = evaluate_spiketrains(gtst_red, sst, t_jitt = 3*pq.ms)

    print 'PAIRS: '
    print pairs

    print 'Computing performance'
    performance = compute_performance(counts)

    print 'Calculating confusion matrix'
    conf = confusion_matrix(gtst, sst, pairs[:,1])

    print 'Matching only identified spike trains'
    pairs_gt_red = pairs[:, 0][np.where(pairs[:, 0]!=-1)]
    gtst_id = np.array(gtst_red)[pairs_gt_red]
    counts_id, pairs_id = evaluate_spiketrains(gtst_id, sst, t_jitt = 3*pq.ms)

    print 'Computing performance on only identified spike trains'
    performance_id = compute_performance(counts_id)

    print 'Calculating confusion matrix'
    conf_id = confusion_matrix(gtst_id, sst, pairs_id[:, 1])

    if plot_fig:
        ax1, ax2 = plot_matched_raster(gtst_red, sst, pairs)

    # save results
    # if not os.path.isdir(join(self.yass_folder, 'results')):
    #     os.makedirs(join(self.yass_folder, 'results'))
    # np.save(join(self.yass_folder, 'results', 'counts'), self.counts)
    # np.save(join(self.yass_folder, 'results', 'performance'), self.performance)
    # np.save(join(self.yass_folder, 'results', 'time'), self.processing_time)



