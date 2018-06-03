from __future__ import print_function

import numpy as np
from numpy import linalg as la
import time
import warnings
import matplotlib.pylab as plt
import quantities as pq
from scipy import stats
from scipy.linalg import sqrtm
from scipy.linalg import eigh
from scipy.linalg import LinAlgError
from sklearn.decomposition import PCA

import orICA as ori




if __name__ == '__main__':
    import sys
    from os.path import join
    from scipy import stats
    import MEAutility as MEA
    from tools import *
    import yaml
    from spike_sorting import plot_mixing

    debug=True

    if debug:
        # folder = 'recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_physrot_Neuronexus-32-cut-30_10_' \
        #              '10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_10-05-2018:11:37_3002/'
        # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/SqMEA-10-15um/recording_ica_physrot' \
        #          '_SqMEA-10-15um_20_20.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_22-05-2018:14:45_25'
        folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_' \
                 'physrot_Neuronexus-32-cut-30_15_60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_29-05-2018:16:38_2416'
        block = 1000
        recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)
        electrode_name = info['General']['electrode name']
        fs = info['General']['fs']
        nsec_window = 10
        nsamples_window = int(fs.rescale('Hz').magnitude * nsec_window)
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        templates = np.load(join(folder, 'templates.npy'))
        mixing = np.load(join(folder, 'mixing.npy')).T
        sources = np.load(join(folder, 'sources.npy'))
        adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
        n_sources = sources.shape[0]
        # lambda_val = 0.0001
        lambda_val = 0.995

        ori = ori.onlineORICAss(recordings, fs=fs, forgetfac='cooling', skew_thresh=0.8, lambda_0=lambda_val, verbose=True,
                                block=block, step_size=1, window=5, initial_window=0, detect_trheshold=10,
                                onlineDetection=False)

        # Time analysis

        # SNR evaluation

