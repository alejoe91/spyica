'''

'''


import numpy as np
import os, sys
from os.path import join
import matplotlib.pyplot as plt
import elephant
import scipy.signal as ss
import scipy.stats as stats
import quantities as pq
import json
import yaml
import time

import spiketrain_generator as stg
from tools import *
from plot_spikeMEA_old import *
import ICA as ica

root_folder = os.getcwd()

plt.ion()
plt.show()
plot_source = True
plot_cc = True


# TODO extract one spike per source (amplitude clustering)

class SpikeSorter:
    def __init__(self, save=False, rec_folder=None, alg=None):
        self.rec_folder = rec_folder
        self.model = alg
        self.corr_trhesh = 0.3
        self.skew_thresh = 1


        self.true_st = np.load(join(self.rec_folder, 'spiketrains.npy'))
        self.recordings = np.load(join(self.rec_folder, 'recordings.npy'))
        rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(self.rec_folder, rec_info), 'r') as f:
            self.info = yaml.load(f)

        self.electrode_name = self.info['General']['electrode name']
        self.fs = self.info['General']['fs']

        # load MEA info
        with open(join(root_folder, 'electrodes', self.electrode_name + '.json')) as meafile:
            elinfo = json.load(meafile)

        x_plane = 0.
        pos = MEA.get_elcoords(x_plane, **elinfo)

        x_plane = 0.
        self.mea_pos = MEA.get_elcoords(x_plane, **elinfo)
        self.mea_pitch = elinfo['pitch']
        self.mea_dim = elinfo['dim']
        mea_shape = elinfo['shape']

        self.ica = False
        self.cica = False
        self.gfica = False
        self.sfa = False
        if alg == 'all':
            self.ica = True
            self.cica = True
            self.gfica = True
            self.sfa = True
        else:
            alg_split = alg.lower().split('-')
            if 'ica' in alg_split:
                self.ica = True
            if 'cica' in alg_split:
                self.cica = True
            if 'gfica' in alg_split:
                self.gfica = True
            if 'sfa' in alg_split:
                self.sfa = True

        if self.ica:
            print 'Applying instantaneous ICA'
            t_start = time.time()
            self.s_ica, A_ica, W_ica = ica.instICA(self.recordings)
            print 'Elapsed time: ', time.time() - t_start

            if plot_source:
                plot_mea_recording(self.s_ica, self.mea_pos, self.mea_dim, color='r')

            # clean sources based on skewness and correlation
            spike_sources, source_idx = clean_sources(self.s_ica, corr_thresh=self.corr_trhesh,
                                                      skew_thresh=self.skew_thresh)
            if plot_source:
                plt.figure()
                plt.plot(np.transpose(spike_sources))

            n_sources = spike_sources.shape[0]
            n_new_sources = 0
            iter = 1

            while n_new_sources != n_sources:
                print 'Iter', iter, ' n_sources = ', n_sources, ' n_new_sources = ', n_new_sources
                s_new, A_new, W_new = ica.instICA(spike_sources)
                spike_sources_new, sources_idx_new = clean_sources(s_new, corr_thresh=self.corr_trhesh,
                                                                   skew_thresh=self.skew_thresh)
                n_new_sources = spike_sources_new.shape[0]
                iter += 1

            if iter > 1:
                spike_sources = spike_sources_new

            self.cleaned_sources = spike_sources

            detect spikes and align
            self.spike_times, self.spike_amps, self.spike_waveforms = detect_and_align(spike_sources, self.fs)

            cc_matr = evaluate_spiketrains(self.true_st, self.spike_times)
            if plot_cc:
                plt.figure()
                plt.imshow(cc_matr)


        if self.cica:
            print 'Applying convolutive embedded ICA'
            t_start = time.time()
            s_c, A_c, W_c = ica.cICAemb(self.recordings)
            print 'Elapsed time: ', time.time() - t_start

        if self.gfica:
            print 'Applying gradient-flow ICA'
            t_start = time.time()
            s_gf, A_gf, W_gf = ica.gFICA(self.recordings, mea_dim)
            s_gf_int = integrate_sources(s_gf)
            print 'Elapsed time: ', time.time() - t_start

            gf_mea = np.reshape(np.reshape(mea_pos, (mea_dim[0], mea_dim[1], mea_pos.shape[1]))[:-1, :-1],
                                ((mea_dim[0]-1)*(mea_dim[1]-1), mea_pos.shape[1]))
            gf_dim = (mea_dim[0]-1, mea_dim[1]-1)

        if self.sfa:
            pass


if __name__ == '__main__':
    '''
        COMMAND-LINE 
        -f filename
        -fs sampling frequency
        -ncells number of cells
        -pexc proportion of exc cells
        -bx x boundaries
        -minamp minimum amplitude
        -noise uncorrelated-correlated
        -noiselev level of rms noise in uV
        -dur duration
        -fexc freq exc neurons
        -finh freq inh neurons
        -nofilter if filter or not
        -over overlapping spike threshold (0.6)
        -sync added synchorny rate'
    '''

    if '-r' in sys.argv:
        pos = sys.argv.index('-r')
        rec_folder = sys.argv[pos + 1]
    if '-mod' in sys.argv:
        pos = sys.argv.index('-mod')
        mod = sys.argv[pos + 1]
    else:
        mod = 'ICA'
    # if '-dur' in sys.argv:
    #     pos = sys.argv.index('-dur')
    #     dur = sys.argv[pos + 1]
    # else:
    #     dur = 5
    # if '-ncells' in sys.argv:
    #     pos = sys.argv.index('-ncells')
    #     ncells = sys.argv[pos + 1]
    # else:
    #     ncells = 30
    # if '-pexc' in sys.argv:
    #     pos = sys.argv.index('-pexc')
    #     pexc = sys.argv[pos + 1]
    # else:
    #     pexc = 0.7
    # if '-fexc' in sys.argv:
    #     pos = sys.argv.index('-fexc')
    #     fexc = sys.argv[pos + 1]
    # else:
    #     fexc = 5
    # if '-finh' in sys.argv:
    #     pos = sys.argv.index('-finh')
    #     finh = sys.argv[pos + 1]
    # else:
    #     finh = 15
    # if '-bx' in sys.argv:
    #     pos = sys.argv.index('-bx')
    #     bx = sys.argv[pos + 1]
    # else:
    #     bx = [20, 60]
    # if '-minamp' in sys.argv:
    #     pos = sys.argv.index('-minamp')
    #     minamp = sys.argv[pos + 1]
    # else:
    #     minamp = 50
    # if '-over' in sys.argv:
    #     pos = sys.argv.index('-over')
    #     over = sys.argv[pos + 1]
    # else:
    #     over = 0.6
    # if '-sync' in sys.argv:
    #     pos = sys.argv.index('-sync')
    #     sync = sys.argv[pos + 1]
    # else:
    #     sync = 0.5
    # if '-noise' in sys.argv:
    #     pos = sys.argv.index('-noise')
    #     noise = sys.argv[pos + 1]
    # else:
    #     noise = 'uncorrelated'
    # if '-noiselev' in sys.argv:
    #     pos = sys.argv.index('-noiselev')
    #     noiselev = sys.argv[pos + 1]
    # else:
    #     noiselev = 2.6
    # if '-nofilter' in sys.argv:
    #     filter = False
    # else:
    #     filter = True
    # if '-nomod' in sys.argv:
    #     modulation = False
    # else:
    #     modulation = True
    if len(sys.argv) == 1:
        print 'Arguments: \n   -r recording filename\n   -mod ICA - cICA - gfICA - SFA'
              # '   -ncells number of cells\n' \
              # '   -pexc proportion of exc cells\n   -bx x boundaries [xmin,xmax]\n   -minamp minimum amplitude\n' \
              # '   -noise uncorrelated-correlated\n   -noiselev level of rms noise in uV\n   -dur duration\n' \
              # '   -fexc freq exc neurons\n   -finh freq inh neurons\n   -nofilter if filter or not\n' \
              # '   -over overlapping spike threshold (0.6)\n   -sync added synchorny rate\n' \
              # '   -nomod no spike amp modulation'
    elif '-r' not in sys.argv:
        raise AttributeError('Provide model folder for data')
    else:
        sps = SpikeSorter(save=True, rec_folder=rec_folder, alg=mod)

