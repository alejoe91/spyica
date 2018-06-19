'''

'''
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
import ICA as ica
import smoothICA as sICA
import orICA

root_folder = os.getcwd()

#TODO iterative approach

class SpikeSorter:
    def __init__(self, save=False, rec_folder=None, alg=None, lag=None, gfmode=None, duration=None,
                 tstart=None, tstop=None, run_ss=None, plot_figures=True, merge_spikes=False, mu=0, eta=0,
                 npass=1, block=1000, feat='pca', clust='mog', keepall=True, ndim=None):
        '''

        Parameters
        ----------
        save
        rec_folder
        alg
        lag
        gfmode
        duration
        tstart
        tstop
        run_ss
        plot_figures
        merge_spikes
        mu
        eta
        npass
        block
        feat
        clust
        keepall
        '''
        self.rec_folder = rec_folder
        self.rec_name = os.path.split(rec_folder)[-1]
        if self.rec_name == '':
            split = os.path.split(rec_folder)[0]
            self.rec_name = os.path.split(split)[-1]

        print self.rec_name

        self.iterations = 2

        self.plot_figures = plot_figures
        if self.plot_figures:
            plt.ion()
            plt.show()

        self.duration = duration
        self.tstart = tstart
        self.tstop = tstop
        self.run_ss = run_ss
        self.merge_spikes = merge_spikes

        self.model = alg
        self.kurt_thresh = 1
        self.skew_thresh = 0.1
        self.root = os.getcwd()
        sys.path.append(self.root)

        self.clustering = clust
        self.threshold = 5
        self.keep = keepall
        self.feat = feat
        self.npass = npass
        self.block = block
        self.ndim = ndim

        self.minimum_spikes_per_cluster = 15

        if self.rec_name.startswith('recording'):
            self.gtst = np.load(join(self.rec_folder, 'spiketrains.npy'))
            self.recordings = np.load(join(self.rec_folder, 'recordings.npy')).astype('int16')
            self.templates = np.load(join(self.rec_folder, 'templates.npy'))
            self.templates_cat = np.load(join(self.rec_folder, 'templates_cat.npy'))
            self.templates_loc = np.load(join(self.rec_folder, 'templates_loc.npy'))

            rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
            with open(join(self.rec_folder, rec_info), 'r') as f:
                self.info = yaml.load(f)

            self.electrode_name = self.info['General']['electrode name']
            self.fs = self.info['General']['fs']
            self.times = (range(self.recordings.shape[1]) / self.fs).rescale('s')

            self.overlapping = self.info['Synchrony']['overlap_pairs']
            # if 'overlap' not in self.gtst[0].annotations.keys():
            #     print 'Finding overlapping spikes'
            #     annotate_overlapping(self.gtst, overlapping_pairs=self.overlapping)

            # load MEA info
            elinfo = MEA.return_mea_info(self.electrode_name)

            x_plane = 0.
            pos = MEA.get_elcoords(x_plane, **elinfo)

            x_plane = 0.
            self.mea_pos = MEA.get_elcoords(x_plane, **elinfo)
            self.mea_pitch = elinfo['pitch']
            self.mea_dim = elinfo['dim']
            mea_shape = elinfo['shape']

        elif self.rec_name.startswith('savedata'):
            f = h5py.File(join(self.rec_folder, 'ViSAPy_filterstep_1.h5'))
            self.recordings = np.transpose(f['data']) * 1000 # in uV
            self.fs = f['srate'].value * pq.Hz
            self.elec = f['electrode']
            self.times = range(self.recordings.shape[1]) / self.fs
            self.electrode_name = 'visapy'

            mea_pos = []

            # electrodes are in the x-z
            for x, y, z in zip(self.elec['x'].value, self.elec['y'].value, self.elec['z'].value):
                mea_pos.append([y, x, z])

            self.mea_pos = np.array(mea_pos)
            self.mea_dim = [2, 15]
            self.mea_pitch = [22., 22.]

            # read gtst
            with open(join(self.rec_folder, 'ViSAPy_ground_truth.gdf')) as f:
                lines = f.readlines()
            parsed = np.array([l.split() for l in lines], dtype=int)

            spike_ids = parsed[:, 0]
            spike_times = self.times[parsed[:, 1]]

            self.gtst = []
            # build gtst
            for id in np.unique(spike_ids):
                self.gtst.append(neo.SpikeTrain(spike_times[np.where(spike_ids==id)],
                                 t_start=self.times[0], t_stop=self.times[-1]))

        self.max_idx = self.min_idx = None
        if self.tstart or self.tstop or self.duration:
            if self.tstart is not None:
                self.tstart = float(self.tstart) * pq.s
                self.min_idx = np.where(self.times > self.tstart)[0][0]
            else:
                self.min_idx = 0
                self.tstart = 0 * pq.s

            if self.tstop is not None:
                self.tstop = int(self.tstop) * pq.s
                self.max_idx = np.where(self.times < self.tstop)[0][-1]
                self.duration = self.tstop - self.tstart

            if self.duration is not None:
                self.duration = int(self.duration)*pq.s
                if self.min_idx is None:
                    self.min_idx = 0
                    self.tstart = 0*pq.s
                if self.max_idx is None:
                    nsamples = int(self.duration*self.fs.rescale('Hz'))
                    self.max_idx = self.min_idx + nsamples
                    self.tstop = self.tstart + self.duration

                self.times = self.times[self.min_idx:self.max_idx] - self.tstart
                self.recordings = self.recordings[:, self.min_idx:self.max_idx]

                sliced_gt = []
                for gt in self.gtst:
                    st = gt.time_slice(t_start=self.tstart, t_stop=self.tstop)
                    st -= self.tstart
                    st.t_start = 0*pq.s
                    st.t_stop = self.tstop-self.tstart
                    if len(st > self.minimum_spikes_per_cluster):
                        sliced_gt.append(st)
                self.gtst = sliced_gt
            else:
                raise Exception('set at least tstop or duration')

        self.ica = False
        self.orica = False
        self.orica_online = False
        self.threshold_online = False
        self.smooth = False
        self.cica = False
        self.corica = False
        self.gfica = False
        self.klusta = False
        self.kilo = False
        self.mountain = False
        self.circus = False
        self.yass = False

        if alg == 'all':
            self.ica = True
            self.smooth = True
            self.cica = True
            self.gfica = True
            self.klusta = True
        else:
            alg_split = alg.lower().split('-')
            if 'ica' in alg_split:
                self.ica = True
            if 'orica' in alg_split:
                self.orica = True
            if 'oricaonline' in alg_split:
                self.orica_online = True
            if 'threshonline' in alg_split:
                self.threshold_online = True
            if 'smooth' in alg_split:
                self.smooth = True
            if 'cica' in alg_split:
                self.cica = True
            if 'corica' in alg_split:
                self.corica = True
            if 'gfica' in alg_split:
                self.gfica = True
            if 'klusta' in alg_split:
                self.klusta = True
            if 'kilosort' in alg_split:
                self.kilo = True
            if 'mountainsort' in alg_split:
                self.mountain = True
            if 'spykingcircus' in alg_split:
                self.circus = True
            if 'yass' in alg_split:
                self.yass = True

        self.lag = lag
        self.gfmode = gfmode

        if self.ica:
            if self.run_ss:
                print 'Applying instantaneous ICA'
                t_start = time.time()
                t_start_proc = time.time()

                if not os.path.isdir(join(self.rec_folder, 'ica')):
                    os.makedirs(join(self.rec_folder, 'ica'))
                self.ica_folder = join(self.rec_folder, 'ica')

                chunk_size = int(2*pq.s * self.fs.rescale('Hz'))
                n_chunks = 1
                self.s_ica, self.A_ica, self.W_ica = ica.instICA(self.recordings, n_comp=self.ndim,
                                                                 n_chunks=n_chunks, chunk_size=chunk_size)
                print 'ICA Finished. Elapsed time: ', time.time() - t_start, ' sec.'

                # clean sources based on skewness and correlation
                spike_sources, self.source_idx = clean_sources(self.s_ica, kurt_thresh=self.kurt_thresh,
                                                               skew_thresh=self.skew_thresh)

                self.cleaned_sources_ica = spike_sources
                self.cleaned_A_ica = self.A_ica[self.source_idx]
                self.cleaned_W_ica = self.W_ica[self.source_idx]

                print 'Number of cleaned sources: ', self.cleaned_sources_ica.shape[0]

                print 'Clustering Sources with: ', self.clustering

                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.gtst[0].t_start,
                                                            t_stop=self.gtst[0].t_stop, n_std=self.threshold)
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.detected_spikes]

                    self.spike_trains, self.amps, self.nclusters, self.keep, self.score = \
                        cluster_spike_amplitudes(self.detected_spikes, metric='cal',
                                                 alg=self.clustering,
                                                 features=self.feat, keep_all=self.keep)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.spike_trains, sources=spike_sources)

                    self.sst = self.spike_trains_rej

                elif self.clustering=='klusta':
                    self.klusta_folder = join(self.ica_folder, 'klusta')
                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    self.klusta_full_path = join(self.klusta_folder, 'recording')
                    # create prb and prm files
                    prb_path = export_prb_file(self.cleaned_sources_ica.shape[0], 'ica', self.klusta_folder,
                                               geometry=False, graph=False, separate_channels=True)
                    # save binary file
                    filename = join(self.klusta_folder, 'recordings')
                    file_path = save_binary_format(join(self.klusta_folder, 'recordings'), self.cleaned_sources_ica,
                                                   spikesorter='klusta')

                    # set up klusta config file
                    with open(join(self.root, 'spikesorter_files', 'klusta_files',
                                   'config.prm'), 'r') as f:
                        klusta_config = f.readlines()

                    nchan = self.cleaned_sources_ica.shape[0]
                    threshold = self.threshold

                    klusta_config = ''.join(klusta_config).format(
                        filename, prb_path, float(self.fs.rescale('Hz')), nchan, 'float32', threshold
                    )
                    with open(join(self.klusta_folder, 'config.prm'), 'w') as f:
                        f.writelines(klusta_config)

                    print('Running klusta')

                    try:
                        import klusta
                        import klustakwik2
                    except ImportError:
                        raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

                    import subprocess
                    try:
                        subprocess.check_output(['klusta', join(self.klusta_folder, 'config.prm'), '--overwrite'])
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    kwikfile = [f for f in os.listdir(self.klusta_folder) if f.endswith('.kwik')]
                    if len(kwikfile) > 0:
                        kwikfile = join(self.klusta_folder, kwikfile[0])
                        if os.path.exists(kwikfile):
                            kwikio = neo.io.KwikIO(filename=kwikfile, )
                            blk = kwikio.read_block(raw_data_units='uV')
                            self.detected_spikes = blk.segments[0].spiketrains
                    else:
                        raise Excaption('No kwik file!')

                    self.spike_trains_rej, self.independent_spike_idx = \
                        reject_duplicate_spiketrains(self.detected_spikes)
                    self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                    # self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                    self.sst = self.spike_trains

                elif self.clustering == 'mountain':
                    self.mountain_sig = 'rec'
                    sys.path.append(join(self.root, 'spikesorter_files', 'mountainsort_files'))
                    import mlpy
                    from shutil import copyfile

                    if not os.path.isdir(join(self.ica_folder, 'mountain')):
                        os.makedirs(join(self.ica_folder, 'mountain'))
                    self.mountain_folder = join(self.ica_folder, 'mountain')
                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    self.mountain_full_path = join(self.mountain_folder, rec_name)

                    if self.mountain_sig == 'ic':

                        # write data file
                        filename = join(self.mountain_folder, 'raw.mda')
                        mlpy.writemda32(self.cleaned_sources_ica, filename)
                        print 'saving ', filename


                        # # write csv probe file
                        # with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                        #     for pos in self.mea_pos:
                        #         f.write(str(pos[1]))
                        #         f.write(',')
                        #         f.write(str(pos[2]))
                        #         f.write('\n')

                        # write param file
                        detect_threshold = None
                        params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': 0,
                                  'adjacency_radius': 0}
                    elif self.mountain_sig == 'rec':
                        # write data file
                        filename = join(self.mountain_folder, 'raw.mda')
                        self.reconstructed = np.matmul(self.cleaned_A_ica.T, self.cleaned_sources_ica)
                        mlpy.writemda32(self.reconstructed, filename)
                        print 'saving ', filename

                        # write csv probe file
                        with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                            for pos in self.mea_pos:
                                f.write(str(pos[1]))
                                f.write(',')
                                f.write(str(pos[2]))
                                f.write('\n')

                        # write param file
                        detect_threshold = None
                        radius = 50
                        params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': 0,
                                  'adjacency_radius': radius}

                    with open(join(self.mountain_folder, 'params.json'), 'w') as f:
                        json.dump(params, f)

                    # copy mountainsort3.mlp
                    copyfile(join(self.root, 'spikesorter_files', 'mountainsort_files', 'mountainsort3.mlp'),
                             join(self.mountain_folder, 'mountainsort3.mlp'))

                    print('Running MountainSort')
                    self.curate = True

                    import subprocess
                    os.chdir(self.mountain_folder)
                    try:
                        if self.mountain_sig == 'ic':
                            if self.curate:
                                subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                         '--firings_out=firings.mda',
                                                         '--_params=params.json',
                                                         '--curate=true'])
                            else:
                                subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                         '--firings_out=firings.mda',
                                                         '--_params=params.json'])
                        elif self.mountain_sig == 'rec':
                            if self.curate:
                                subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                         '--geom=geom.csv', '--firings_out=firings.mda',
                                                         '--_params=params.json',
                                                         '--curate=true'])
                            else:
                                subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                         '--geom=geom.csv', '--firings_out=firings.mda',
                                                         '--_params=params.json'])
                                
                        self.processing_time = time.time() - t_start_proc
                        print 'Elapsed time: ', self.processing_time
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)

                    print 'Parsing output files...'
                    self.firings = mlpy.readmda(join(self.mountain_folder, 'firings.mda'))
                    self.spike_trains = []
                    clust_id, n_counts = np.unique(self.firings[2], return_counts=True)
                    self.ml_times = self.times[self.firings[1].astype(int)]

                    self.counts = 0
                    for clust, count in zip(clust_id, n_counts):
                        if count > self.minimum_spikes_per_cluster:
                            idx = np.where(self.firings[2] == clust)[0]
                            assert len(np.unique(self.firings[0, idx]) == 1)
                            self.counts += len(idx)
                            spike_times = self.ml_times[idx]
                            spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                            spiketrain.annotate(ica_source=int(np.unique(self.firings[0, idx])) - 1)
                            self.spike_trains.append(spiketrain)
                    # TODO extract waveforms
                    self.sst = self.spike_trains

                else:
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.gtst[0].t_start,
                                                            t_stop=self.gtst[0].t_stop)
                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.detected_spikes)
                    self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains_rej]
                    self.sst = self.spike_trains_rej

                print 'Number of spike trains after clustering: ', len(self.spike_trains)
                print 'Number of spike trains after duplicate rejection: ', len(self.sst)

                if 'ica_source' in self.sst[0].annotations.keys():
                    self.independent_spike_idx = [s.annotations['ica_source'] for s in self.sst]

                self.ica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                self.A_spike_sources = self.cleaned_A_ica[self.independent_spike_idx]
                self.W_spike_sources = self.cleaned_W_ica[self.independent_spike_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.ica_spike_sources, self.recordings, self.times, self.fs)

                self.processing_time = time.time() - t_start_proc
                print 'Elapsed time: ', self.processing_time

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('ICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.ica_folder, 'results')):
                    os.makedirs(join(self.ica_folder, 'results'))
                np.save(join(self.ica_folder, 'results', 'counts'), self.counts)
                np.save(join(self.ica_folder, 'results', 'performance'), self.performance)
                np.save(join(self.ica_folder, 'results', 'time'), self.processing_time)

        if self.orica:
            # TODO: quantify smoothing with and without mu
            if self.run_ss:
                print 'Applying Online Recursive ICA'
                t_start = time.time()
                t_start_proc = time.time()
                if not os.path.isdir(join(self.rec_folder, 'orica')):
                    os.makedirs(join(self.rec_folder, 'orica'))
                self.orica_folder = join(self.rec_folder, 'orica')

                chunk_size = int(2 * pq.s * self.fs.rescale('Hz'))
                n_chunks = 1
                adj_graph = extract_adjacency(self.mea_pos, np.max(self.mea_pitch) + 5)
                self.s_orica, self.A_orica, self.W_orica = orICA.instICA(self.recordings, n_comp=self.ndim,
                                                                         n_chunks=n_chunks, chunk_size=chunk_size,
                                                                         mu=mu, adjacency_graph=adj_graph,
                                                                         numpass=npass, block_size=block, mode='original')

                # self.avg_smoothing = []
                # for i in range(self.recordings.shape[0]):
                #     self.avg_smoothing.append(np.mean([1. / len(adj) * np.sum(self.W_orica[i, j] - self.W_orica[i, adj])**2
                #                                               for j, adj in enumerate(adj_graph)]))

                # clean sources based on skewness and correlation
                spike_sources, self.source_idx = clean_sources(self.s_orica,
                                                               kurt_thresh=self.kurt_thresh,
                                                               skew_thresh=self.skew_thresh)

                self.cleaned_sources_orica = spike_sources
                self.cleaned_A_orica = self.A_orica[self.source_idx]
                self.cleaned_W_orica = self.W_orica[self.source_idx]
                # self.cleaned_smoothing = np.array(self.avg_smoothing)[self.source_idx]
                print 'Number of cleaned sources: ', self.cleaned_sources_orica.shape[0]


                print 'Clustering Sources with: ', self.clustering
                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.gtst[0].t_start,
                                                            t_stop=self.gtst[0].t_stop, n_std=self.threshold)
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.detected_spikes]

                    self.spike_trains, self.amps, self.nclusters, self.keep, self.score = \
                        cluster_spike_amplitudes(self.detected_spikes, metric='cal',
                                                 alg=self.clustering,
                                                 features=self.feat, keep_all=self.keep)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.spike_trains)
                    self.sst = self.spike_trains_rej

                elif self.clustering=='klusta':

                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    if not os.path.isdir(join(self.orica_folder, 'klusta')):
                        os.makedirs(join(self.orica_folder, 'klusta'))
                    self.klusta_folder = join(self.orica_folder, 'klusta')
                    self.klusta_full_path = join(self.klusta_folder, 'recording')
                    # create prb and prm files
                    prb_path = export_prb_file(self.cleaned_sources_orica.shape[0], 'ica', self.klusta_folder,
                                               geometry=False, graph=False, separate_channels=True)
                    # save binary file
                    filename = join(self.klusta_folder, 'recordings')
                    file_path = save_binary_format(join(self.klusta_folder, 'recordings'), self.cleaned_sources_orica,
                                                   spikesorter='klusta')

                    # set up klusta config file
                    with open(join(self.root, 'spikesorter_files', 'klusta_files',
                                   'config.prm'), 'r') as f:
                        klusta_config = f.readlines()

                    nchan = self.cleaned_sources_orica.shape[0]
                    threshold = self.threshold

                    klusta_config = ''.join(klusta_config).format(
                        filename, prb_path, float(self.fs.rescale('Hz')), nchan, 'int16', threshold
                    )
                    with open(join(self.klusta_folder, 'config.prm'), 'w') as f:
                        f.writelines(klusta_config)

                    print('Running klusta')

                    try:
                        import klusta
                        # import klustakwik2
                    except ImportError:
                        raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

                    import subprocess
                    try:
                        subprocess.check_output(['klusta', join(self.klusta_folder, 'config.prm'), '--overwrite'])
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    kwikfile = [f for f in os.listdir(self.klusta_folder) if f.endswith('.kwik')]
                    if len(kwikfile) > 0:
                        kwikfile = join(self.klusta_folder, kwikfile[0])
                        if os.path.exists(kwikfile):
                            kwikio = neo.io.KwikIO(filename=kwikfile, )
                            blk = kwikio.read_block(raw_data_units='uV')
                            detected_spikes = blk.segments[0].spiketrains
                    else:
                        raise Excaption('No kwik file!')

                    # remove extra index in kwik spiketrains
                    self.detected_spikes = []
                    for det in detected_spikes:
                        st = neo.SpikeTrain([t[0].rescale('ms').magnitude for t in det.times]*pq.ms,
                                            t_start=det.t_start, t_stop=det.t_stop)
                        st.annotations = det.annotations
                        self.detected_spikes.append(st)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.detected_spikes)
                    if 'group_id' in self.spike_trains_rej[0].annotations.keys():
                        self.independent_spike_idx = [s.annotations['group_id'] for s in self.spike_trains_rej]

                    self.sst = self.spike_trains_rej

                elif self.clustering=='klusta-back':

                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    if not os.path.isdir(join(self.orica_folder, 'klusta')):
                        os.makedirs(join(self.orica_folder, 'klusta'))
                    self.klusta_folder = join(self.orica_folder, 'klusta')
                    self.klusta_full_path = join(self.klusta_folder, 'recording')
                    # create prb and prm files
                    prb_path = export_prb_file(self.mea_pos.shape[0], self.electrode_name, self.klusta_folder,
                                               pos=self.mea_pos, adj_dist=2 * np.max(self.mea_pitch))
                    # save binary file
                    filename = join(self.klusta_folder, 'recordings')
                    self.reconstructed = np.matmul(self.cleaned_A_orica.T, self.s_orica[self.source_idx])
                    file_path = save_binary_format(join(self.klusta_folder, 'recordings'), self.reconstructed,
                                                   spikesorter='klusta')

                    # set up klusta config file
                    with open(join(self.root, 'spikesorter_files', 'klusta_files',
                                   'config.prm'), 'r') as f:
                        klusta_config = f.readlines()

                    nchan = self.mea_pos.shape[0]
                    threshold = self.threshold

                    klusta_config = ''.join(klusta_config).format(
                        filename, prb_path, float(self.fs.rescale('Hz')), nchan, 'float32', threshold
                    )
                    with open(join(self.klusta_folder, 'config.prm'), 'w') as f:
                        f.writelines(klusta_config)

                    print('Running klusta')

                    try:
                        import klusta
                        # import klustakwik2
                    except ImportError:
                        raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

                    import subprocess
                    try:
                        subprocess.check_output(['klusta', join(self.klusta_folder, 'config.prm'), '--overwrite', '--detect-only'])
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    kwikfile = [f for f in os.listdir(self.klusta_folder) if f.endswith('.kwik')]
                    if len(kwikfile) > 0:
                        kwikfile = join(self.klusta_folder, kwikfile[0])
                        if os.path.exists(kwikfile):
                            kwikio = neo.io.KwikIO(filename=kwikfile, )
                            blk = kwikio.read_block(raw_data_units='uV')
                            detected_spikes = blk.segments[0].spiketrains
                    else:
                        raise Excaption('No kwik file!')

                    # remove extra index in kwik spiketrains
                    self.detected_spikes = []
                    for det in detected_spikes:
                        st = neo.SpikeTrain([t[0].rescale('ms').magnitude for t in det.times]*pq.ms,
                                            t_start=det.t_start, t_stop=det.t_stop)
                        st.annotations = det.annotations
                        self.detected_spikes.append(st)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.detected_spikes)
                    if 'group_id' in self.spike_trains_rej[0].annotations.keys():
                        self.independent_spike_idx = [s.annotations['group_id'] for s in self.spike_trains_rej]

                    self.sst = self.spike_trains_rej

                elif self.clustering == 'mountain':

                    sys.path.append(join(self.root, 'spikesorter_files', 'mountainsort_files'))
                    import mlpy
                    from shutil import copyfile

                    if not os.path.isdir(join(self.orica_folder, 'mountain')):
                        os.makedirs(join(self.orica_folder, 'mountain'))
                    self.mountain_folder = join(self.orica_folder, 'mountain')
                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    self.mountain_full_path = join(self.mountain_folder, rec_name)

                    # write data file
                    filename = join(self.mountain_folder, 'raw.mda')
                    mlpy.writemda32(self.cleaned_sources_orica, filename)
                    print 'saving ', filename

                    # # write csv probe file
                    # with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                    #     for pos in self.mea_pos:
                    #         f.write(str(pos[1]))
                    #         f.write(',')
                    #         f.write(str(pos[2]))
                    #         f.write('\n')

                    # write param file
                    detect_threshold = None
                    params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': -1,
                              'adjacency_radius': 0}

                    with open(join(self.mountain_folder, 'params.json'), 'w') as f:
                        json.dump(params, f)

                    # copy mountainsort3.mlp
                    copyfile(join(self.root, 'spikesorter_files', 'mountainsort_files', 'mountainsort3.mlp'),
                             join(self.mountain_folder, 'mountainsort3.mlp'))

                    print('Running MountainSort')
                    self.curate = True

                    import subprocess
                    os.chdir(self.mountain_folder)
                    try:
                        if self.curate:
                            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                     '--firings_out=firings.mda',
                                                     '--_params=params.json',
                                                     '--curate=true'])
                        else:
                            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                     '--firings_out=firings.mda',
                                                     '--_params=params.json'])
                        self.processing_time = time.time() - t_start_proc
                        print 'Elapsed time: ', self.processing_time
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)


                    print 'Parsing output files...'
                    self.firings = mlpy.readmda(join(self.mountain_folder, 'firings.mda'))
                    self.spike_trains = []
                    clust_id, n_counts = np.unique(self.firings[2], return_counts=True)
                    self.ml_times = self.times[self.firings[1].astype(int)]

                    self.counts = 0
                    for clust, count in zip(clust_id, n_counts):
                        if count > self.minimum_spikes_per_cluster:
                            idx = np.where(self.firings[2] == clust)[0]
                            assert len(np.unique(self.firings[0, idx]) == 1)
                            self.counts += len(idx)
                            spike_times = self.ml_times[idx]
                            spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                            spiketrain.annotate(ica_source=int(np.unique(self.firings[0, idx]))-1)
                            self.spike_trains.append(spiketrain)
                    self.sst = self.spike_trains

                elif self.clustering == 'mountain-back':

                    sys.path.append(join(self.root, 'spikesorter_files', 'mountainsort_files'))
                    import mlpy
                    from shutil import copyfile

                    if not os.path.isdir(join(self.orica_folder, 'mountain')):
                        os.makedirs(join(self.orica_folder, 'mountain'))
                    self.mountain_folder = join(self.orica_folder, 'mountain')
                    rec_name = os.path.split(self.rec_folder)
                    if rec_name[-1] == '':
                        rec_name = os.path.split(rec_name[0])[-1]
                    else:
                        rec_name = rec_name[-1]
                    self.mountain_full_path = join(self.mountain_folder, rec_name)

                    # write data file
                    filename = join(self.mountain_folder, 'raw.mda')
                    self.reconstructed = np.matmul(self.cleaned_A_orica.T, self.s_orica[self.source_idx])
                    mlpy.writemda32(self.reconstructed, filename)
                    print 'saving ', filename

                    # write csv probe file
                    with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                        for pos in self.mea_pos:
                            f.write(str(pos[1]))
                            f.write(',')
                            f.write(str(pos[2]))
                            f.write('\n')

                    # write param file
                    detect_threshold = None
                    radius = 100
                    params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': -1,
                              'adjacency_radius': radius}

                    with open(join(self.mountain_folder, 'params.json'), 'w') as f:
                        json.dump(params, f)

                    # copy mountainsort3.mlp
                    copyfile(join(self.root, 'spikesorter_files', 'mountainsort_files', 'mountainsort3.mlp'),
                             join(self.mountain_folder, 'mountainsort3.mlp'))

                    print('Running MountainSort')
                    self.curate = False

                    import subprocess
                    os.chdir(self.mountain_folder)
                    try:
                        if self.curate:
                            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                     '--firings_out=firings.mda',  '--geom=geom.csv',
                                                     '--_params=params.json',
                                                     '--curate=true'])
                        else:
                            subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                     '--firings_out=firings.mda', '--geom=geom.csv',
                                                     '--_params=params.json'])

                        self.processing_time = time.time() - t_start_proc
                        print 'Elapsed time: ', self.processing_time
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)


                    print 'Parsing output files...'
                    self.firings = mlpy.readmda(join(self.mountain_folder, 'firings.mda'))
                    self.spike_trains = []
                    clust_id, n_counts = np.unique(self.firings[2], return_counts=True)
                    self.ml_times = self.times[self.firings[1].astype(int)]

                    self.counts = 0
                    for clust, count in zip(clust_id, n_counts):
                        if count > self.minimum_spikes_per_cluster:
                            idx = np.where(self.firings[2] == clust)[0]
                            assert len(np.unique(self.firings[0, idx]) == 1)
                            self.counts += len(idx)
                            spike_times = self.ml_times[idx]
                            spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                            # spiketrain.annotate(ica_source=int(np.unique(self.firings[0, idx]))-1)
                            self.spike_trains.append(spiketrain)
                    self.spike_trains, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                    self.sst = self.spike_trains
                else:
                    self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                         t_start=self.gtst[0].t_start,
                                                         t_stop=self.gtst[0].t_stop)
                    self.spike_trains, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                    self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                    self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                    self.sst = self.spike_trains

                print 'Number of spike trains after clustering: ', len(self.spike_trains)
                print 'Number of spike trains after duplicate rejection: ', len(self.sst)

                self.processing_time = time.time() - t_start_proc

                if 'ica_source' in self.sst[0].annotations.keys():
                    self.independent_spike_idx = [s.annotations['ica_source'] for s in self.sst]

                self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                self.orA_spike_sources = self.cleaned_A_orica[self.independent_spike_idx]
                self.orW_spike_sources = self.cleaned_W_orica[self.independent_spike_idx]
                # self.orica_smoothing_spike_sources = self.cleaned_smoothing[self.independent_spike_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.orica_spike_sources, self.recordings, self.times, self.fs)

                # print 'Average smoothing: ', np.mean(self.orica_smoothing_spike_sources), ' mu=', mu
                print 'Elapsed time: ', time.time() - t_start

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('ORICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.orica_folder, 'results')):
                    os.makedirs(join(self.orica_folder, 'results'))
                np.save(join(self.orica_folder, 'results', 'counts'), self.counts)
                np.save(join(self.orica_folder, 'results', 'performance'), self.performance)
                np.save(join(self.orica_folder, 'results', 'time'), self.processing_time)

        if self.orica_online:
            if self.run_ss:
                print 'Applying ORICA online'
                t_start = time.time()
                t_start_proc = time.time()
                if not os.path.isdir(join(self.rec_folder, 'orica-online')):
                    os.makedirs(join(self.rec_folder, 'orica-online'))
                self.orica_folder = join(self.rec_folder, 'orica-online')

                self.setting_time = 5
                self.window = 5
                self.step = 1
                self.detection_thresh_online = 10
                self.skew_thresh_online = 0.8

                self.ori = orICA.onlineORICAss(self.recordings, fs=self.fs, forgetfac='cooling',
                                               skew_thresh=self.skew_thresh_online, lambda_0=0.995,
                                               verbose=True, block=self.block, step_size=self.step,
                                               window=self.window, initial_window=self.setting_time,
                                               detect_trheshold=self.detection_thresh_online)

                self.processing_time = time.time() - t_start_proc

                # Convert spikes to neo
                self.sst = []
                for (k, t) in zip(self.ori.spikes.keys(), self.ori.spikes.values()):
                    times = t/self.fs
                    st = neo.SpikeTrain(times=times, t_start=0*pq.s, t_stop = self.gtst[0].t_stop)
                    st.annotate(ica_source=k)
                    self.sst.append(st)

                self.oorica_spike_sources_idx = np.sort(self.ori.spikes.keys())
                self.oorica_spike_sources = self.ori.y[self.oorica_spike_sources_idx]
                self.oorA_spike_sources = self.ori.mixing[-1, self.oorica_spike_sources_idx]
                self.oorW_spike_sources = self.ori.unmixing[-1, self.oorica_spike_sources_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.oorica_spike_sources, self.recordings, self.times, self.fs)

                print 'Elapsed time: ', time.time() - t_start



                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst,
                                                                             t_start=self.setting_time)

                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('ORICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.orica_folder, 'results')):
                    os.makedirs(join(self.orica_folder, 'results'))
                np.save(join(self.orica_folder, 'results', 'counts'), self.counts)
                np.save(join(self.orica_folder, 'results', 'performance'), self.performance)
                np.save(join(self.orica_folder, 'results', 'time'), self.processing_time)

        if self.threshold_online:
            if self.run_ss:
                print 'Applying threshold online'
                t_start = time.time()
                t_start_proc = time.time()
                if not os.path.isdir(join(self.rec_folder, 'threshold-online')):
                    os.makedirs(join(self.rec_folder, 'threshold-online'))
                self.orica_folder = join(self.rec_folder, 'threshold-online')

                self.detection_threshold = -100
                self.spikes = threshold_spike_sorting(self.recordings, self.detection_threshold)

                self.processing_time = time.time() - t_start_proc

                # Convert spikes to neo
                self.sst = []
                for (k, t) in zip(self.spikes.keys(), self.spikes.values()):
                    times = t/self.fs
                    st = neo.SpikeTrain(times=times, t_start=0*pq.s, t_stop = self.gtst[0].t_stop)
                    st.annotate(electrode=k)
                    self.sst.append(st)

                # extract_wf(self.sst, self.oorica_spike_sources, self.recordings, self.times, self.fs)

                print 'Elapsed time: ', time.time() - t_start

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('ORICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.orica_folder, 'results')):
                    os.makedirs(join(self.orica_folder, 'results'))
                np.save(join(self.orica_folder, 'results', 'counts'), self.counts)
                np.save(join(self.orica_folder, 'results', 'performance'), self.performance)
                np.save(join(self.orica_folder, 'results', 'time'), self.processing_time)

        if self.corica:
            if self.run_ss:
                print 'Applying Convolutive Online Recursive ICA'
                t_start = time.time()
                chunk_size = int(2 * pq.s * self.fs.rescale('Hz'))
                n_chunks = 1
                self.s_orica, self.A_orica, self.W_orica = orICA.cICAemb(self.recordings)

                # clean sources based on skewness and correlation
                spike_sources, self.source_idx = clean_sources(self.s_orica, kurt_thresh=self.kurt_thresh,
                                                               skew_thresh=self.skew_thresh)


                self.cleaned_sources_orica = spike_sources
                self.cleaned_A_orica = self.A_orica[self.source_idx]
                self.cleaned_W_orica = self.W_orica[self.source_idx]
                print 'Number of cleaned sources: ', self.cleaned_sources_orica.shape[0]


                print 'Clustering Sources with: ', self.clustering
                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.gtst[0].t_start,
                                                            t_stop=self.gtst[0].t_stop, n_std=self.threshold)
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.detected_spikes]

                    self.spike_trains, self.amps, self.nclusters, self.keep, self.score = \
                        cluster_spike_amplitudes(self.detected_spikes, metric='cal',
                                                 alg=self.clustering)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.spike_trains)
                else:
                    self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                         t_start=self.gtst[0].t_start,
                                                         t_stop=self.gtst[0].t_stop)
                    self.spike_trains, self.independent_spike_idx = reject_duplicate_spiketrains(self.spike_trains)
                    self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                    self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                    self.sst = self.spike_trains

                self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                self.orA_spike_sources = self.cleaned_A_orica[self.independent_spike_idx]
                self.orW_spike_sources = self.cleaned_W_orica[self.independent_spike_idx]

                self.sst = self.spike_trains_rej

                print 'Elapsed time: ', time.time() - t_start

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('convORICA', fontsize=20)


        if self.klusta:
            print 'Applying Klustakwik algorithm'
            t_start = time.time()

            if not os.path.isdir(join(self.rec_folder, 'klusta')):
                os.makedirs(join(self.rec_folder, 'klusta'))
            self.klusta_folder = join(self.rec_folder, 'klusta')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            self.klusta_full_path = join(self.klusta_folder, 'recording')
            # create prb and prm files
            prb_path = export_prb_file(self.mea_pos.shape[0], self.electrode_name, self.klusta_folder,
                                       pos=self.mea_pos, adj_dist=2*np.max(self.mea_pitch))
            # save binary file
            filename = join(self.klusta_folder, 'recordings')
            file_path = save_binary_format(join(self.klusta_folder, 'recordings'), self.recordings,
                                           spikesorter='klusta')

            # set up klusta config file
            with open(join(self.root, 'spikesorter_files', 'klusta_files',
                           'config.prm'), 'r') as f:
                klusta_config = f.readlines()

            nchan = self.recordings.shape[0]
            threshold = self.threshold

            klusta_config = ''.join(klusta_config).format(
                filename, prb_path, float(self.fs.rescale('Hz')), nchan, 'float32', threshold
            )

            with open(join(self.klusta_folder, 'config.prm'), 'w') as f:
                f.writelines(klusta_config)

            if self.run_ss:
                print('Running klusta')
                try:
                    import klusta
                    # import klustakwik2
                except ImportError:
                    raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

                import subprocess
                try:
                    t_start_proc = time.time()
                    subprocess.check_output(['klusta', join(self.klusta_folder, 'config.prm'), '--overwrite'])
                    self.processing_time = time.time() - t_start_proc
                    print 'Elapsed time: ', self.processing_time
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                kwikfile = [f for f in os.listdir(self.klusta_folder) if f.endswith('.kwik')]
                if len(kwikfile) > 0:
                    kwikfile = join(self.klusta_folder, kwikfile[0])
                    if os.path.exists(kwikfile):
                        kwikio = neo.io.KwikIO(filename=kwikfile, )
                        blk = kwikio.read_block(raw_data_units='uV')
                        self.sst = blk.segments[0].spiketrains
                else:
                    raise Exception('No kwik file!')

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('KLUSTA', fontsize=20)

                print 'Elapsed time: ', time.time() - t_start

                # save results
                if not os.path.isdir(join(self.klusta_folder, 'results')):
                    os.makedirs(join(self.klusta_folder, 'results'))
                np.save(join(self.klusta_folder, 'results', 'counts'), self.counts)
                np.save(join(self.klusta_folder, 'results', 'performance'), self.performance)
                np.save(join(self.klusta_folder, 'results', 'time'), self.processing_time)


        if self.kilo:
            print 'Applying Kilosort algorithm'

            t_start = time.time()

            # setup folders
            if not os.path.isdir(join(self.rec_folder, 'kilosort')):
                os.makedirs(join(self.rec_folder, 'kilosort'))
            self.kilo_folder = join(self.rec_folder, 'kilosort')
            self.kilo_process_folder = join(self.kilo_folder, 'process')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            self.kilo_raw = join(self.kilo_folder, 'raw.dat')

            save_binary_format(self.kilo_raw,
                               self.recordings,
                               spikesorter='kilosort',
                               dtype='int16')
            threshold = self.threshold

            # set up kilosort config files and run kilosort on data
            with open(join(self.root, 'spikesorter_files', 'kilosort_files',
                                   'kilosort_master.txt'), 'r') as f:
                kilosort_master = f.readlines()
            with open(join(self.root, 'spikesorter_files', 'kilosort_files',
                                   'kilosort_config.txt'), 'r') as f:
                kilosort_config = f.readlines()
            with open(join(self.root, 'spikesorter_files', 'kilosort_files',
                                   'kilosort_channelmap.txt'), 'r') as f:
                kilosort_channelmap = f.readlines()
            nchan = self.recordings.shape[0]
            dat_file = 'raw.dat'
            kilo_thresh = 6
            Nfilt = (nchan//32)*32*4
            if Nfilt == 0:
                Nfilt = 64
            nsamples = 128*1024 + 32 #self.recordings.shape[1]

            kilosort_master = ''.join(kilosort_master).format(
                self.kilo_folder
            )
            kilosort_config = ''.join(kilosort_config).format(
                nchan, nchan, int(self.fs.rescale('Hz')), dat_file, Nfilt, nsamples, kilo_thresh
            )
            kilosort_channelmap = ''.join(kilosort_channelmap
                                          ).format(nchan, list(self.mea_pos[:, 1]), list(self.mea_pos[:, 2]),
                                                   int(self.fs.rescale('Hz')))
            for fname, value in zip(['kilosort_master.m', 'kilosort_config.m',
                                     'kilosort_channelmap.m'],
                                    [kilosort_master, kilosort_config,
                                     kilosort_channelmap]):
                with open(join(self.kilo_folder, fname), 'w') as f:
                    f.writelines(value)
            # start sorting with kilosort

            if self.run_ss:
                print('Running KiloSort')
                os.chdir(self.kilo_folder)
                try:
                    t_start_proc = time.time()
                    import subprocess
                    subprocess.call(['matlab', '-nodisplay', '-nodesktop',
                                     '-nosplash', '-r',
                                     'run kilosort_master.m; exit;'])
                    print 'Elapsed time: ', time.time() - t_start_proc
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                os.chdir(self.root)

                print 'Parsing output files...'
                self.spike_times = np.load(join(self.kilo_process_folder, 'spike_times.npy'))
                self.spike_clusters = np.load(join(self.kilo_process_folder, 'spike_clusters.npy'))
                self.kl_templates = np.load(join(self.kilo_process_folder, 'templates.npy')).swapaxes(1, 2)
                self.spike_templates_id = np.load(join(self.kilo_process_folder, 'spike_templates.npy'))
                self.spike_templates = []

                with open(join(self.kilo_process_folder, 'time.txt')) as f:
                    self.processing_time = float(f.readlines()[0])

                self.spike_trains = []
                clust_id, n_counts = np.unique(self.spike_clusters, return_counts=True)
                self.kl_times = self.times[self.spike_times.astype(int)]

                self.counts = 0

                for clust, count in zip(clust_id, n_counts):
                    if count > self.minimum_spikes_per_cluster:
                        idx = np.where(self.spike_clusters == clust)[0]
                        self.spike_templates.append(self.kl_templates[clust])
                        self.counts += len(idx)
                        spike_times = self.kl_times[idx]
                        spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                        self.spike_trains.append(spiketrain)

                self.spike_templates = np.array(self.spike_templates)

                # print 'Finding independent spiketrains...'
                # self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                # print 'Found ', len(self.sst), ' independent spiketrains!'
                self.sst = self.spike_trains

                print 'Evaluating spiketrains...'
                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance = compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('KILOSORT', fontsize=20)

                print 'Total elapsed time: ', time.time() - t_start

                # save results
                if not os.path.isdir(join(self.kilo_folder, 'results')):
                    os.makedirs(join(self.kilo_folder, 'results'))
                np.save(join(self.kilo_folder, 'results', 'counts'), self.counts)
                np.save(join(self.kilo_folder, 'results', 'performance'), self.performance)
                np.save(join(self.kilo_folder, 'results', 'time'), self.processing_time)


        if self.mountain:
            print 'Applying Mountainsort algorithm'

            sys.path.append(join(self.root, 'spikesorter_files', 'mountainsort_files'))

            import mlpy
            from shutil import copyfile

            t_start = time.time()

            if not os.path.isdir(join(self.rec_folder, 'mountain')):
                os.makedirs(join(self.rec_folder, 'mountain'))
            self.mountain_folder = join(self.rec_folder, 'mountain')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            self.mountain_full_path = join(self.mountain_folder, rec_name)

            # write data file
            filename = join(self.mountain_folder, 'raw.mda')
            mlpy.writemda32(self.recordings, filename)
            print 'saving ', filename
            radius = 50

            # write csv probe file
            with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                for pos in self.mea_pos:
                    f.write(str(pos[1]))
                    f.write(',')
                    f.write(str(pos[2]))
                    f.write('\n')

            # write param file
            detect_threshold = None
            params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': -1,
                      "adjacency_radius": radius}
            with open(join(self.mountain_folder, 'params.json'), 'w') as f:
                json.dump(params, f)

            # copy mountainsort3.mlp
            copyfile(join(self.root, 'spikesorter_files', 'mountainsort_files', 'mountainsort3.mlp'),
                     join(self.mountain_folder, 'mountainsort3.mlp'))

            if self.run_ss:
                print('Running MountainSort')
                self.curate=True

                import subprocess
                os.chdir(self.mountain_folder)
                try:
                    t_start_proc = time.time()
                    if self.curate:
                        subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                 '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json',
                                                 '--curate=true'])
                    else:
                        subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                                 '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json'])
                    self.processing_time = time.time() - t_start_proc
                    print 'Elapsed time: ', self.processing_time
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                os.chdir(self.root)

                print 'Parsing output files...'
                self.firings = mlpy.readmda(join(self.mountain_folder, 'firings.mda'))
                self.spike_trains = []
                clust_id, n_counts = np.unique(self.firings[2], return_counts=True)
                self.ml_times = self.times[self.firings[1].astype(int)]

                self.counts = 0
                for clust, count in zip(clust_id, n_counts):
                    if count > self.minimum_spikes_per_cluster:
                        idx = np.where(self.firings[2] == clust)[0]
                        self.counts += len(idx)
                        spike_times = self.ml_times[idx]
                        spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                        self.spike_trains.append(spiketrain)

                print 'Finding independent spiketrains...'
                self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                print 'Found ', len(self.sst), ' independent spiketrains!'

                print 'Evaluating spiketrains...'
                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('MOUNTAINSORT', fontsize=20)

                print 'Total elapsed time: ', time.time() - t_start

                # save results
                if not os.path.isdir(join(self.mountain_folder, 'results')):
                    os.makedirs(join(self.mountain_folder, 'results'))
                np.save(join(self.mountain_folder, 'results', 'counts'), self.counts)
                np.save(join(self.mountain_folder, 'results', 'performance'), self.performance)
                np.save(join(self.mountain_folder, 'results', 'time'), self.processing_time)

        if self.circus:
            print 'Applying Spyking-circus algorithm'
            t_start = time.time()

            if not os.path.isdir(join(self.rec_folder, 'spykingcircus')):
                os.makedirs(join(self.rec_folder, 'spykingcircus'))
            self.spykingcircus_folder = join(self.rec_folder, 'spykingcircus')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            # create prb and prm files
            prb_path = export_prb_file(self.mea_pos.shape[0], self.electrode_name, self.spykingcircus_folder,
                                       pos=self.mea_pos, adj_dist=2*np.max(self.mea_pitch), spikesorter='spykingcircus',
                                       radius=50)

            filename = 'recordings'

            # # save binary file
            # save_binary_format(join(self.spykingcircus_folder, dat_file), self.recordings, spikesorter='spykingcircus')
            np.save(join(self.spykingcircus_folder, filename), self.recordings)

            # set up spykingcircus config file
            with open(join(self.root, 'spikesorter_files', 'spykingcircus_files',
                           'config.params'), 'r') as f:
                circus_config = f.readlines()

            nchan = self.recordings.shape[0]
            threshold = 6 #6
            filter = False

            if self.merge_spikes:
                auto=1e-5
            else:
                auto=0

            circus_config = ''.join(circus_config).format(
                'numpy', float(self.fs.rescale('Hz')), prb_path, threshold, filter, auto
            )

            with open(join(self.spykingcircus_folder, filename + '.params'), 'w') as f:
                f.writelines(circus_config)

            if self.run_ss:
                print('Running Spyking-Circus')

                import subprocess
                try:
                    import multiprocessing
                    n_cores = multiprocessing.cpu_count()
                    os.chdir(self.spykingcircus_folder)
                    t_start_proc = time.time()
                    subprocess.check_output(['spyking-circus', 'recordings.npy', '-c', str(n_cores)])
                    if self.merge_spikes:
                        subprocess.call(['spyking-circus', 'recordings.npy', '-m', 'merging', '-c', str(n_cores)])
                    self.processing_time = time.time() - t_start_proc
                    print 'Elapsed time: ', self.processing_time
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                print 'Parsing output files...'
                os.chdir(self.root)
                if self.merge_spikes:
                    f = h5py.File(join(self.spykingcircus_folder, filename, filename + '.result-merged.hdf5'))
                else:
                    f = h5py.File(join(self.spykingcircus_folder, filename, filename + '.result.hdf5'))
                self.spike_times = []
                self.spike_clusters = []

                for temp in f['spiketimes'].keys():
                    self.spike_times.append(f['spiketimes'][temp].value)
                    self.spike_clusters.append(int(temp.split('_')[-1]))

                self.spike_trains = []
                self.counts = 0

                for i_st, st in enumerate(self.spike_times):
                    count = len(st)
                    self.counts += count
                    if count > self.minimum_spikes_per_cluster:
                        spike_times = self.times[sorted(st)]
                        spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                        self.spike_trains.append(spiketrain)
                    else:
                        print 'Discarded spike train ', i_st

                self.sst = self.spike_trains
                print 'Found ', len(self.sst), ' independent spiketrains!'

                print 'Evaluating spiketrains...'
                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('SPYKING-CIRCUS', fontsize=20)

                print 'Elapsed time: ', time.time() - t_start

                # save results
                if not os.path.isdir(join(self.spykingcircus_folder, 'results')):
                    os.makedirs(join(self.spykingcircus_folder, 'results'))
                np.save(join(self.spykingcircus_folder, 'results', 'counts'), self.counts)
                np.save(join(self.spykingcircus_folder, 'results','performance'), self.performance)
                np.save(join(self.spykingcircus_folder, 'results', 'time'), self.processing_time)


        if self.yass:
            print 'Applying YASS algorithm'
            t_start = time.time()

            if not os.path.isdir(join(self.rec_folder, 'yass')):
                os.makedirs(join(self.rec_folder, 'yass'))
            self.yass_folder = join(self.rec_folder, 'yass')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]

            filename = 'recordings'
            dat_file = 'recordings.dat'

            # save binary file
            save_binary_format(join(self.yass_folder, dat_file), self.recordings, spikesorter='yass', dtype='int16')

            # save probe file
            np.save(join(self.yass_folder, self.electrode_name), list(self.mea_pos))

            # set up yass config file
            with open(join(self.root, 'spikesorter_files', 'yass_files',
                           'config_sample_complete.yaml'), 'r') as f:
                yass_config = f.readlines()

            nchan = self.recordings.shape[0]
            threshold = self.threshold
            filter = False

            yass_config = ''.join(yass_config).format(
                './', dat_file, self.electrode_name + '.npy', 'int16', int(self.fs.rescale('Hz')), nchan, filter
            )

            with open(join(self.yass_folder, 'config.yaml'), 'w') as f:
                f.writelines(yass_config)

            if self.run_ss:
                print('Running YASS')

                import subprocess
                try:
                    os.chdir(self.yass_folder)
                    t_start_proc = time.time()
                    subprocess.check_output(['yass', 'config.yaml'])
                    self.processing_time = time.time() - t_start_proc
                    print 'Elapsed time: ', self.processing_time
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                print 'Parsing output files...'
                os.chdir(self.root)

                import csv
                self.spike_times = []
                self.spike_clusters = []
                with open(join(self.yass_folder, 'spike_train.csv'), 'rb') as csvfile:
                    file = csv.reader(csvfile, delimiter=',')
                    for row in file:
                        self.spike_times.append(int(row[0]))
                        self.spike_clusters.append(int(row[1]))

                self.spike_trains = []
                clust_id, n_counts = np.unique(self.spike_clusters, return_counts=True)

                self.counts = 0

                for clust, count in zip(clust_id, n_counts):
                    if count > self.minimum_spikes_per_cluster:
                        idx = np.where(self.spike_clusters == clust)[0]
                        self.counts += len(idx)
                        spike_times = self.times[np.array(self.spike_times)[idx]]
                        spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                        self.spike_trains.append(spiketrain)

                self.sst = self.spike_trains

                print 'Evaluating spiketrains...'
                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print 'PAIRS: '
                print self.pairs

                self.performance = compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('YASS', fontsize=20)

                print 'Elapsed time: ', time.time() - t_start

                # save results
                if not os.path.isdir(join(self.yass_folder, 'results')):
                    os.makedirs(join(self.yass_folder, 'results'))
                np.save(join(self.yass_folder, 'results', 'counts'), self.counts)
                np.save(join(self.yass_folder, 'results', 'performance'), self.performance)
                np.save(join(self.yass_folder, 'results', 'time'), self.processing_time)

        print 'DONE'


    def plot_results(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
        raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

        fig.tight_layout()
        return ax1, ax2

def compute_performance(counts):

    tp_rate = float(counts['TP']) / counts['TOT_GT'] * 100
    tpo_rate = float(counts['TPO']) / counts['TOT_GT'] * 100
    tpso_rate = float(counts['TPSO']) / counts['TOT_GT'] * 100
    tot_tp_rate = float(counts['TP'] + counts['TPO'] + counts['TPSO']) / counts['TOT_GT'] * 100

    cl_rate = float(counts['CL']) / counts['TOT_GT'] * 100
    clo_rate = float(counts['CLO']) / counts['TOT_GT'] * 100
    clso_rate = float(counts['CLSO']) / counts['TOT_GT'] * 100
    tot_cl_rate = float(counts['CL'] + counts['CLO'] + counts['CLSO']) / counts['TOT_GT'] * 100

    fn_rate = float(counts['FN']) / counts['TOT_GT'] * 100
    fno_rate = float(counts['FNO']) / counts['TOT_GT'] * 100
    fnso_rate = float(counts['FNSO']) / counts['TOT_GT'] * 100
    tot_fn_rate = float(counts['FN'] + counts['FNO'] + counts['FNSO']) / counts['TOT_GT'] * 100

    fp_gt = float(counts['FP']) / counts['TOT_GT'] * 100
    fp_st = float(counts['FP']) / counts['TOT_ST'] * 100

    accuracy = tot_tp_rate / (tot_tp_rate + tot_fn_rate + fp_gt) * 100
    sensitivity = tot_tp_rate / (tot_tp_rate + tot_fn_rate) * 100
    miss_rate = tot_fn_rate / (tot_tp_rate + tot_fn_rate) * 100
    precision = tot_tp_rate / (tot_tp_rate + fp_gt) * 100
    false_discovery_rate = fp_gt / (tot_tp_rate + fp_gt) * 100

    print 'PERFORMANCE: \n'
    print '\nTP: ', tp_rate, ' %'
    print 'TPO: ', tpo_rate, ' %'
    print 'TPSO: ', tpso_rate, ' %'
    print 'TOT TP: ', tot_tp_rate, ' %'

    print '\nCL: ', cl_rate, ' %'
    print 'CLO: ', clo_rate, ' %'
    print 'CLSO: ', clso_rate, ' %'
    print 'TOT CL: ', tot_cl_rate, ' %'

    print '\nFN: ', fn_rate, ' %'
    print 'FNO: ', fno_rate, ' %'
    print 'FNSO: ', fnso_rate, ' %'
    print 'TOT FN: ', tot_fn_rate, ' %'

    print '\nFP (%GT): ', fp_gt, ' %'
    print '\nFP (%ST): ', fp_st, ' %'

    print '\nACCURACY: ', accuracy, ' %'
    print 'SENSITIVITY: ', sensitivity, ' %'
    print 'MISS RATE: ', miss_rate, ' %'
    print 'PRECISION: ', precision, ' %'
    print 'FALSE DISCOVERY RATE: ', false_discovery_rate, ' %'

    performance = {'tot_tp': tot_tp_rate, 'tot_cl': tot_cl_rate, 'tot_fn': tot_fn_rate, 'tot_fp': fp_gt,
                   'accuracy': accuracy, 'sensitivity': sensitivity, 'precision': precision, 'miss_rate': miss_rate,
                   'false_disc_rate': false_discovery_rate}

    return performance


def plot_mixing(mixing, mea_pos, mea_dim, gs=None, fig=None):
    '''

    Parameters
    ----------
    mixing
    mea_pos
    mea_dim
    mea_pitch

    Returns
    -------

    '''
    from neuroplot import plot_weight

    n_sources = len(mixing)
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))

    if fig is None and gs is None:
        fig = plt.figure()

    if gs is not None:
        from matplotlib import gridspec
        gs_plot = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs)

    for n in range(n_sources):
        if gs is not None:
            ax_w = fig.add_subplot(gs_plot[n])
        else:
            ax_w = fig.add_subplot(rows, cols, n+1)
        mix = mixing[n]/np.ptp(mixing[n])
        plot_weight(mix, mea_dim, ax=ax_w)

    return fig


def play_mixing(mixing, mea_pos, mea_dim):
    '''

    Parameters
    ----------
    mixing
    mea_pos
    mea_dim
    mea_pitch

    Returns
    -------

    '''
    from neuroplot import plot_weight

    n_sources = len(mixing)
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))
    fig_t = plt.figure()

    anim = []

    for n in range(n_sources):
        ax_w = fig_t.add_subplot(rows, cols, n+1)
        mix = mixing[n]/np.ptp(mixing[n])
        im = play_spike(mix, mea_dim, ax=ax_w, fig=fig_t)
        anim.append(im)

    return fig_t


def plot_templates(templates, mea_pos, mea_pitch, single_figure=True):
    '''

    Parameters
    ----------
    templates
    mea_pos
    mea_pitch

    Returns
    -------

    '''
    from neuroplot import plot_weight

    n_sources = len(templates)
    fig_t = plt.figure()

    if single_figure:
        colors = plt.rcParams['axes.color_cycle']
        ax_t = fig_t.add_subplot(111)

        for n, t in enumerate(templates):
            print 'Plotting spike ', n, ' out of ', n_sources
            if len(t.shape) == 3:
                # plot_mea_recording(w[:5], mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=0.1)
                plot_mea_recording(t.mean(axis=0), mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)
            else:
                plot_mea_recording(t, mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_t, lw=2)

    else:
        cols = int(np.ceil(np.sqrt(n_sources)))
        rows = int(np.ceil(n_sources / float(cols)))

        for n in range(n_sources):
            ax_t = fig_t.add_subplot(rows, cols, n+1)
            plot_mea_recording(templates[n], mea_pos, mea_pitch, ax=ax_t)

    return fig_t



def templates_weights(templates, weights, mea_pos, mea_dim, mea_pitch, pairs=None):
    '''

    Parameters
    ----------
    templates
    weights
    pairs
    mea_pos
    mea_dim
    mea_pitch

    Returns
    -------

    '''
    from matplotlib import gridspec
    n_neurons = len(templates)
    gs0 = gridspec.GridSpec(1, 2)

    cols = int(np.ceil(np.sqrt(n_neurons)))
    rows = int(np.ceil(n_neurons / float(cols)))

    gs_t = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[0])
    gs_w = gridspec.GridSpecFromSubplotSpec(rows, cols, subplot_spec=gs0[1])
    fig_t = plt.figure()

    min_v = np.min(templates)

    for n in range(n_neurons):
        ax_t = fig_t.add_subplot(gs_t[n])
        ax_w = fig_t.add_subplot(gs_w[n])

        plot_mea_recording(templates[n], mea_pos, mea_pitch, ax=ax_t)
        if pairs is not None:
            if pairs[n, 0] != -1:
                plot_weight(weights[pairs[n, 1]], mea_dim, ax=ax_w)
            else:
                ax_w.axis('off')
        else:
            plot_weight(weights[n], mea_dim, ax=ax_w)

    return fig_t

def plot_ic_waveforms(spiketrains):
    ica_sources = [s.annotations['ica_source'] for s in spiketrains]
    sources = np.unique(ica_sources)
    n_sources = len(sources)
    cols = int(np.ceil(np.sqrt(n_sources)))
    rows = int(np.ceil(n_sources / float(cols)))
    fig_t = plt.figure()
    colors = plt.rcParams['axes.color_cycle']

    for n, s in enumerate(sources):
        ax_t = fig_t.add_subplot(rows, cols, n + 1)

        count = 0
        for st, ic in enumerate(ica_sources):
            if ic == s:
                ax_t.plot(spiketrains[st].annotations['ica_wf'].T, color=colors[count], lw=0.2)
                ax_t.plot(np.mean(spiketrains[st].annotations['ica_wf'], axis=0), color=colors[count], lw=1)
                count += 1

    return fig_t

def plot_waveforms(spiketrains, mea_pos, mea_pitch):
    wf = [s.waveforms for s in spiketrains]
    fig_w = plt.figure()
    ax_w = fig_w.add_subplot(111)
    colors = plt.rcParams['axes.color_cycle']

    for n, w in enumerate(wf):
        print 'Plotting spike ', n, ' out of ', len(wf)
        # plot_mea_recording(w[:5], mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=0.1)
        plot_mea_recording(w.mean(axis=0), mea_pos, mea_pitch, colors=colors[np.mod(n, len(colors))], ax=ax_w, lw=2)

    return fig_w


if __name__ == '__main__':
    '''
        COMMAND-LINE 

    '''

    if '-r' in sys.argv:
        pos = sys.argv.index('-r')
        rec_folder = sys.argv[pos + 1]
    if '-mod' in sys.argv:
        pos = sys.argv.index('-mod')
        mod = sys.argv[pos + 1]
    else:
        mod = 'ICA'
    if '-dur' in sys.argv:
        pos = sys.argv.index('-dur')
        dur = sys.argv[pos + 1]
    else:
        dur = None
    if '-tstart' in sys.argv:
        pos = sys.argv.index('-tstart')
        tstart = sys.argv[pos + 1]
    else:
        tstart = None
    if '-tstop' in sys.argv:
        pos = sys.argv.index('-tstop')
        tstop = sys.argv[pos + 1]
    else:
        tstop = None
    if '-thresh' in sys.argv:
        pos = sys.argv.index('-thresh')
        thresh = int(sys.argv[pos + 1])
    else:
        thresh = 4
    if '-M' in sys.argv:
        pos = sys.argv.index('-M')
        ndim = int(sys.argv[pos + 1])
    else:
        ndim = 'all'
    if '-norun' in sys.argv:
        spikesort = False
    else:
        spikesort = True
    if '-noplot' in sys.argv:
        plot_figures = False
    else:
        plot_figures = True
    if '-merge' in sys.argv:
        merge_spikes = True
    else:
        merge_spikes = False
    if '-mu' in sys.argv:
        pos = sys.argv.index('-mu')
        mu = float(sys.argv[pos + 1])
    else:
        mu = 0
    if '-eta' in sys.argv:
        pos = sys.argv.index('-eta')
        eta = float(sys.argv[pos + 1])
    else:
        eta = 0
    if '-npass' in sys.argv:
        pos = sys.argv.index('-npass')
        npass = int(sys.argv[pos + 1])
    else:
        npass = 1
    if '-block' in sys.argv:
        pos = sys.argv.index('-block')
        block = int(sys.argv[pos + 1])
    else:
        block = 1000
    if '-feat' in sys.argv:
        pos = sys.argv.index('-feat')
        feat = sys.argv[pos + 1]
    else:
        feat = 'pca'
    if '-clust' in sys.argv:
        pos = sys.argv.index('-clust')
        clust = sys.argv[pos + 1]
    else:
        clust = 'mog'
    if '-nokeep' in sys.argv:
        keepall = False
    else:
        keepall = True
    if '-nosave' in sys.argv:
        save = False
    else:
        save = True

    debug = False
    if debug:
        rec_folder='recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_physrot_Neuronexus-32-cut-30_10_' \
                   '10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_10-05-2018:11:37_3002/'
        block=100
        mod='picard'
        feat='pca'
        keepall=True

    if len(sys.argv) == 1 and not debug:
        print 'Arguments: \n   -r recording filename\n   -mod ICA - cICA - orica - corica - oricaonline - threshonline - klusta' \
              'kilosort - mountainsort - spykingcircus  -yass\n   -dur duration in s\n   -tstart start time in s\n' \
              '   -tstop stop time in s\n   -M   number of dimensions'

    elif '-r' not in sys.argv and not debug:
        raise AttributeError('Provide model folder for data')
    else:
        sps = SpikeSorter(save=False, rec_folder=rec_folder, alg=mod, duration=dur,
                          tstart=tstart, tstop=tstop, run_ss=spikesort, plot_figures=plot_figures,
                          merge_spikes=merge_spikes, mu=mu, eta=eta, npass=npass, block=block, feat=feat,
                          clust=clust, keepall=keepall, ndim=ndim)