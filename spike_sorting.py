'''
Spike sorting of neural recordins with various spike sorters
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
# import smoothICA as sICA
import orICA

root_folder = os.getcwd()

#TODO iterative approach

class SpikeSorter:
    def __init__(self, save=False, rec_folder=None, alg=None, lag=None, gfmode=None, duration=None,
                 tstart=None, tstop=None, run_ss=None, plot_figures=True, merge_spikes=False, mu=0, eta=0,
                 npass=1, block=1000, feat='pca', clust='mog', keepall=True, ndim=None, eval=False):
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

        print(self.rec_name)

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

        if 'exp' in self.rec_folder:
            self.recordings = np.load(join(self.rec_folder, 'recordings.npy')) #.astype('int16')
            rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
            with open(join(self.rec_folder, rec_info), 'r') as f:
                self.info = yaml.load(f)

            self.electrode_name = self.info['General']['electrode name']
            fs = self.info['General']['fs']
            if isinstance(fs, str):
                self.fs = pq.Quantity(float(fs.split()[0]), fs.split()[1])
            elif isinstance(fs, pq.Quantity):
                self.fs = fs

            self.times = (range(self.recordings.shape[1]) / self.fs).rescale('s')

            # self.overlapping = self.info['Synchrony']['overlap_pairs']
            # if 'overlap' not in self.gtst[0].annotations.keys():
            #     print('Finding overlapping spikes'
            #     annotate_overlapping(self.gtst, overlapping_pairs=self.overlapping)

            # load MEA info
            elinfo = MEA.return_mea_info(self.electrode_name)
            self.mea = MEA.return_mea(self.electrode_name, sortlist=None)
            self.mea_pos = self.mea.positions
            mea_shape = elinfo['shape']

            self.t_start = 0 * pq.s
            dur = self.info['General']['duration']

            if 's' in dur:
                self.t_stop = float(dur[:-1]) * pq.s
            else:
                self.t_stop = float(self.info['General']['duration']) * pq.s

        elif self.rec_name.startswith('recording'):
            self.gtst = np.load(join(self.rec_folder, 'spiketrains.npy'))
            self.recordings = np.load(join(self.rec_folder, 'recordings.npy')) #.astype('int16')
            self.templates = np.load(join(self.rec_folder, 'templates.npy'))
            self.templates_cat = np.load(join(self.rec_folder, 'templates_cat.npy'))
            self.templates_loc = np.load(join(self.rec_folder, 'templates_loc.npy'))

            rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
            with open(join(self.rec_folder, rec_info), 'r') as f:
                self.info = yaml.load(f)

            self.electrode_name = self.info['General']['electrode name']
            fs = self.info['General']['fs']
            if isinstance(fs, str):
                self.fs = pq.Quantity(float(fs.split()[0]), fs.split()[1])
            elif isinstance(fs, pq.Quantity):
                self.fs = fs

            self.times = (range(self.recordings.shape[1]) / self.fs).rescale('s')
            self.overlapping = self.info['Synchrony']['overlap_pairs']
            # if 'overlap' not in self.gtst[0].annotations.keys():
            #     print('Finding overlapping spikes'
            #     annotate_overlapping(self.gtst, overlapping_pairs=self.overlapping)

            # load MEA info
            elinfo = MEA.return_mea_info(self.electrode_name)
            self.mea = MEA.return_mea(self.electrode_name)
            self.mea_pos = self.mea.positions
            mea_shape = elinfo['shape']

            self.t_start = 0 * pq.s
            dur = self.info['General']['duration']

            if 's' in dur:
                self.t_stop = float(dur[:-1]) * pq.s
            else:
                self.t_stop = float(self.info['General']['duration']) * pq.s

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
                print('Applying instantaneous ICA')
                t_start = time.time()
                t_start_proc = time.time()

                if not os.path.isdir(join(self.rec_folder, 'ica')):
                    os.makedirs(join(self.rec_folder, 'ica'))
                self.ica_folder = join(self.rec_folder, 'ica')

                chunk_size = int(2*pq.s * self.fs.rescale('Hz'))
                n_chunks = 1
                self.s_ica, self.A_ica, self.W_ica = ica.instICA(self.recordings, n_comp=self.ndim,
                                                                 n_chunks=n_chunks, chunk_size=chunk_size)
                print('ICA Finished. Elapsed time: ', time.time() - t_start, ' sec.')

                # clean sources based on skewness and correlation
                spike_sources, self.source_idx = clean_sources(self.s_ica, kurt_thresh=self.kurt_thresh,
                                                               skew_thresh=self.skew_thresh)

                self.cleaned_sources_ica = spike_sources
                self.cleaned_A_ica = self.A_ica[self.source_idx]
                self.cleaned_W_ica = self.W_ica[self.source_idx]

                print('Number of cleaned sources: ', self.cleaned_sources_ica.shape[0])

                print('Clustering Sources with: ', self.clustering)

                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.t_start,
                                                            t_stop=self.t_stop, n_std=self.threshold)
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
                        print('saving ', filename)


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
                        print('saving ', filename)

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
                        print('Elapsed time: ', self.processing_time)
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)

                    print('Parsing output files...')
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
                            spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                            spiketrain.annotate(ica_source=int(np.unique(self.firings[0, idx])) - 1)
                            self.spike_trains.append(spiketrain)
                    # TODO extract waveforms
                    self.sst = self.spike_trains

                else:
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.t_start,
                                                            t_stop=self.t_stop)
                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.detected_spikes)
                    self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains_rej]
                    self.sst = self.spike_trains_rej

                print('Number of spike trains after clustering: ', len(self.spike_trains))
                print('Number of spike trains after duplicate rejection: ', len(self.sst))

                if 'ica_source' in self.sst[0].annotations.keys():
                    self.independent_spike_idx = [s.annotations['ica_source'] for s in self.sst]

                self.ica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                self.A_spike_sources = self.cleaned_A_ica[self.independent_spike_idx]
                self.W_spike_sources = self.cleaned_W_ica[self.independent_spike_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.recordings, self.times, self.fs, ica=True, sources=self.ica_spike_sources)

                self.processing_time = time.time() - t_start_proc
                print('Elapsed time: ', self.processing_time)

                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print('PAIRS: '
                # print(self.pairs
                #
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('ICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.ica_folder, 'results')):
                    os.makedirs(join(self.ica_folder, 'results'))
                if save:
                    np.save(join(self.ica_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.ica_folder, 'results', 'time'), self.processing_time)
                #
                # np.save(join(self.ica_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.ica_folder, 'results', 'performance'), self.performance)
                # np.save(join(self.ica_folder, 'results', 'time'), self.processing_time)

        if self.orica:
            # TODO: quantify smoothing with and without mu
            if self.run_ss:
                print('Applying Online Recursive ICA')
                t_start = time.time()
                t_start_proc = time.time()
                if not os.path.isdir(join(self.rec_folder, 'orica')):
                    os.makedirs(join(self.rec_folder, 'orica'))
                self.orica_folder = join(self.rec_folder, 'orica')

                chunk_size = int(2 * pq.s * self.fs.rescale('Hz'))
                n_chunks = 1
                self.s_orica, self.A_orica, self.W_orica = orICA.instICA(self.recordings, n_comp=self.ndim,
                                                                         n_chunks=n_chunks, chunk_size=chunk_size,
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
                print('Number of cleaned sources: ', self.cleaned_sources_orica.shape[0])


                print('Clustering Sources with: ', self.clustering)
                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.t_start,
                                                            t_stop=self.t_stop, n_std=self.threshold)
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
                    print('saving ', filename)

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
                        print('Elapsed time: ', self.processing_time)
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)


                    print('Parsing output files...')
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
                            spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
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
                    print('saving ', filename)

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
                        print('Elapsed time: ', self.processing_time)
                    except subprocess.CalledProcessError as e:
                        raise Exception(e.output)

                    os.chdir(self.root)


                    print('Parsing output files...')
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
                            spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                            # spiketrain.annotate(ica_source=int(np.unique(self.firings[0, idx]))-1)
                            self.spike_trains.append(spiketrain)
                    self.spike_trains, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                    self.sst = self.spike_trains
                else:
                    self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                         t_start=self.t_start,
                                                         t_stop=self.t_stop)
                    self.spike_trains, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                    self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                    self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                    self.sst = self.spike_trains

                print('Number of spike trains after clustering: ', len(self.spike_trains))
                print('Number of spike trains after duplicate rejection: ', len(self.sst))

                self.processing_time = time.time() - t_start_proc

                if 'ica_source' in self.sst[0].annotations.keys():
                    self.independent_spike_idx = [s.annotations['ica_source'] for s in self.sst]

                self.orica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.orica_spike_sources = self.cleaned_sources_orica[self.independent_spike_idx]
                self.orA_spike_sources = self.cleaned_A_orica[self.independent_spike_idx]
                self.orW_spike_sources = self.cleaned_W_orica[self.independent_spike_idx]
                # self.orica_smoothing_spike_sources = self.cleaned_smoothing[self.independent_spike_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.recordings, self.times, self.fs, ica=True, sources=self.orica_spike_sources)

                # print('Average smoothing: ', np.mean(self.orica_smoothing_spike_sources), ' mu=', mu
                print('Elapsed time: ', time.time() - t_start)

                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print(self.pairs
                #
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('ORICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.orica_folder, 'results')):
                    os.makedirs(join(self.orica_folder, 'results'))
                if save:
                    np.save(join(self.orica_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.orica_folder, 'results', 'time'), self.processing_time)

                # np.save(join(self.orica_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.orica_folder, 'results', 'performance'), self.performance)
                # np.save(join(self.orica_folder, 'results', 'time'), self.processing_time)

        if self.orica_online:
            if self.run_ss:
                print('Applying ORICA online')
                t_start = time.time()
                t_start_proc = time.time()
                if not os.path.isdir(join(self.rec_folder, 'orica-online')):
                    os.makedirs(join(self.rec_folder, 'orica-online'))
                self.orica_online_folder = join(self.rec_folder, 'orica-online')

                self.pca_window = 10
                self.ica_window = 10
                self.skew_window = 5
                self.step = 1
                self.detection_thresh_online = 10
                self.skew_thresh_online = 0.5

                self.lambda_val = 0.995
                self.ff = 'cooling'

                online = False
                detect = True
                calibPCA = True

                self.ori = orICA.onlineORICAss(self.recordings, fs=self.fs, onlineWhitening=online, calibratePCA=calibPCA,
                                               forgetfac=self.ff, lambda_0=self.lambda_val,
                                               numpass=1, block=self.block, step_size=self.step,
                                               skew_window=self.skew_window, pca_window=self.pca_window,
                                               ica_window=self.ica_window, verbose=True,
                                               detect_trheshold=10, onlineDetection=False)

                # self.ori = orICA.onlineORICAss(self.recordings, fs=self.fs, forgetfac='cooling',
                #                                skew_thresh=self.skew_thresh_online, lambda_0=0.995,
                #                                verbose=True, block=self.block, step_size=self.step,
                #                                window=self.window, initial_window=self.setting_time,
                #                                detect_trheshold=self.detection_thresh_online)

                self.processing_time = time.time() - t_start_proc

                # last_idxs = find_consistent_sorces(self.ori.source_idx, thresh=0.5)
                last_idxs = ori.all_sources
                last_idxs = last_idxs[np.argsort(np.abs(stats.skew(self.ori.y[last_idxs], axis=1)))[::-1]]
                n_id = len(last_idxs)

                y_on_selected = self.ori.y_on[last_idxs]
                self.y_on = np.array(
                    [-np.sign(sk) * s for (sk, s) in zip(stats.skew(y_on_selected, axis=1), y_on_selected)])

                print('Rough spike detection to choose thresholds')
                self.detected_spikes = detect_and_align(self.y_on, self.fs, self.recordings, n_std=8,
                                                        ref_period=2*pq.ms, upsample=1)

                # Manual thresholding
                nsamples = int((5 * self.fs.rescale('Hz')).magnitude)
                self.thresholds = []
                for i, st in enumerate(self.detected_spikes):
                    fig = plt.figure(figsize=(10,10))
                    # only plot 10% of the spikes
                    perm = np.random.permutation(len(st))
                    nperm = int(0.2 * len(st))
                    plt.plot(st.annotations['ica_wf'][perm[:nperm]].T, lw=0.1, color='g')
                    plt.title('IC  ' + str(i + 1))
                    coord = plt.ginput()
                    th = coord[0][1]
                    plt.close(fig)
                    self.thresholds.append(th)

                ## User defined thresholds and detection
                print('Detecting spikes based on user-defined thresholds')
                self.spikes = threshold_spike_sorting(self.y_on, self.thresholds)

                # Convert spikes to neo
                self.spike_trains = []
                for i, k in enumerate(np.sort(self.spikes.keys())):
                    t = self.spikes[k]
                    tt = t / self.fs
                    st = neo.SpikeTrain(times=tt, t_start=(self.pca_window + self.ica_window) * pq.s, t_stop=self.t_stop)
                    st.annotate(ica_source=last_idxs[i])
                    self.spike_trains.append(st)

                self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)

                print('Number of spiky sources: ', len(self.spike_trains))
                print('Number of spike trains after duplicate rejection: ', len(self.sst))

                self.oorica_spike_sources_idx = last_idxs[self.independent_spike_idx]
                self.oorica_spike_sources = self.ori.y[self.oorica_spike_sources_idx]
                self.oorA_spike_sources = self.ori.mixing[-1, self.oorica_spike_sources_idx]
                self.oorW_spike_sources = self.ori.unmixing[-1, self.oorica_spike_sources_idx]

                if 'ica_wf' not in self.sst[0].annotations.keys():
                    extract_wf(self.sst, self.recordings, self.times, self.fs, ica=True, sources=self.oorica_spike_sources)

                print('Elapsed time: ', time.time() - t_start)

                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst, t_start=self.setting_time)
                # print(self.pairs
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('ORICA', fontsize=20)

                # save results
                if not os.path.isdir(join(self.orica_online_folder, 'results')):
                    os.makedirs(join(self.orica_online_folder, 'results'))
                if save:
                    np.save(join(self.orica_online_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.orica_online_folder, 'results', 'y_on'), self.y_on)
                    np.save(join(self.orica_online_folder, 'results', 'y'), self.oorica_spike_sources)
                    np.save(join(self.orica_online_folder, 'results', 'mixing'), self.oorA_spike_sources)
                    np.save(join(self.orica_online_folder, 'results', 'time'), self.processing_time)

                    with open(join(self.orica_online_folder, 'orica_params.yaml'), 'w') as f:
                        orica_par = {'pca_window': self.pca_window, 'ica_window': self.ica_window,
                                     'skew_window': self.skew_window, 'step': self.step,
                                     'skew_thresh': self.skew_thresh_online, 'lambda_val': self.lambda_val,
                                     'ff': self.ff}
                        yaml.dump(orica_par, f)


        if self.threshold_online:
            if self.run_ss:
                print('Applying threshold online')
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
                    st = neo.SpikeTrain(times=times, t_start=0*pq.s, t_stop = self.t_stop)
                    st.annotate(electrode=k)
                    self.sst.append(st)

                # extract_wf(self.sst, self.oorica_spike_sources, self.recordings, self.times, self.fs)

                print('Elapsed time: ', time.time() - t_start)

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print(self.pairs)

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
                print('Applying Convolutive Online Recursive ICA')
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
                print('Number of cleaned sources: ', self.cleaned_sources_orica.shape[0])


                print('Clustering Sources with: ', self.clustering)
                if self.clustering=='kmeans' or self.clustering=='mog':
                    # detect spikes and align
                    self.detected_spikes = detect_and_align(spike_sources, self.fs, self.recordings,
                                                            t_start=self.t_start,
                                                            t_stop=self.t_stop, n_std=self.threshold)
                    self.spike_amps = [sp.annotations['ica_amp'] for sp in self.detected_spikes]

                    self.spike_trains, self.amps, self.nclusters, self.keep, self.score = \
                        cluster_spike_amplitudes(self.detected_spikes, metric='cal',
                                                 alg=self.clustering)

                    self.spike_trains_rej, self.independent_spike_idx, self.dup = \
                        reject_duplicate_spiketrains(self.spike_trains)
                else:
                    self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                         t_start=self.t_start,
                                                         t_stop=self.t_stop)
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

                print('Elapsed time: ', time.time() - t_start)

                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print(self.pairs)

                self.performance =  compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('convORICA', fontsize=20)


        if self.klusta:
            print('Applying Klustakwik algorithm')
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
                    print('Elapsed time: ', self.processing_time)
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

                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print('PAIRS: '
                # print(self.pairs
                #
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('KLUSTA', fontsize=20)

                print('Elapsed time: ', time.time() - t_start)

                # save results
                if not os.path.isdir(join(self.klusta_folder, 'results')):
                    os.makedirs(join(self.klusta_folder, 'results'))
                if save:
                    np.save(join(self.klusta_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.klusta_folder, 'results', 'time'), self.processing_time)
                # np.save(join(self.klusta_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.klusta_folder, 'results', 'performance'), self.performance)
                # np.save(join(self.klusta_folder, 'results', 'time'), self.processing_time)


        if self.kilo:
            print('Applying Kilosort algorithm')

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
                    print('Elapsed time: ', time.time() - t_start_proc)
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                os.chdir(self.root)

                print('Parsing output files...')
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
                        spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                        self.spike_trains.append(spiketrain)

                self.spike_templates = np.array(self.spike_templates)

                # print('Finding independent spiketrains...'
                # self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                # print('Found ', len(self.sst), ' independent spiketrains!'
                self.sst = self.spike_trains

                # print('Evaluating spiketrains...'
                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print('PAIRS: '
                # print(self.pairs
                #
                # self.performance = compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('KILOSORT', fontsize=20)

                print('Total elapsed time: ', time.time() - t_start)

                # save results
                if not os.path.isdir(join(self.kilo_folder, 'results')):
                    os.makedirs(join(self.kilo_folder, 'results'))
                if save:
                    np.save(join(self.kilo_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.kilo_folder, 'results', 'time'), self.processing_time)

                # np.save(join(self.kilo_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.kilo_folder, 'results', 'performance'), self.performance)
                # np.save(join(self.kilo_folder, 'results', 'time'), self.processing_time)


        if self.mountain:
            print('Applying Mountainsort algorithm')

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
            print('saving ', filename)
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
                    print('Elapsed time: ', self.processing_time)
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                os.chdir(self.root)

                print('Parsing output files...')
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
                        spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                        self.spike_trains.append(spiketrain)

                # print('Finding independent spiketrains...'
                # self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                self.sst = self.spike_trains
                print('Found ', len(self.sst), ' independent spiketrains!')

                print('Extracting waveforms')
                extract_wf(self.sst, self.recordings, self.times, self.fs)

                # print('Evaluating spiketrains...'
                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print('PAIRS: '
                # print(self.pairs
                #
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('MOUNTAINSORT', fontsize=20)

                print('Total elapsed time: ', time.time() - t_start)

                # save results
                if not os.path.isdir(join(self.mountain_folder, 'results')):
                    os.makedirs(join(self.mountain_folder, 'results'))
                if save:
                    np.save(join(self.mountain_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.mountain_folder, 'results', 'time'), self.processing_time)
                # np.save(join(self.mountain_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.mountain_folder, 'results', 'performance'), self.performance)
                # np.save(join(self.mountain_folder, 'results', 'time'), self.processing_time)

        if self.circus:
            print('Applying Spyking-circus algorithm')
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
                    print('Elapsed time: ', self.processing_time)
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                print('Parsing output files...')
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
                        spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                        self.spike_trains.append(spiketrain)
                    else:
                        print('Discarded spike train ', i_st)

                self.sst = self.spike_trains
                print('Found ', len(self.sst), ' independent spiketrains!')

                # print('Evaluating spiketrains...'
                # self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)
                #
                # print('PAIRS: '
                # print(self.pairs
                #
                # self.performance =  compute_performance(self.counts)
                #
                # if self.plot_figures:
                #     ax1, ax2 = self.plot_results()
                #     ax1.set_title('SPYKING-CIRCUS', fontsize=20)

                print('Elapsed time: ', time.time() - t_start)

                # save results
                if not os.path.isdir(join(self.spykingcircus_folder, 'results')):
                    os.makedirs(join(self.spykingcircus_folder, 'results'))
                if save:
                    np.save(join(self.spykingcircus_folder, 'results', 'sst'), self.sst)
                    np.save(join(self.spykingcircus_folder, 'results', 'time'), self.processing_time)

                # np.save(join(self.spykingcircus_folder, 'results', 'counts'), self.counts)
                # np.save(join(self.spykingcircus_folder, 'results','performance'), self.performance)
                # np.save(join(self.spykingcircus_folder, 'results', 'time'), self.processing_time)


        if self.yass:
            print('Applying YASS algorithm')
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
                    print('Elapsed time: ', self.processing_time)
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                print('Parsing output files...')
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
                        spiketrain = neo.SpikeTrain(spike_times, t_start=self.t_start, t_stop=self.t_stop)
                        self.spike_trains.append(spiketrain)

                self.sst = self.spike_trains

                print('Evaluating spiketrains...')
                self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst)

                print('PAIRS: ')
                print(self.pairs)

                self.performance = compute_performance(self.counts)

                if self.plot_figures:
                    ax1, ax2 = self.plot_results()
                    ax1.set_title('YASS', fontsize=20)

                print('Elapsed time: ', time.time() - t_start)

                # save results
                if not os.path.isdir(join(self.yass_folder, 'results')):
                    os.makedirs(join(self.yass_folder, 'results'))
                np.save(join(self.yass_folder, 'results', 'counts'), self.counts)
                np.save(join(self.yass_folder, 'results', 'performance'), self.performance)
                np.save(join(self.yass_folder, 'results', 'time'), self.processing_time)

        print('DONE')

        if eval:
            print('Matching spike trains')
            self.counts, self.pairs = evaluate_spiketrains(self.gtst, self.sst, t_jitt=3 * pq.ms)

            print('PAIRS: ')
            print(self.pairs)

            print('Computing performance')
            self.performance = compute_performance(self.counts)

            print('Calculating confusion matrix')
            self.conf = confusion_matrix(self.gtst, self.sst, self.pairs[:, 1])

            # print('Matching only identified spike trains'
            # pairs_gt_red = pairs[:, 0][np.where(pairs[:, 0] != -1)]
            # gtst_id = np.array(gtst_red)[pairs_gt_red]
            # counts_id, pairs_id = evaluate_spiketrains(gtst_id, sst, t_jitt=3 * pq.ms)
            #
            # print('Computing performance on only identified spike trains'
            # performance_id = compute_performance(counts_id)
            #
            # print('Calculating confusion matrix'
            # conf_id = confusion_matrix(gtst_id, sst, pairs_id[:, 1])

            if plot_figures:
                ax1, ax2 = plot_matched_raster(self.gtst, self.sst, self.pairs)


    def plot_results(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
        raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

        fig.tight_layout()
        return ax1, ax2


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
        feat = 'amp'
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
    if '-eval' in sys.argv:
        eval = True
    else:
        eval = False

    debug = False
    if debug:
        rec_folder='recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_physrot_Neuronexus-32-cut-30_10_' \
                   '10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_10-05-2018:11:37_3002/'
        block=100
        mod='picard'
        feat='zca'
        keepall=True

    if len(sys.argv) == 1 and not debug:
        print('Arguments: \n   -r recording filename\n   -mod ICA - orica - oricaonline - threshonline - klusta' \
              '- kilosort - mountainsort - spykingcircus  -yass\n   -dur duration in s\n   -tstart start time in s\n' \
              '   -tstop stop time in s\n   -M   number of dimensions\n   -thresh threshold for spike detection\n' \
              '   -block ORICA block size\n   -feat amp|pca feature to use for clustering\n   -clust mog|kmeans ' \
              'clustering algorithm\n   -nokeep only keep largest cluster')

    elif '-r' not in sys.argv and not debug:
        raise AttributeError('Provide model folder for data')
    else:
        sps = SpikeSorter(save=save, rec_folder=rec_folder, alg=mod, duration=dur,
                          tstart=tstart, tstop=tstop, run_ss=spikesort, plot_figures=plot_figures,
                          merge_spikes=merge_spikes, mu=mu, eta=eta, npass=npass, block=block, feat=feat,
                          clust=clust, keepall=keepall, ndim=ndim, eval=eval)