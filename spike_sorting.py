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
# import quantities as pq
import json
import yaml
import time
import h5py

# import spiketrain_generator as stg
from tools import *
from plot_spikeMEA import *
import ICA as ica
import smoothICA as sICA
from sfa.incsfa import IncSFANode
from sfa.trainer import TrainerNode
# import sfa.incsfa as sfa

root_folder = os.getcwd()

plt.ion()
plt.show()
plot_source = False
plot_cc = False
plot_rasters = True


class SpikeSorter:
    def __init__(self, save=False, rec_folder=None, alg=None, lag=None, gfmode=None, duration=None,
                 tstart=None, tstop=None):
        self.rec_folder = rec_folder
        self.rec_name = os.path.split(rec_folder)[-1]
        if self.rec_name == '':
            split = os.path.split(rec_folder)[0]
            self.rec_name = os.path.split(split)[-1]

        self.duration = duration
        self.tstart = tstart
        self.tstop = tstop

        self.model = alg
        self.corr_thresh = 0.5
        self.skew_thresh = 0.2
        self.root = os.getcwd()
        sys.path.append(self.root)

        self.clustering = 'kmeans'
        self.alg = 'kmeans'

        self.minimum_spikes_per_cluster = 3

        if self.rec_name.startswith('recording'):
            self.gtst = np.load(join(self.rec_folder, 'spiketrains.npy'))
            self.recordings = np.load(join(self.rec_folder, 'recordings.npy'))
            self.templates = np.load(join(self.rec_folder, 'templates.npy'))
            self.templates_cat = np.load(join(self.rec_folder, 'templates_cat.npy'))
            self.templates_loc = np.load(join(self.rec_folder, 'templates_loc.npy'))

            rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
            with open(join(self.rec_folder, rec_info), 'r') as f:
                self.info = yaml.load(f)

            self.electrode_name = self.info['General']['electrode name']
            self.fs = self.info['General']['fs']
            self.times = range(self.recordings.shape[1]) / self.fs

            self.overlapping = self.info['Synchrony']['overlap_pairs']
            if 'overlap' not in self.gtst[0].annotations.keys():
                print 'Finding overlapping spikes'
                annotate_overlapping(self.gtst, overlapping_pairs=self.overlapping)

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

        elif self.rec_name.startswith('savedata'):
            f = h5py.File(join(self.rec_folder, 'ViSAPy_filterstep_1.h5'))
            self.recordings = np.transpose(f['data'])
            self.fs = f['srate'].value * pq.Hz
            self.elec = f['electrode']
            self.times = range(self.recordings.shape[1]) / self.fs

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
                self.tstart = int(self.tstart) * pq.s
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
        self.smooth = False
        self.cica = False
        self.gfica = False
        self.klusta = False
        self.kilo = False
        self.mountain = False

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
            if 'smooth' in alg_split:
                self.smooth = True
            if 'cica' in alg_split:
                self.cica = True
            if 'gfica' in alg_split:
                self.gfica = True
            if 'klusta' in alg_split:
                self.klusta = True
            if 'kilosort' in alg_split:
                self.kilo = True
            if 'mountainsort' in alg_split:
                self.mountain = True

        self.lag = lag
        self.gfmode = gfmode

        if self.ica:
            print 'Applying instantaneous ICA'
            t_start = time.time()
            self.s_ica, self.A_ica, self.W_ica = ica.instICA(self.recordings)

            if plot_source:
                plot_mea_recording(self.s_ica, self.mea_pos, self.mea_dim, color='r')

            # clean sources based on skewness and correlation
            spike_sources, self.source_idx, self.correlated_pairs = clean_sources(self.s_ica,
                                                                                  corr_thresh=self.corr_thresh,
                                                                                  skew_thresh=self.skew_thresh)
            if plot_source:
                plt.figure()
                plt.plot(np.transpose(spike_sources))

            self.cleaned_sources_ica = spike_sources
            print 'Number of cleaned sources: ', self.cleaned_sources_ica.shape[0]

            print 'Clustering Sources with: ', self.clustering

            if self.clustering=='kmeans' or self.clustering=='mog':
                # detect spikes and align
                self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                     t_start=self.gtst[0].t_start,
                                                     t_stop=self.gtst[0].t_stop)
                self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]

                self.sst, self.amps, self.nclusters, keep, score = \
                    cluster_spike_amplitudes(self.spike_amps, self.spike_trains, metric='cal', alg=self.alg)
                self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.sst)

                self.ica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
                self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]

            elif self.clustering=='klusta':

                if not os.path.isdir(join(self.rec_folder, 'ica')):
                    os.makedirs(join(self.rec_folder, 'ica'))
                self.klusta_folder = join(self.rec_folder, 'ica')
                rec_name = os.path.split(self.rec_folder)
                if rec_name[-1] == '':
                    rec_name = os.path.split(rec_name[0])[-1]
                else:
                    rec_name = rec_name[-1]
                self.klusta_full_path = join(self.klusta_folder, rec_name)
                # create prb and prm files
                prb_path = export_prb_file(self.cleaned_sources_ica.shape[0], 'ica', self.klusta_folder,
                                           geometry=False, graph=False, separate_channels=True)
                klusta_prm = create_klusta_prm(self.klusta_full_path, prb_path, nchan=self.cleaned_sources_ica.shape[0],
                                               fs=self.fs, klusta_filter=False)
                # save binary file
                save_binary_format(self.klusta_full_path, self.cleaned_sources_ica, spikesorter='klusta')

                print('Running klusta')

                try:
                    import klusta
                    import klustakwik2
                except ImportError:
                    raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

                import subprocess
                try:
                    subprocess.check_output(['klusta', klusta_prm, '--overwrite'])
                except subprocess.CalledProcessError as e:
                    raise Exception(e.output)

                kwikfile = [f for f in os.listdir(self.klusta_folder) if f.endswith('.kwik')]
                if len(kwikfile) > 0:
                    kwikfile = join(self.klusta_folder, kwikfile[0])
                    if os.path.exists(kwikfile):
                        kwikio = neo.io.KwikIO(filename=kwikfile, )
                        blk = kwikio.read_block(raw_data_units='uV')
                        self.possible_sst = blk.segments[0].spiketrains
                else:
                    raise Excaption('No kwik file!')

                # for each source, keep the spikes cluster with largest amplitude
                # self.sst = []
                # group_ids = np.unique([sp.annotations['group_id'] for sp in self.possible_sst])
                # for id in group_ids:
                #     sst_idx = np.where(np.array([sp.annotations['group_id'] for sp in self.sst]) == id)[0]
                #     if len(sst_idx) > 1:
                #         amps = np.array([np.max(np.abs(np.mean(np.squeeze(sps.sst[id].waveforms), axis=1)))
                #                          for id in sst_idx])
                #         self.sst.append(self.possible_sst[np.argmax(amps)])
                #     else:
                #         self.sst.append(self.possible_sst[sst_idx])

                self.spike_trains, self.independent_spike_idx = reject_duplicate_spiketrains(self.possible_sst)
                self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                # self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                self.sst = self.spike_trains

            else:
                self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                      t_start=self.gtst[0].t_start,
                                                      t_stop=self.gtst[0].t_stop)
                self.spike_trains, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
                self.ica_spike_sources = self.cleaned_sources_ica[self.independent_spike_idx]
                self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
                self.sst = self.spike_trains


            print 'Elapsed time: ', time.time() - t_start

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO'])/\
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_ST'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('ICA', fontsize=20)

        if self.smooth:
            print 'Applying smoothed ICA'
            t_start = time.time()
            self.s_sica, self.A_sica, self.W_sica = sICA.smoothICA(self.recordings)

            if plot_source:
                plot_mea_recording(self.s_sica, self.mea_pos, self.mea_dim, color='r')

            # clean sources based on skewness and correlation
            spike_sources, self.source_idx, self.correlated_pairs = clean_sources(self.s_sica,
                                                                                  corr_thresh=self.corr_thresh,
                                                                                  skew_thresh=self.skew_thresh)
            if plot_source:
                plt.figure()
                plt.plot(np.transpose(spike_sources))

            self.cleaned_sources_sica = spike_sources
            print 'Number of cleaned sources: ', self.cleaned_sources_sica.shape[0]

            # raise Exception()

            # print 'Clustering Sources with: ', self.clustering
            # if self.clustering=='kmeans':
            #     # detect spikes and align
            #     self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
            #                                          t_start=self.gtst[0].t_start,
            #                                          t_stop=self.gtst[0].t_stop)
            #     self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
            #
            #     self.sst, self.amps, self.nclusters, keep, score = \
            #         cluster_spike_amplitudes(self.spike_amps, self.spike_trains, metric='cal', alg=self.alg)
            #     self.sst, self.independent_spike_idx, self.duplicates = reject_duplicate_spiketrains(self.sst)
            #
            #     self.sica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
            #     self.sica_spike_sources = self.cleaned_sources_sica[self.independent_spike_idx]
            # else:
            #     self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
            #                                          t_start=self.gtst[0].t_start,
            #                                          t_stop=self.gtst[0].t_stop)
            #     self.spike_trains, self.independent_spike_idx = reject_duplicate_spiketrains(self.spike_trains)
            #     self.sica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
            #     self.sica_spike_sources = self.cleaned_sources_sica[self.independent_spike_idx]
            #     self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]
            #     self.sst = self.spike_trains
            #
            # print 'Elapsed time: ', time.time() - t_start
            #
            # self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            # if plot_cc:
            #     plt.figure()
            #     plt.imshow(self.cc_matr)
            # print self.pairs
            #
            # print 'PERFORMANCE: \n'
            # print '\nTP: ', float(self.counts['TP']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
            #                   self.counts['TOT_GT'] * 100, ' %'
            #
            # print '\nCL: ', float(self.counts['CL']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
            #                   self.counts['TOT_GT'] * 100, ' %'
            #
            # print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            # print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
            #                   self.counts['TOT_GT'] * 100, ' %'
            #
            # print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_ST'] * 100, ' %'
            #
            # if plot_rasters:
            #     fig = plt.figure()
            #     ax1 = fig.add_subplot(211)
            #     ax2 = fig.add_subplot(212)
            #
            #     raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
            #     raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)
            #
            #     ax1.set_title('ICA GT', fontsize=20)
            #     ax2.set_title('ICA ST', fontsize=20)

        if self.cica:
            print 'Applying convolutive embedded ICA'
            t_start = time.time()
            self.s_cica, self.A_cica, self.W_cica = ica.cICAemb(self.recordings, L=self.lag)
            print 'Elapsed time: ', time.time() - t_start

            if plot_source:
                plot_mea_recording(self.s_cica, self.mea_pos, self.mea_dim, color='r')

            # clean sources based on skewness and correlation
            spike_sources, self.source_idx, self.correlated_pairs = clean_sources(self.s_cica, corr_thresh=self.corr_thresh,
                                                                                  skew_thresh=self.skew_thresh)
            if plot_source:
                plt.figure()
                plt.plot(np.transpose(spike_sources))

            self.cleaned_sources_cica = spike_sources
            print 'Number of cleaned sources: ', self.cleaned_sources_cica.shape[0]

            # detect spikes and align
            self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                 t_start=self.gtst[0].t_start,
                                                 t_stop=self.gtst[0].t_stop)

            self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]

            self.sst, self.amps, self.nclusters, keep, score = \
                cluster_spike_amplitudes(self.spike_amps, self.spike_trains, metric='cal', alg=self.alg)

            self.sst, self.independent_spike_idx, self.duplicates = reject_duplicate_spiketrains(self.sst)
            self.cica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
            self.cica_spike_sources = self.cleaned_sources_cica[self.independent_spike_idx]

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO'])/self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO'])/\
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_ST'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                sst_order = self.pairs[:, 1]

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('cICA GT', fontsize=20)
                ax2.set_title('cICA ST', fontsize=20)


        if self.gfica:
            print 'Applying gradient-flow ICA'
            t_start = time.time()
            self.s_gfica, self.A_gfica, self.W_gfica = ica.gFICA(self.recordings, self.mea_dim, mode=self.gfmode)
            # s_gf_int = integrate_sources(s_gf)
            print 'Elapsed time: ', time.time() - t_start

            # gf_mea = np.reshape(np.reshape(mea_pos, (mea_dim[0], mea_dim[1], mea_pos.shape[1]))[:-1, :-1],
            #                     ((mea_dim[0]-1)*(mea_dim[1]-1), mea_pos.shape[1]))
            # gf_dim = (mea_dim[0]-1, mea_dim[1]-1)
            #
            # print 'Elapsed time: ', time.time() - t_start

            if plot_source:
                plot_mea_recording(self.s_gfica[:n_elec], self.mea_pos, self.mea_dim, color='r')
                plot_mea_recording(self.s_gfica[:n_elec], self.mea_pos, self.mea_dim, color='g')
                plot_mea_recording(self.s_gfica[:n_elec], self.mea_pos, self.mea_dim, color='bhn')

            # clean sources based on skewness and correlation
            spike_sources, self.source_idx, self.correlated_pairs = clean_sources(self.s_gfica,
                                                                                  corr_thresh=self.corr_thresh,
                                                                                  skew_thresh=self.skew_thresh)

            if plot_source:
                plt.figure()
                plt.plot(np.transpose(spike_sources))

            self.cleaned_sources_gfica = spike_sources
            print 'Number of cleaned sources: ', self.cleaned_sources_gfica.shape[0]


            # detect spikes and align
            self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                 t_start=self.gtst[0].t_start,
                                                 t_stop=self.gtst[0].t_stop)

            self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]

            self.sst, self.amps, self.nclusters, keep, score = \
                cluster_spike_amplitudes(self.spike_amps, self.spike_trains, metric='cal', alg=self.alg)

            self.sst, self.independent_spike_idx, self.duplicates = reject_duplicate_spiketrains(self.sst)
            self.gfica_spike_sources_idx = self.source_idx[self.independent_spike_idx]
            self.gfica_spike_sources = self.cleaned_sources_gfica[self.independent_spike_idx]

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_ST'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                sst_order = self.pairs[:, 1]

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('gfICA GT', fontsize=20)
                ax2.set_title('gfICA ST', fontsize=20)


        if self.klusta:
            print 'Applying Klustakwik algorithm'
            if not os.path.isdir(join(self.rec_folder, 'klusta')):
                os.makedirs(join(self.rec_folder, 'klusta'))
            self.klusta_folder = join(self.rec_folder, 'klusta')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            self.klusta_full_path = join(self.klusta_folder, rec_name)
            # create prb and prm files
            prb_path = export_prb_file(self.mea_pos.shape[0], self.electrode_name, self.klusta_folder,
                                       pos=self.mea_pos, adj_dist=2*np.max(self.mea_pitch))
            klusta_prm = create_klusta_prm(self.klusta_full_path, prb_path, nchan=self.recordings.shape[0],
                                           fs=self.fs, klusta_filter=False)
            # save binary file
            save_binary_format(self.klusta_full_path, self.recordings, spikesorter='klusta')

            print('Running klusta')

            try:
                import klusta
                import klustakwik2
            except ImportError:
                raise ImportError('Install klusta and klustakwik2 packages to run klusta option')

            import subprocess
            try:
                t_start = time.time()
                subprocess.check_output(['klusta', klusta_prm, '--overwrite'])
                print 'Elapsed time: ', time.time() - t_start
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

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_GT'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('KLUSTA')

        if self.kilo:
            print 'Applying Kilosort algorithm'

            # setup folders
            if not os.path.isdir(join(self.rec_folder, 'kilosort')):
                os.makedirs(join(self.rec_folder, 'kilosort'))
            self.kilo_folder = join(self.rec_folder, 'kilosort')
            self.kilo_results_folder = join(self.kilo_folder, 'results')
            rec_name = os.path.split(self.rec_folder)
            if rec_name[-1] == '':
                rec_name = os.path.split(rec_name[0])[-1]
            else:
                rec_name = rec_name[-1]
            self.kilo_raw = join(self.kilo_folder, 'raw.dat')

            save_binary_format(self.kilo_raw,
                               (self.recordings / 0.195).astype('int16'),
                               spikesorter='kilosort',
                               dtype='int16')
            threshold = 4.

            # set up kilosort config files and run kilosort on data
            with open(join(self.root, 'kilosort_files',
                                   'kilosort_master.txt'), 'r') as f:
                kilosort_master = f.readlines()
            with open(join(self.root, 'kilosort_files',
                                   'kilosort_config.txt'), 'r') as f:
                kilosort_config = f.readlines()
            with open(join(self.root, 'kilosort_files',
                                   'kilosort_channelmap.txt'), 'r') as f:
                kilosort_channelmap = f.readlines()
            nchan = self.recordings.shape[0]
            dat_file = 'raw.dat'

            kilosort_master = ''.join(kilosort_master).format(
                self.kilo_folder
            )
            kilosort_config = ''.join(kilosort_config).format(
                nchan, nchan, int(self.fs.rescale('Hz')), dat_file, threshold,
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

            print('Running kilosort')

            cwd = os.getcwd()
            os.chdir(self.kilo_folder)
            try:
                t_start = time.time()
                import subprocess
                subprocess.call(['matlab', '-nodisplay', '-nodesktop',
                                 '-nosplash', '-r',
                                 'run kilosort_master.m; exit;'])
                print 'Elapsed time: ', time.time() - t_start
            except subprocess.CalledProcessError as e:
                raise Exception(e.output)

            os.chdir(cwd)

            print 'Parsing output files...'
            self.spike_times = np.load(join(self.kilo_results_folder, 'spike_times.npy'))
            self.spike_clusters = np.load(join(self.kilo_results_folder, 'spike_clusters.npy'))
            self.spike_templates = np.load(join(self.kilo_results_folder, 'templates.npy')).swapaxes(1, 2)

            self.spike_trains = []
            clust_id, n_counts = np.unique(self.spike_clusters, return_counts=True)
            self.kl_times = self.times[self.spike_times.astype(int)]

            self.counts = 0

            for clust, count in zip(clust_id, n_counts):
                if count > self.minimum_spikes_per_cluster:
                    idx = np.where(self.spike_clusters == clust)[0]
                    self.counts += len(idx)
                    spike_times = self.kl_times[idx]
                    spiketrain = neo.SpikeTrain(spike_times, t_start=0 * pq.s, t_stop=self.gtst[0].t_stop)
                    self.spike_trains.append(spiketrain)

            print 'Finding independent spiketrains...'
            self.sst, self.independent_spike_idx, self.dup = reject_duplicate_spiketrains(self.spike_trains)
            print 'Found ', len(self.sst), ' independent spiketrains!'

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_GT'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('KILOSORT')

        if self.mountain:
            print 'Applying Mountainsort algorithm'

            import mlpy
            from shutil import copyfile

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
            mlpy.writemda32(self.recordings, join(self.mountain_folder, 'raw.mda'))

            # write csv probe file
            with open(join(self.mountain_folder, 'geom.csv'), 'w') as f:
                for pos in self.mea_pos:
                    f.write(str(pos[1]))
                    f.write(',')
                    f.write(str(pos[2]))
                    f.write('\n')

            print self.mountain_folder

            # write param filw
            params = {'samplerate': int(self.fs.rescale('Hz').magnitude), 'detect_sign': -1,
                      "adjacency_radius": float(2*np.min(self.mea_pitch)-2)}
            with open(join(self.mountain_folder, 'params.json'), 'w') as f:
                json.dump(params, f)

            # copy mountainsort3.mlp
            copyfile(join(self.root, 'mountainsort_files', 'mountainsort3.mlp'),
                     join(self.mountain_folder, 'mountainsort3.mlp'))

            print('Running mountainsort')
            self.curate=True

            import subprocess
            os.chdir(self.mountain_folder)
            try:
                t_start = time.time()
                if self.curate:
                    subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                             '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json',
                                             '--curate=true'])
                else:
                    subprocess.check_output(['mlp-run', 'mountainsort3.mlp', 'sort', '--raw=raw.mda',
                                             '--geom=geom.csv', '--firings_out=firings.mda', '--_params=params.json'])
                print 'Elapsed time: ', time.time() - t_start
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
            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT_GT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT_GT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT_GT'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)
                #
                # ax1.set_xlim([0, np.max(self.times.rescale('s'))])
                # ax2.set_xlim([0, np.max(self.times.rescale('s'))])

                ax1.set_title('MOUNTAINSORT')

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
    if '-L' in sys.argv:
        pos = sys.argv.index('-L')
        lag = int(sys.argv[pos + 1])
    else:
        lag = 2
    if '-gfmode' in sys.argv:
        pos = sys.argv.index('-gfmode')
        gfmode = sys.argv[pos + 1]
    else:
        gfmode = 'time'

    if len(sys.argv) == 1:
        print 'Arguments: \n   -r recording filename\n   -mod ICA - cICA - smooth - gfICA - SFA - klusta' \
              'kilosort - mountainsort\n   -dur duration in seconds' \
              '-L   cICA lag\n   -gfmode gradient-flow mode (time - space - spacetime)\n  '
              #-bx x boundaries [xmin,xmax]\n   -minamp minimum amplitude\n' \
              # '   -noise uncorrelated-correlated\n   -noiselev level of rms noise in uV\n   -dur duration\n' \
              # '   -fexc freq exc neurons\n   -finh freq inh neurons\n   -nofilter if filter or not\n' \
              # '   -over overlapping spike threshold (0.6)\n   -sync added synchorny rate\n' \
              # '   -nomod no spike amp modulation'
    elif '-r' not in sys.argv:
        raise AttributeError('Provide model folder for data')
    else:
        sps = SpikeSorter(save=True, rec_folder=rec_folder, alg=mod, lag=lag, gfmode=gfmode, duration=dur,
                          tstart=tstart, tstop=tstop)

