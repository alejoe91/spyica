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

import spiketrain_generator as stg
from tools import *
from plot_spikeMEA import *
import ICA as ica
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
    def __init__(self, save=False, rec_folder=None, alg=None):
        self.rec_folder = rec_folder
        self.model = alg
        self.corr_thresh = 0.5
        self.skew_thresh = 1

        self.clustering = 'kmeans'
        self.alg = 'kmeans'

        self.gtst = np.load(join(self.rec_folder, 'spiketrains.npy'))
        self.recordings = np.load(join(self.rec_folder, 'recordings.npy'))
        self.templates = np.load(join(self.rec_folder, 'templates.npy'))

        rec_info = [f for f in os.listdir(self.rec_folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(self.rec_folder, rec_info), 'r') as f:
            self.info = yaml.load(f)

        self.electrode_name = self.info['General']['electrode name']
        self.fs = self.info['General']['fs']

        self.times = range(self.recordings.shape[1])/self.fs

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

        self.ica = False
        self.cica = False
        self.gfica = False
        self.sfa = False
        self.klusta = False
        if alg == 'all':
            self.ica = True
            self.cica = True
            self.gfica = True
            self.sfa = True
            self.klusta = True
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
            if 'klusta' in alg_split:
                self.klusta = True

        if self.ica:
            print 'Applying instantaneous ICA'
            t_start = time.time()
            self.s_ica, A_ica, W_ica = ica.instICA(self.recordings)

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
            if self.clustering=='kmeans':
                # detect spikes and align
                self.spike_trains = detect_and_align(spike_sources, self.fs, self.recordings,
                                                     t_start=self.gtst[0].t_start,
                                                     t_stop=self.gtst[0].t_stop)
                self.spike_amps = [sp.annotations['ica_amp'] for sp in self.spike_trains]

                self.sst, self.amps, self.nclusters, keep, score = \
                    cluster_spike_amplitudes(self.spike_amps, self.spike_trains, metric='cal', alg=self.alg)

                self.sst, self.independent_spike_idx, self.duplicates = reject_duplicate_spiketrains(self.sst)
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
                self.spike_trains, self.independent_spike_idx = reject_duplicate_spiketrains(self.spike_trains)
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

                sst_order = self.pairs[:, 1]

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('ICA GT', fontsize=20)
                ax2.set_title('ICA ST', fontsize=20)


        if self.cica:
            print 'Applying convolutive embedded ICA'
            t_start = time.time()
            self.s_cica, A_cica, W_cica = ica.cICAemb(self.recordings, L=2)
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
                raster_plots(np.array(self.sst)[sst_order], color_st=self.pairs[:, 0], ax=ax2)

                ax1.set_title('convICA')

        if self.gfica:
            print 'Applying gradient-flow ICA'
            t_start = time.time()
            self.s_gfica, A_gf, W_gf = ica.gFICA(self.recordings, self.mea_dim)
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
                raster_plots(np.array(self.sst)[sst_order], color_st=self.pairs[:, 0], ax=ax2)

                ax1.set_title('gfICA')

        if self.sfa:
            print 'Applying Slow Features Analysis'
            import mdp
            t_start = time.time()
            input_dim =  self.recordings.shape[0]
            whitening_output_dim = self.recordings.shape[0]
            output_dim = self.recordings.shape[0]

            self.node = mdp.nodes.SFANode()
            self.node.train(np.transpose(self.recordings))
            self.sf = np.transpose(self.node.execute(np.transpose(self.recordings)))
            # iterval = 5
            #
            # self.node = IncSFANode(input_dim, whitening_output_dim, output_dim, eps=0.05)
            # trainer = TrainerNode(self.node, mode='Incremental', ticker=100)
            # trainer.train(np.transpose(self.recordings), iterval=iterval, monitor_keys=['slowFeatures'])
            # v = trainer.monitorVar['slowFeatures']
            # self.sf = np.transpose(self.node.execute(np.transpose(self.recordings)))
            print 'Elapsed time: ', time.time() - t_start


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
                raise Excaption('No kwik file!')

            self.counts, self.pairs, self.cc_matr = evaluate_spiketrains(self.gtst, self.sst)
            if plot_cc:
                plt.figure()
                plt.imshow(self.cc_matr)
            print self.pairs

            print 'PERFORMANCE: \n'
            print '\nTP: ', float(self.counts['TP']) / self.counts['TOT'] * 100, ' %'
            print 'TPO: ', float(self.counts['TPO']) / self.counts['TOT'] * 100, ' %'
            print 'TPSO: ', float(self.counts['TPSO']) / self.counts['TOT'] * 100, ' %'
            print 'TOT TP: ', float(self.counts['TP'] + self.counts['TPO'] + self.counts['TPSO']) / \
                              self.counts['TOT'] * 100, ' %'

            print '\nCL: ', float(self.counts['CL']) / self.counts['TOT'] * 100, ' %'
            print 'CLO: ', float(self.counts['CLO']) / self.counts['TOT'] * 100, ' %'
            print 'CLSO: ', float(self.counts['CLSO']) / self.counts['TOT'] * 100, ' %'
            print 'TOT CL: ', float(self.counts['CL'] + self.counts['CLO'] + self.counts['CLSO']) / \
                              self.counts['TOT'] * 100, ' %'

            print '\nFN: ', float(self.counts['FN']) / self.counts['TOT'] * 100, ' %'
            print 'FNO: ', float(self.counts['FNO']) / self.counts['TOT'] * 100, ' %'
            print 'FNSO: ', float(self.counts['FNSO']) / self.counts['TOT'] * 100, ' %'
            print 'TOT FN: ', float(self.counts['FN'] + self.counts['FNO'] + self.counts['FNSO']) / \
                              self.counts['TOT'] * 100, ' %'

            print '\nTOT FP: ', float(self.counts['FP']) / self.counts['TOT'] * 100, ' %'

            if plot_rasters:
                fig = plt.figure()
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)

                raster_plots(self.gtst, color_st=self.pairs[:, 0], ax=ax1)
                raster_plots(self.sst, color_st=self.pairs[:, 1], ax=ax2)

                ax1.set_title('KLUSTA')

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

