import time

import neo
import numpy as np
import quantities as pq
import scipy.stats as stats
import spyica.orica as orica
import spyica.SpyICASorter as ss

from spikeinterface import BaseRecording
from .tools import detect_and_align, reject_duplicate_spiketrains, \
    threshold_spike_sorting, find_consistent_sorces


def ica_spike_sorting(recording, clustering='mog', n_comp='all',
                      features='amp', skew_thresh=0.2, kurt_thresh=1,
                      n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
                      keep_all_clusters=False, verbose=True):
    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    traces = recording.get_traces().astype(dtype).T

    t_init = time.time()
    cleaned_sources_ica, cleaned_A_ica, cleaned_W_ica, source_idx = \
        ss.clean_ica(traces, n_comp, t_init, n_chunks=n_chunks,
                     chunk_size=chunk_size, kurt_thresh=kurt_thresh,
                     skew_thresh=skew_thresh, verbose=verbose)

    sst, independent_spike_idx = ss.clustering(traces, fs, cleaned_sources_ica, recording.get_num_frames(0),
                                               clustering, spike_thresh, keep_all_clusters, features, verbose)

    if 'ica_source' in sst[0].annotations.keys():
        independent_spike_idx = [s.annotations['ica_source'] for s in sst]

    ica_spike_sources_idx = source_idx[independent_spike_idx]
    ica_spike_sources = cleaned_sources_ica[independent_spike_idx]
    A_spike_sources = cleaned_A_ica[independent_spike_idx]
    W_spike_sources = cleaned_W_ica[independent_spike_idx]

    processing_time = time.time() - t_init
    if verbose:
        print('Elapsed time: ', processing_time)

    sorting = ss.set_times_labels(sst, fs)

    # TODO add spike properties and features

    return sorting  # , cleaned_sources_ica


def orica_spike_sorting(recording, clustering='mog', n_comp='all',
                        features='amp', skew_thresh=0.2, kurt_thresh=1,
                        n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
                        keep_all_clusters=False, block_size=800, ff='cooling', num_pass=1,
                        verbose=True):
    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    traces = recording.get_traces().astype(dtype).T

    t_init = time.time()

    cleaned_sources_orica, cleaned_A_orica, cleaned_W_orica, source_idx = \
        ss.clean_ica(traces, n_comp, t_init, ica_alg='orica', n_chunks=n_chunks,
                     chunk_size=chunk_size, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh,
                     num_pass=num_pass, block_size=block_size, verbose=verbose)

    sst, independent_spike_idx = ss.clustering(traces, fs, cleaned_sources_orica, recording.get_num_frames(0),
                                               clustering, spike_thresh, keep_all_clusters, features, verbose)

    if 'ica_source' in sst[0].annotations.keys():
        independent_spike_idx = [s.annotations['ica_source'] for s in sst]

    ica_spike_sources_idx = source_idx[independent_spike_idx]
    ica_spike_sources = cleaned_sources_orica[independent_spike_idx]
    A_spike_sources = cleaned_A_orica[independent_spike_idx]
    W_spike_sources = cleaned_W_orica[independent_spike_idx]

    processing_time = time.time() - t_init
    if verbose:
        print('Elapsed time: ', processing_time)

    sorting = ss.set_times_labels(sst, fs)

    return sorting  # , ica_spike_sources


def online_orica_spike_sorting(recording, n_comp='all', pca_window=0, ica_window=0, skew_window=5, step=1,
                               skew_thresh=0.5,
                               online=False, detect=True, calibPCA=True, ff='cooling', lambda_val=0.995, dtype='int16',
                               verbose=True, detect_thresh=10, white_mode='pca', pca_block=2000,
                               ica_block=800):
    import matplotlib.pylab as plt

    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()

    traces = recording.get_traces().astype(dtype).T

    t_init = time.time()

    ori = orica.onlineORICAss(traces, fs=fs, onlineWhitening=online, calibratePCA=calibPCA, ndim=n_comp,
                              forgetfac=ff, lambda_0=lambda_val, numpass=1, step_size=step,
                              skew_window=skew_window, pca_window=pca_window, ica_window=ica_window, verbose=True,
                              detect_trheshold=detect_thresh, onlineDetection=False, white_mode=white_mode,
                              pca_block=pca_block, ica_block=ica_block)
    if verbose:
        t_orica_online = time.time() - t_init
        print('Online ORICA completed in: ', t_orica_online)

    last_idxs = find_consistent_sorces(ori.source_idx, thresh=0.5)
    # last_idxs = ori.all_sources
    last_idxs = last_idxs[np.argsort(np.abs(stats.skew(ori.y[last_idxs], axis=1)))[::-1]]
    n_id = len(last_idxs)

    y_on_selected = ori.y_on[last_idxs]
    y_on = np.array(
        [-np.sign(sk) * s for (sk, s) in zip(stats.skew(y_on_selected, axis=1), y_on_selected)])

    if verbose:
        print('Rough spike detection to choose thresholds')
    t_start = 0 * pq.s
    t_stop = recording.get_num_frames() / float(fs) * pq.s
    detected_spikes = detect_and_align(y_on, fs, traces, t_start=t_start, t_stop=t_stop, n_std=5, upsample=1)

    # Manual thresholding
    thresholds = []
    for i, st in enumerate(detected_spikes):
        fig = plt.figure(figsize=(10, 10))
        # only plot 10% of the spikes
        perm = np.random.permutation(len(st))
        nperm = int(0.2 * len(st))
        plt.plot(st.annotations['ica_wf'][perm[:nperm]].T, lw=0.1, color='g')
        plt.title('IC  ' + str(i + 1))
        coord = plt.ginput()
        th = coord[0][1]
        plt.close(fig)
        thresholds.append(th)

    ## User defined thresholds and detection
    if verbose:
        print('Detecting spikes based on user-defined thresholds')
    spikes = threshold_spike_sorting(y_on, thresholds)

    # Convert spikes to neo
    spike_trains = []
    for i, k in enumerate(sorted(spikes.keys())):
        times = spikes[k]
        tt = np.array(times) / float(fs)
        # st = neo.SpikeTrain(times=tt, t_start=(pca_window + ica_window) * pq.s, t_stop=t_stop)
        st = neo.SpikeTrain(times=tt * pq.s, t_start=t_start, t_stop=t_stop)
        st.annotate(ica_source=last_idxs[i])
        spike_trains.append(st)

    sst, independent_spike_idx, dup = reject_duplicate_spiketrains(spike_trains)

    if verbose:
        print('Number of spiky sources: ', len(spike_trains))
        print('Number of spike trains after duplicate rejection: ', len(sst))

    ica_spike_sources_idx = last_idxs[independent_spike_idx]
    ica_spike_sources = ori.y[ica_spike_sources_idx]
    A_spike_sources = ori.mixing[-1, ica_spike_sources_idx]
    W_spike_sources = ori.unmixing[-1, ica_spike_sources_idx]

    if verbose:
        print('Elapsed time: ', time.time() - t_init)

    sorting = ss.set_times_labels(sst, fs)

    return sorting  # , ica_spike_sources


def ica_alg(recording, clustering='mog', n_comp='all',
            features='amp', skew_thresh=0.2, kurt_thresh=1,
            n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
            keep_all_clusters=False, verbose=True):
    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()

    fs = recording.get_sampling_frequency()
    traces = recording.get_traces().astype(dtype).T
    t_init = time.time()

    cleaned_sources_ica, cleaned_A_ica, cleaned_W_ica, source_idx = \
        ss.clean_ica(traces, n_comp, t_init, n_chunks=n_chunks,
                     chunk_size=chunk_size, kurt_thresh=kurt_thresh,
                     skew_thresh=skew_thresh, verbose=verbose)

    sst, independent_spike_idx = ss.clustering(traces, fs, cleaned_sources_ica, recording.get_num_frames(0),
                                               clustering, spike_thresh, keep_all_clusters, features, verbose)

    if 'ica_source' in sst[0].annotations.keys():
        independent_spike_idx = [s.annotations['ica_source'] for s in sst]

    ica_spike_sources_idx = source_idx[independent_spike_idx]
    ica_spike_sources = cleaned_sources_ica[independent_spike_idx]
    A_spike_sources = cleaned_A_ica[independent_spike_idx]
    W_spike_sources = cleaned_W_ica[independent_spike_idx]

    return ica_spike_sources, A_spike_sources, W_spike_sources


def orica_alg(recording, n_comp='all',
              dtype='int16', block_size=800, ff='cooling', num_pass=1,
              verbose=True):
    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()

    if verbose:
        print('Applying Online Recursive ICA')
        t_init = time.time()
    traces = recording.get_traces().astype(dtype).T
    ori = orica.ORICA(traces, ndim=n_comp, block_size=block_size, numpass=num_pass,
                      forgetfac=ff, verbose=verbose)
    if verbose:
        t_orica = time.time() - t_init
        print('ORICA completed in: ', t_orica)

    return ori


def online_orica_alg(recording, n_comp='all', pca_window=10, ica_window=10, skew_window=5, step=1, skew_thresh=0.5,
                     online=False, detect=True, calibPCA=True, ff='cooling', lambda_val=0.995, dtype='int16',
                     verbose=True, detect_thresh=10, pca_block=3000, ica_block=1000):
    if not isinstance(recording, BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying Online ORICA spike sorting')
        t_init = time.time()

    traces = recording.get_traces().astype(dtype).T
    ori = orica.onlineORICAss(traces, fs=fs, onlineWhitening=online, calibratePCA=calibPCA, ndim=n_comp,
                              forgetfac=ff, lambda_0=lambda_val, numpass=1, pca_block=pca_block, ica_block=ica_block,
                              step_size=step,
                              skew_window=skew_window, pca_window=pca_window, ica_window=ica_window, verbose=True,
                              detect_trheshold=detect_thresh, onlineDetection=False)

    if verbose:
        t_orica_online = time.time() - t_init
        print('Online ORICA completed in: ', t_orica_online)

    return ori
