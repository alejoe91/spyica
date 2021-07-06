import time

import neo
import numpy as np
import quantities as pq
import scipy.stats as stats
import spikeinterface as si
import spyica.ica as ica
import spyica.orica as orica

from .tools import clean_sources, cluster_spike_amplitudes, detect_and_align, \
    reject_duplicate_spiketrains, threshold_spike_sorting, find_consistent_sorces


def ica_spike_sorting(recording, clustering='mog', n_comp='all',
                      features='amp', skew_thresh=0.2, kurt_thresh=1,
                      n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
                      keep_all_clusters=False, verbose=True):
    if not isinstance(recording, si.BaseRecording):
        raise Exception("Input a RecordingExtractor object!")

    # TODO use random snippets (e.g. 20% of the data) / or spiky signals for fast ICA

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying FastICA algorithm')
        t_init = time.time()
    traces = recording.get_traces().astype(dtype)
    s_ica, A_ica, W_ica = ica.instICA(traces, n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)
    if verbose:
        t_ica = time.time() - t_init
        print('FastICA completed in: ', t_ica)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    if verbose:
        print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])
        print('Clustering Sources with: ', clustering)

    t_start = 0 * pq.s
    t_stop = recording.get_num_frames(0) / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, traces,
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)
        if verbose:
            print('Number of spike trains after clustering: ', len(spike_trains))
        sst, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        if verbose:
            print('Number of spike trains after duplicate rejection: ', len(sst))
    else:
        raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

    if 'ica_source' in sst[0].annotations.keys():
        independent_spike_idx = [s.annotations['ica_source'] for s in sst]

    ica_spike_sources_idx = source_idx[independent_spike_idx]
    ica_spike_sources = cleaned_sources_ica[independent_spike_idx]
    A_spike_sources = cleaned_A_ica[independent_spike_idx]
    W_spike_sources = cleaned_W_ica[independent_spike_idx]

    processing_time = time.time() - t_init
    if verbose:
        print('Elapsed time: ', processing_time)

    times = np.array([], dtype=int)
    labels = np.array([])
    for i_s, st in enumerate(sst):
        times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
        labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

    sorting = si.NumpySorting.from_times_labels(times.astype(int), labels, fs)

    # TODO add spike properties and features

    return sorting #, cleaned_sources_ica


def orica_spike_sorting(recording, clustering='mog', n_comp='all',
                        features='amp', skew_thresh=0.2, kurt_thresh=1,
                        n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
                        keep_all_clusters=False, block_size=800, ff='cooling', num_pass=1,
                        verbose=True):
    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying Online Recursive ICA')
        t_init = time.time()
    traces = recording.get_traces().astype(dtype)
    s_orica, A_orica, W_orica = orica.instICA(traces, n_comp=n_comp,
                                              n_chunks=n_chunks, chunk_size=chunk_size,
                                              numpass=num_pass, block_size=block_size)
    if verbose:
        t_orica = time.time() - t_init
        print('ORICA completed in: ', t_orica)

    # clean sources based on skewness and correlation
    cleaned_sources_orica, source_idx = clean_sources(s_orica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_orica = A_orica[source_idx]
    cleaned_W_orica = W_orica[source_idx]

    if verbose:
        print('Number of cleaned sources: ', cleaned_sources_orica.shape[0])
        print('Clustering Sources with: ', clustering)

    t_start = 0 * pq.s
    t_stop = recording.get_num_frames() / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_orica, fs, traces,
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)
        if verbose:
            print('Number of spike trains after clustering: ', len(spike_trains))
        sst, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_orica)
        if verbose:
            print('Number of spike trains after duplicate rejection: ', len(sst))
    else:
        raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

    if 'ica_source' in sst[0].annotations.keys():
        independent_spike_idx = [s.annotations['ica_source'] for s in sst]

    ica_spike_sources_idx = source_idx[independent_spike_idx]
    ica_spike_sources = cleaned_sources_orica[independent_spike_idx]
    A_spike_sources = cleaned_A_orica[independent_spike_idx]
    W_spike_sources = cleaned_W_orica[independent_spike_idx]

    processing_time = time.time() - t_init
    if verbose:
        print('Elapsed time: ', processing_time)

    sorting = se.NumpySortingExtractor()
    times = np.array([])
    labels = np.array([])
    for i_s, st in enumerate(sst):
        times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
        labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

    sorting.set_times_labels(times, labels)
    sorting.set_sampling_frequency(recording.get_sampling_frequency)

    return sorting #, ica_spike_sources


def online_orica_spike_sorting(recording, n_comp='all', pca_window=0, ica_window=0, skew_window=5, step=1,
                               skew_thresh=0.5,
                               online=False, detect=True, calibPCA=True, ff='cooling', lambda_val=0.995, dtype='int16',
                               verbose=True, detect_thresh=10, white_mode='pca', pca_block=2000,
                               ica_block=800):
    import matplotlib.pylab as plt

    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying Online ORICA spike sorting')
        t_init = time.time()
    fs = recording.get_sampling_frequency()
    traces = recording.get_traces().astype(dtype)
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

    sorting = se.NumpySortingExtractor()
    times = np.array([])
    labels = np.array([])
    for i_s, st in enumerate(sst):
        times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
        labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

    sorting.set_times_labels(times, labels)
    sorting.set_sampling_frequency(recording.get_sampling_frequency)

    return sorting #, ica_spike_sources


def ica_alg(recording, fs, clustering='mog', n_comp='all',
            features='amp', skew_thresh=0.2, kurt_thresh=1,
            n_chunks=0, chunk_size=0, spike_thresh=5, dtype='int16',
            keep_all_clusters=False, verbose=True):
    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    if verbose:
        print('Applying FastICA')
        t_init = time.time()
    traces = recording.get_traces().astype(dtype)
    s_ica, A_ica, W_ica = ica.instICA(traces, n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)
    if verbose:
        t_ica = time.time() - t_init
        print('FastICA completed in: ', t_ica)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    if verbose:
        print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])
        print('Clustering Sources with: ', clustering)

    t_start = 0 * pq.s
    t_stop = recording.get_num_frames(0) / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, traces,
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)
        if verbose:
            print('Number of spike trains after clustering: ', len(spike_trains))
        sst, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        if verbose:
            print('Number of spike trains after duplicate rejection: ', len(sst))
    else:
        raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

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
    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying Online Recursive ICA')
        t_init = time.time()
    traces = recording.get_traces().astype(dtype)
    ori = orica.ORICA(traces, ndim=n_comp, block_size=block_size, numpass=num_pass,
                      forgetfac=ff, verbose=verbose)
    if verbose:
        t_orica = time.time() - t_init
        print('ORICA completed in: ', t_orica)

    return ori


def online_orica_alg(recording, n_comp='all', pca_window=10, ica_window=10, skew_window=5, step=1, skew_thresh=0.5,
                     online=False, detect=True, calibPCA=True, ff='cooling', lambda_val=0.995, dtype='int16',
                     verbose=True, detect_thresh=10, pca_block=3000, ica_block=1000):
    if n_comp == 'all':
        n_comp = recording.get_num_channels()
    fs = recording.get_sampling_frequency()
    if verbose:
        print('Applying Online ORICA spike sorting')
        t_init = time.time()
    fs = recording.get_sampling_frequency()
    traces = recording.get_traces().astype(dtype)
    ori = orica.onlineORICAss(traces, fs=fs, onlineWhitening=online, calibratePCA=calibPCA, ndim=n_comp,
                              forgetfac=ff, lambda_0=lambda_val, numpass=1, pca_block=pca_block, ica_block=ica_block,
                              step_size=step,
                              skew_window=skew_window, pca_window=pca_window, ica_window=ica_window, verbose=True,
                              detect_trheshold=detect_thresh, onlineDetection=False)

    if verbose:
        t_orica_online = time.time() - t_init
        print('Online ORICA completed in: ', t_orica_online)

    return ori
