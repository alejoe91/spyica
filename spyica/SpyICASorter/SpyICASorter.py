from __future__ import print_function

import time
import quantities as pq
import numpy as np
import spyica.ica as ica
import spyica.orica as orica

from spikeinterface import NumpySorting
from spyica.tools import clean_sources, cluster_spike_amplitudes, detect_and_align, \
    reject_duplicate_spiketrains


def clean_ica(traces, n_comp, t_init, ica_alg='ica', n_chunks=0, chunk_size=0,
              kurt_thresh=1, skew_thresh=0.2, num_pass=1, block_size=800, verbose=True):
    if ica_alg == 'ica' or ica_alg == 'orica':
        if verbose and ica_alg == 'ica':
            print('Applying FastICA algorithm')
        elif verbose and ica_alg == 'orica':
            print('Applying offline ORICA')
    else:
        raise Exception("Only 'ica' and 'orica' are implemented")

    # TODO use random snippets (e.g. 20% of the data) / or spiky signals for fast ICA
    if ica_alg == 'ica':
        s_ica, A_ica, W_ica = ica.instICA(traces, n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)
    else:
        s_ica, A_ica, W_ica = orica.instICA(traces, n_comp=n_comp,
                                            n_chunks=n_chunks, chunk_size=chunk_size,
                                            numpass=num_pass, block_size=block_size)
    if verbose:
        t_ica = time.time() - t_init
        if ica_alg == 'ica':
            print('FastICA completed in: ', t_ica)
        elif ica_alg == 'orica':
            print('ORICA completed in:', t_ica)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    if verbose:
        print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])

    return cleaned_sources_ica, cleaned_A_ica, cleaned_W_ica, source_idx


def clustering(traces, fs, cleaned_sources_ica, num_frames, clustering='mog', spike_thresh=5,
               keep_all_clusters=False, features='amp', verbose=True):
    if verbose:
        print('Clustering Sources with: ', clustering)

    t_start = 0 * pq.s
    t_stop = num_frames / float(fs) * pq.s

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

    return sst, independent_spike_idx


def set_times_labels(sst, fs):
    times = np.array([], dtype=int)
    labels = np.array([])
    for i_s, st in enumerate(sst):
        times = np.concatenate((times, (st.times.magnitude * fs).astype(int)))
        labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

    return NumpySorting.from_times_labels(times.astype(int), labels, fs)
