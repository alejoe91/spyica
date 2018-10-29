from .tools import clean_sources, cluster_spike_amplitudes, detect_and_align, reject_duplicate_spiketrains
from .ICA import instICA
import quantities as pq
import spikeinterface as si
import numpy as np

def ica_spike_sorting(recording, clustering='mog', n_comp='all',
                      features='amp', skew_thresh=0.2, kurt_thresh=1,
                      n_chunks=0, chunk_size=0, spike_thresh=5,
                      keep_all_clusters=False):

    if not isinstance(recording, si.RecordingExtractor):
        raise Exception("Input a RecordingExtractor object!")

    if n_comp == 'all':
        n_comp = recording.getNumChannels()
    fs = recording.getSamplingFrequency()
    s_ica, A_ica, W_ica = instICA(recording.getTraces(), n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])
    print('Clustering Sources with: ', clustering)

    t_start = 0*pq.s
    t_stop = recording.getNumFrames() / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, recording.getTraces(),
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)

        spike_trains_rej, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        sst = spike_trains_rej

        sorting = si.NumpySortingExtractor()
        times = np.array([])
        labels = np.array([])
        for i_s, st in enumerate(sst):
            times = np.concatenate((times, (st.times.magnitude*fs).astype(int)))
            labels = np.concatenate((labels, np.array([i_s+1]*len(st.times))))

        sorting.setTimesLabels(times, labels)
        #TODO add spike properties and features
    else:
        raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

    return sorting, cleaned_sources_ica


def orica_spike_sorting(recording, clustering='mog', n_comp='all',
                      features='amp', skew_thresh=0.2, kurt_thresh=1,
                      n_chunks=0, chunk_size=0, spike_thresh=5,
                      keep_all_clusters=False):

    if n_comp == 'all':
        n_comp = recording.getNumChannels()
    fs = recording.getSamplingFrequency()
    s_ica, A_ica, W_ica = instICA(recording.getTraces(), n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])
    print('Clustering Sources with: ', clustering)

    t_start = 0*pq.s
    t_stop = recording.getNumFrames() / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, recording.getTraces(),
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)

        spike_trains_rej, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        sst = spike_trains_rej

        sorting = si.NumpySortingExtractor()
        times = np.array([])
        labels = np.array([])
        for i_s, st in enumerate(sst):
            times = times.append((times, (st.times.magnitude*fs).astype(int)))
            labels = labels.append((labels, np.array([i_s+1]*len(st.times))))

        sorting.setTimesLabels(times, labels)

        return sorting


def online_orica_spike_sorting(recording, clustering='mog', n_comp='all',
                      features='amp', skew_thresh=0.2, kurt_thresh=1,
                      n_chunks=0, chunk_size=0, spike_thresh=5,
                      keep_all_clusters=False):

    if n_comp == 'all':
        n_comp = recording.getNumChannels()
    fs = recording.getSamplingFrequency()
    s_ica, A_ica, W_ica = instICA(recording.getTraces(), n_comp=n_comp, n_chunks=n_chunks, chunk_size=chunk_size)

    # clean sources based on skewness and correlation
    cleaned_sources_ica, source_idx = clean_sources(s_ica, kurt_thresh=kurt_thresh, skew_thresh=skew_thresh)
    cleaned_A_ica = A_ica[source_idx]
    cleaned_W_ica = W_ica[source_idx]

    print('Number of cleaned sources: ', cleaned_sources_ica.shape[0])
    print('Clustering Sources with: ', clustering)

    t_start = 0*pq.s
    t_stop = recording.getNumFrames() / float(fs) * pq.s

    if clustering == 'kmeans' or clustering == 'mog':
        # detect spikes and align
        detected_spikes = detect_and_align(cleaned_sources_ica, fs, recording.getTraces(),
                                           t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
        spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
        spike_trains, amps, nclusters, keep, score = \
            cluster_spike_amplitudes(detected_spikes, metric='cal',
                                     alg=clustering, features=features, keep_all=keep_all_clusters)

        spike_trains_rej, independent_spike_idx, dup = \
            reject_duplicate_spiketrains(spike_trains, sources=cleaned_sources_ica)
        sst = spike_trains_rej

        sorting = si.NumpySortingExtractor()
        times = np.array([])
        labels = np.array([])
        for i_s, st in enumerate(sst):
            times = times.append((times, (st.times.magnitude*fs).astype(int)))
            labels = labels.append((labels, np.array([i_s+1]*len(st.times))))

        sorting.setTimesLabels(times, labels)

        return sorting



