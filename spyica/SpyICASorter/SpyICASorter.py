from __future__ import print_function

import time
import quantities as pq
import numpy as np
import spyica.ica as ica
import spyica.orica as orica

from spikeinterface import NumpySorting
from spikeinterface import sortingcomponents as sc
from spyica.tools import clean_sources, cluster_spike_amplitudes, detect_and_align, \
    reject_duplicate_spiketrains, clean_tests


class SpyICASorter:
    def __init__(self, recording):
        self.recording = recording
        self.fs = recording.get_sampling_frequency()
        self.cut_traces = None
        self.selected_idxs = None
        self.peaks_subsamp = None
        self.s_ica = None
        self.A_ica = None
        self.W_ica = None
        self.source_idx = None
        self.cleaned_sources_ica = None
        self.cleaned_A_ica = None
        self.cleaned_W_ica = None
        self.sst = None
        self.independent_spike_idx = None

    def mask_traces(self, sample_window_ms=2, percent_spikes=None, balance_spikes_on_channel=False,
                    max_num_spikes=None, use_lambda=True):
        """
        Find mask based on spike peaks

        Parameters
        ----------
        recording: si.RecordingExtractor
            The input recording extractor
        fs: float
            Sampling frequency
        sample_window_ms: float, int, list, or None
            If float or int, it's a symmetric window
            If list, it needs to have 2 elements. Asymmetric window
            If None, all traces are used
        percent_spikes: float
            Percentage of spikes selected
            If None, all spikes are used
        max_num_spikes: int
            Maximum number of spikes allowed
            If None, all spikes are used
        balance_spikes_on_channel: bool
            If true, the number of samples taken from each channel depends on the total number of spikes on the channel
            If false, random subsampling
        use_lambda
        Returns
        -------
        cut_traces: numpy array
            Array with traces

        """
        if sample_window_ms is None:
            self.cut_traces = self.recording.get_traces().astype('int16').T
            return

        # set sample window
        if isinstance(sample_window_ms, float) or isinstance(sample_window_ms, int):
            sample_window_ms = [sample_window_ms, sample_window_ms]
        sample_window = [int(sample_window_ms[0] * self.fs / 1000), int(sample_window_ms[1] * self.fs / 1000)]
        num_channels = self.recording.get_num_channels()
        peaks = sc.detect_peaks(self.recording, progress_bar=True)

        t_init = time.time()
        # subsampling
        if percent_spikes is not None:
            if max_num_spikes is not None and percent_spikes * len(peaks['sample_ind']) > max_num_spikes:
                percent_spikes = max_num_spikes / len(peaks['sample_ind'])
            if balance_spikes_on_channel:
                final_idxs = []
                for chan in np.arange(num_channels):
                    occurrences = list(peaks['channel_ind']).count(chan)
                    num_samples = occurrences * percent_spikes
                    idxs = np.where(peaks['channel_ind'] == chan)[0]
                    idxs = np.random.choice(idxs, int(num_samples))
                    final_idxs.extend(list(idxs))
                final_idxs = sorted(final_idxs)
                self.peaks_subsamp = peaks['sample_ind'][final_idxs]
            else:
                num_samples = len(peaks['sample_ind']) * percent_spikes
                self.peaks_subsamp = np.random.choice(peaks['sample_ind'], int(num_samples))
        else:
            self.peaks_subsamp = peaks['sample_ind']
        t_end = time.time() - t_init

        print(f"Number of detected spikes: {len(peaks['sample_ind'])}")
        print(f"Number of sampled spikes: {len(self.peaks_subsamp)}")
        print(f"Elapsed time subsampling: {t_end}")

        # find idxs
        t_init2 = time.time()
        if use_lambda:
            idxs_spike = map(lambda peak: np.arange(peak - sample_window[0], peak + sample_window[1]),
                             self.peaks_subsamp)
            self.selected_idxs = np.unique(list(idxs_spike))
        else:
            self.selected_idxs = np.array([], dtype=int)
            for peak_ind in self.peaks_subsamp:
                self.selected_idxs = np.concatenate(
                    (self.selected_idxs, np.arange(peak_ind - sample_window[0], peak_ind + sample_window[1])),
                    dtype=int)
            self.selected_idxs = np.sort(np.unique(self.selected_idxs))
        t_end2 = time.time() - t_init2
        print(f"Elapsed time idxs selection: {t_end2}")

        self.selected_idxs = np.array(sorted(list(self.selected_idxs)))
        self.selected_idxs = self.selected_idxs[self.selected_idxs > 1]
        self.selected_idxs = self.selected_idxs[self.selected_idxs < self.recording.get_num_samples(0) - 1]

        t_init3 = time.time()
        # if percent_spikes == 1.0:
        #     for res in np.split(self.selected_idxs, np.where(np.diff(self.selected_idxs) != 1)[0] + 1):
        #         traces = self.recording.get_traces(start_frame=res[0], end_frame=res[-1]).astype("int16")
        #         if self.cut_traces is None:
        #             self.cut_traces = traces
        #         else:
        #             self.cut_traces = np.vstack((self.cut_traces, traces))
        #     self.cut_traces = self.cut_traces.T
        # else:
        #     self.cut_traces = self.recording.get_traces().astype('int16').T[:, self.selected_idxs]
        self.cut_traces = self.recording.get_traces().astype('int16').T[:, self.selected_idxs]
        t_end3 = time.time() - t_init3

        print(f"Sample number for ICA: {len(self.selected_idxs)} from {self.recording.get_num_samples(0)}\n"
              f"Elapsed time getting traces: {t_end3}")

        print(f"Shape: {self.cut_traces.shape}")

    def compute_ica(self, n_comp, ica_alg='ica', n_chunks=0,
                    chunk_size=0, num_pass=1, block_size=800, verbose=True, max_iter=200):
        if ica_alg == 'ica' or ica_alg == 'orica':
            if verbose and ica_alg == 'ica':
                print('Applying FastICA algorithm')
            elif verbose and ica_alg == 'orica':
                print('Applying offline ORICA')
        else:
            raise Exception("Only 'ica' and 'orica' are implemented")

        t_init = time.time()

        # TODO use random snippets (e.g. 20% of the data) / or spiky signals for fast ICA
        if ica_alg == 'ica':
            self.s_ica, self.A_ica, self.W_ica = ica.instICA(self.cut_traces, n_comp=n_comp, n_chunks=n_chunks,
                                                             chunk_size=chunk_size, max_iter=max_iter)
        else:
            self.s_ica, self.A_ica, self.W_ica = orica.instICA(self.cut_traces, n_comp=n_comp,
                                                               n_chunks=n_chunks, chunk_size=chunk_size,
                                                               numpass=num_pass, block_size=block_size)

        print(f"Elapsed time: {time.time() - t_init}")

    def clean_sources_ica(self, method='old', kurt_thresh=1, skew_thresh=0.2, verbose=True):
        if method == 'old':
            # clean sources based on skewness and kurtosis
            self.cleaned_sources_ica, self.source_idx = clean_sources(self.s_ica, kurt_thresh=kurt_thresh,
                                                                      skew_thresh=skew_thresh)
        else:
            self.cleaned_sources_ica, self.source_idx = clean_tests(self.A_ica, self.s_ica, self.recording, method)

        cleaned_src_ica, src_idx = clean_sources(self.s_ica, kurt_thresh=kurt_thresh,
                                                 skew_thresh=skew_thresh)
        print(src_idx, src_idx.shape[0])

        self.cleaned_A_ica = self.A_ica[self.source_idx]
        self.cleaned_W_ica = self.W_ica[self.source_idx]
        print(self.source_idx, self.source_idx.shape[0])

        if verbose:
            print('Number of cleaned sources: ', self.cleaned_sources_ica.shape[0])

    def cluster(self, num_frames, clustering='mog', spike_thresh=5,
                keep_all_clusters=False, features='amp', verbose=True):
        if verbose:
            print('Clustering Sources with: ', clustering)

        traces = self.recording.get_traces().astype('int16').T

        t_start = 0 * pq.s
        t_stop = num_frames / float(self.fs) * pq.s

        if clustering == 'kmeans' or clustering == 'mog':
            # detect spikes and align
            detected_spikes = detect_and_align(self.cleaned_sources_ica, self.fs, traces,
                                               t_start=t_start, t_stop=t_stop, n_std=spike_thresh)
            spike_amps = [sp.annotations['ica_amp'] for sp in detected_spikes]
            spike_trains, amps, nclusters, keep, score = \
                cluster_spike_amplitudes(detected_spikes, metric='cal',
                                         alg=clustering, features=features, keep_all=keep_all_clusters)
            if verbose:
                print('Number of spike trains after clustering: ', len(spike_trains))
            self.sst, self.independent_spike_idx, dup = \
                reject_duplicate_spiketrains(spike_trains, sources=self.cleaned_sources_ica)
            if verbose:
                print('Number of spike trains after duplicate rejection: ', len(self.sst))
        else:
            raise Exception("Only 'mog' and 'kmeans' clustering methods are implemented")

        return self.sst, self.independent_spike_idx

    def set_times_labels(self):
        times = np.array([], dtype=int)
        labels = np.array([])
        for i_s, st in enumerate(self.sst):
            times = np.concatenate((times, (st.times.magnitude * self.fs).astype(int)))
            labels = np.concatenate((labels, np.array([i_s + 1] * len(st.times))))

        return NumpySorting.from_times_labels(times.astype(int), labels, self.fs)
