# Helper functions
from __future__ import division

import numpy as np
import quantities as pq
from sklearn.decomposition import PCA


def apply_pca(X, n_comp):
    from sklearn.decomposition import PCA

    # whiten data
    pca = PCA(n_components=n_comp)
    data = pca.fit_transform(np.transpose(X))

    return np.transpose(data), pca.components_


def whiten_data(X, n_comp=None):
    '''

    Parameters
    ----------
    X: nfeatures x nsa
    n_comp: number of components

    Returns
    -------

    '''
    # whiten data
    if n_comp == None:
        n_comp = np.min(X.shape)

    n_feat, n_samples = X.shape

    pca = PCA(n_components=n_comp, whiten=True)
    data = pca.fit_transform(X.T)
    eigvecs = pca.components_
    eigvals = pca.explained_variance_
    sphere = np.matmul(np.diag(1. / np.sqrt(eigvals)), eigvecs)

    return np.transpose(data), eigvecs, eigvals, sphere


############ SPYICA ######################

# TODO remove neo

def detect_and_align(sources, fs, recordings, t_start=None, t_stop=None, n_std=5, ref_period_ms=2, n_pad_ms=2,
                     upsample=8):
    '''

    Parameters
    ----------
    sources
    fs
    recordings
    t_start
    t_stop
    n_std
    ref_period
    upsample

    Returns
    -------

    '''
    import scipy.signal as ss
    import quantities as pq
    import neo

    idx_spikes = []
    idx_sources = []
    spike_trains = []
    times = np.arange(sources.shape[1])
    ref_period = int((ref_period_ms / 1000.0) * fs)
    n_pad = int((n_pad_ms / 1000.0) * fs)

    for s_idx, s in enumerate(sources):
        thresh = -n_std * np.median(np.abs(s) / 0.6745)
        idx_spike = np.where(s < thresh)[0]
        idx_spikes.append(idx_spike)
        intervals = np.diff(idx_spike)

        sp_times = []
        sp_wf = []
        sp_rec_wf = []
        sp_amp = []
        first_spike = True

        if idx_spike.shape[0] > 1:
            for i_t, t in enumerate(intervals):
                idx = idx_spike[i_t]
                if t > 1 or i_t == len(intervals) - 1:
                    if idx - n_pad > 0 and idx + n_pad < len(s):
                        spike = s[idx - n_pad:idx + n_pad]
                        # t_spike = times[idx - n_pad:idx + n_pad]
                        t_spike = np.arange(idx - n_pad, idx + n_pad)
                        spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                    elif idx - n_pad < 0:
                        spike = s[:idx + n_pad]
                        spike = np.pad(spike, (np.abs(idx - n_pad), 0), 'constant')
                        # t_spike = times[:idx + n_pad]
                        t_spike = np.arange(idx + n_pad)
                        t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant')
                        spike_rec = recordings[:, :idx + n_pad]
                        spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                    elif idx + n_pad > len(s):
                        spike = s[idx - n_pad:]
                        spike = np.pad(spike, (0, idx + n_pad - len(s)), 'constant')
                        # t_spike = times[idx - n_pad:]
                        t_spike = np.arange(idx - n_pad, sources.shape[1])
                        t_spike = np.pad(t_spike, (0, idx + n_pad - len(s)), 'constant')
                        spike_rec = recordings[:, idx - n_pad:]
                        spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - len(s))), 'constant')

                    if first_spike:
                        nsamples = len(spike)
                        nsamples_up = nsamples * upsample
                        first_spike = False

                    # upsample and find minimum
                    if upsample > 1:
                        spike_up = ss.resample_poly(spike, upsample, 1)
                        # times_up = ss.resample_poly(t_spike, upsample, 1
                        t_spike_up = np.linspace(t_spike[0], t_spike[-1], num=len(spike_up))
                    else:
                        spike_up = spike
                        t_spike_up = t_spike

                    min_idx_up = np.argmin(spike_up)
                    min_amp_up = np.min(spike_up)
                    min_time_up = t_spike_up[min_idx_up]

                    min_idx = np.argmin(spike)
                    min_amp = np.min(spike)
                    min_time = t_spike[min_idx]

                    # align waveform
                    shift = nsamples_up // 2 - min_idx_up
                    if shift > 0:
                        spike_up = np.pad(spike_up, (np.abs(shift), 0), 'constant')[:nsamples_up]
                    elif shift < 0:
                        spike_up = np.pad(spike_up, (0, np.abs(shift)), 'constant')[-nsamples_up:]

                    if len(sp_times) != 0:
                        # print(min_time_up, sp_times[-1])
                        if min_time_up - sp_times[-1] > ref_period:
                            sp_wf.append(spike_up)
                            sp_rec_wf.append(spike_rec)
                            sp_amp.append(min_amp_up)
                            sp_times.append(min_time_up)
                    else:
                        sp_wf.append(spike_up)
                        sp_rec_wf.append(spike_rec)
                        sp_amp.append(min_amp_up)
                        sp_times.append(min_time_up)

            if t_start and t_stop:
                for i, sp in enumerate(sp_times):
                    if sp < t_start.rescale('s').magnitude * fs:
                        sp_times[i] = t_start.rescale('s').magnitude * fs
                    if sp > t_stop.rescale('s').magnitude * fs:
                        sp_times[i] = t_stop.rescale('s').magnitude * fs
            elif t_stop:
                for i, sp in enumerate(sp_times):
                    if sp > t_stop.rescale('s').magnitude * fs:
                        sp_times[i] = t_stop.rescale('s').magnitude * fs
            else:
                t_start = 0 * pq.s
                t_stop = sp_times[-1] / fs * pq.s

            spiketrain = neo.SpikeTrain(np.array(sp_times) / fs * pq.s, t_start=t_start, t_stop=t_stop,
                                        waveforms=np.array(sp_rec_wf))
            spiketrain.annotate(ica_amp=np.array(sp_amp))
            spiketrain.annotate(ica_wf=np.array(sp_wf))
            spike_trains.append(spiketrain)
            idx_sources.append(s_idx)

    return spike_trains


# todo use spikeinterfce NumpyExtractor and sorting extractor
def extract_wf(sst, recordings, times, fs, upsample=8, ica=False, sources=[]):
    '''

    Parameters
    ----------
    sst
    sources
    recordings
    times
    fs
    upsample

    Returns
    -------

    '''
    import scipy.signal as ss
    import quantities as pq

    n_pad = int(2 * pq.ms * fs.rescale('kHz'))
    unit = times[0].rescale('ms').units

    nChs, nPts = recordings.shape

    if ica:
        if len(sources) == 0:
            raise Exception('Provide IC sources for IC waveforms')
        for (st, s) in zip(sst, sources):
            sp_wf = []
            sp_rec_wf = []
            sp_amp = []
            first_spike = True

            for t in st:
                idx = np.where(times > t)[0][0]
                # find single waveforms crossing thresholds
                if idx - n_pad > 0 and idx + n_pad < nPts:
                    spike = s[idx - n_pad:idx + n_pad]
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    spike = s[:idx + n_pad]
                    spike = np.pad(spike, (np.abs(idx - n_pad), 0), 'constant')
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > nPts:
                    spike = s[idx - n_pad:]
                    spike = np.pad(spike, (0, idx + n_pad - nPts), 'constant')
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - nPts), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - nPts)), 'constant')

                if first_spike:
                    nsamples = len(spike)
                    nsamples_up = nsamples * upsample
                    first_spike = False

                min_ic_amp = np.min(spike)
                sp_wf.append(spike)
                sp_rec_wf.append(spike_rec)
                sp_amp.append(min_ic_amp)

            st.waveforms = np.array(sp_rec_wf)
            st.annotate(ica_amp=np.array(sp_amp))
            st.annotate(ica_wf=np.array(sp_wf))
    else:
        for st in sst:
            sp_rec_wf = []
            sp_amp = []
            first_spike = True

            for t in st:
                idx = np.where(times > t)[0][0]
                # find single waveforms crossing thresholds
                if idx - n_pad > 0 and idx + n_pad < nPts:
                    t_spike = times[idx - n_pad:idx + n_pad]
                    spike_rec = recordings[:, idx - n_pad:idx + n_pad]
                elif idx - n_pad < 0:
                    t_spike = times[:idx + n_pad]
                    t_spike = np.pad(t_spike, (np.abs(idx - n_pad), 0), 'constant') * unit
                    spike_rec = recordings[:, :idx + n_pad]
                    spike_rec = np.pad(spike_rec, ((0, 0), (np.abs(idx - n_pad), 0)), 'constant')
                elif idx + n_pad > nPts:
                    t_spike = times[idx - n_pad:]
                    t_spike = np.pad(t_spike, (0, idx + n_pad - nPts), 'constant') * unit
                    spike_rec = recordings[:, idx - n_pad:]
                    spike_rec = np.pad(spike_rec, ((0, 0), (0, idx + n_pad - nPts)), 'constant')
                if first_spike:
                    nsamples = len(spike_rec)
                    nsamples_up = nsamples * upsample
                    first_spike = False

                min_amp = np.min(spike_rec)
                sp_rec_wf.append(spike_rec)
                sp_amp.append(min_amp)

            st.waveforms = np.array(sp_rec_wf)


def reject_duplicate_spiketrains(sst, percent_threshold=0.5, min_spikes=3, sources=None, parallel=False,
                                 nprocesses=None):
    '''

    Parameters
    ----------
    sst
    percent_threshold
    min_spikes

    Returns
    -------

    '''
    import neo
    import multiprocessing
    import time

    if nprocesses is None:
        nprocesses = len(sst)

    spike_trains = []
    idx_sources = []
    duplicates = []

    if parallel:
        # t_start = time.time()
        pool = multiprocessing.Pool(nprocesses)
        results = [pool.apply_async(find_duplicates, (i, sp_times, sst,))
                   for i, sp_times in enumerate(sst)]
        duplicates = []
        for result in results:
            duplicates.extend(result.get())

    else:
        # t_start = time.time()
        for i, sp_times in enumerate(sst):
            # check if overlapping with another source
            t_jitt = 1 * pq.ms
            counts = []
            for j, sp in enumerate(sst):
                count = 0
                if i != j:
                    for t_i in sp_times:
                        id_over = np.where((sp > t_i - t_jitt) & (sp < t_i + t_jitt))[0]
                        if len(id_over) != 0:
                            count += 1
                    if count >= percent_threshold * len(sp_times):
                        if [i, j] not in duplicates and [j, i] not in duplicates:
                            print('Found duplicate spike trains: ', i, j, count)
                            duplicates.append([i, j])
                    counts.append(count)

    duplicates = np.array(duplicates)
    discard = []
    if len(duplicates) > 0:
        for i, sp_times in enumerate(sst):
            if i not in duplicates:
                # rej ect spiketrains with less than 3 spikes...
                if len(sp_times) >= min_spikes:
                    spike_trains.append(sp_times)
                    idx_sources.append(i)
            else:
                # Keep spike train with largest number of spikes among duplicates
                idxs = np.argwhere(duplicates == i)
                max_len = []
                c_max = 0
                st_idx = []
                for idx in idxs:
                    row = idx[0]
                    st_idx.append(duplicates[row][0])
                    st_idx.append(duplicates[row][1])
                    c_ = np.max([len(sst[duplicates[row][0]]), len(sst[duplicates[row][1]])])
                    i_max = np.argmax([len(sst[duplicates[row][0]]), len(sst[duplicates[row][1]])])
                    if len(max_len) == 0 or c_ > c_max:
                        max_len = [row, i_max]
                        c_max = c_
                index = duplicates[max_len[0], max_len[1]]
                if index not in idx_sources and index not in discard:
                    spike_trains.append(sst[index])
                    idx_sources.append(index)
                    [discard.append(d) for d in st_idx if d != index and d not in discard]


    else:
        spike_trains = sst
        idx_sources = range(len(sst))

    return spike_trains, idx_sources, duplicates


def find_duplicates(i, sp_times, sst, percent_threshold=0.5, t_jitt=1 * pq.ms):
    counts = []
    duplicates = []
    for j, sp in enumerate(sst):
        count = 0
        if i != j:
            for t_i in sp_times:
                id_over = np.where((sp > t_i - t_jitt) & (sp < t_i + t_jitt))[0]
                if len(id_over) != 0:
                    count += 1
            if count >= percent_threshold * len(sp_times):
                if [i, j] not in duplicates and [j, i] not in duplicates:
                    print('Found duplicate spike trains: ', i, j, count)
                    duplicates.append([i, j])
            counts.append(count)

    return duplicates


def clean_sources(sources, kurt_thresh=0.7, skew_thresh=0.5, remove_correlated=True):
    '''

    Parameters
    ----------
    sources
    kurt_thresh
    skew_thresh

    Returns
    -------

    '''
    import scipy.stats as stat
    import scipy.signal as ss

    sk = stat.skew(sources, axis=1)
    ku = stat.kurtosis(sources, axis=1)

    high_sk = np.where(np.abs(sk) >= skew_thresh)[0]
    low_sk = np.where(np.abs(sk) < skew_thresh)[0]
    high_ku = np.where(ku >= kurt_thresh)[0]
    low_ku = np.where(ku < kurt_thresh)[0]

    idxs = np.unique(np.concatenate((high_sk, high_ku)))

    # sources_sp = sources[high_sk]
    # sources_disc = sources[low_sk]
    # # compute correlation matrix
    # corr = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # mi = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # max_lag = np.zeros((sources_sp.shape[0], sources_sp.shape[0]))
    # for i in range(sources_sp.shape[0]):
    #     s_i = sources_sp[i]
    #     for j in range(i + 1, sources_sp.shape[0]):
    #         s_j = sources_sp[j]
    #         cmat = crosscorrelation(s_i, s_j, maxlag=50)
    #         # cmat = ss.correlate(s_i, s_j)
    #         corr[i, j] = np.max(np.abs(cmat))
    #         max_lag[i, j] = np.argmax(np.abs(cmat))
    #         mi[i, j] = calc_MI(s_i, s_j, bins=100)

    # sources_keep = sources[idxs]
    # corr_idx = np.argwhere(corr > corr_thresh)
    # sk_keep = stat.skew(sources_keep, axis=1)

    # # remove smaller skewnesses
    # remove_ic = []
    # for idxs in corr_idx:
    #     sk_pair = sk_keep[idxs]
    #     remove_ic.append(idxs[np.argmin(np.abs(sk_pair))])
    # remove_ic = np.array(remove_ic)
    #
    # if len(remove_ic) != 0 and remove_correlated:
    #     mask = np.array([True] * len(sources_keep))
    #     mask[remove_ic] = False
    #
    #     spike_sources = sources_keep[mask]
    #     source_idx = high_sk[mask]
    # else:
    # source_idx = high_sk

    spike_sources = sources[idxs]
    sk_sp = stat.skew(spike_sources, axis=1)

    # invert sources with positive skewness
    spike_sources[sk_sp > 0] = -spike_sources[sk_sp > 0]

    return spike_sources, idxs  # , corr_idx, corr, mi


# TODO use isosplit and cluster waveforms from all ICs to spot duplicates
# TODO return IC_templates
def cluster_spike_amplitudes(sst, metric='cal', min_sihlo=0.8, min_cal=100, max_clusters=4,
                             alg='kmeans', features='amp', ncomp=3, keep_all=False):
    '''

    Parameters
    ----------
    spike_amps
    sst
    metric
    min_sihlo
    min_cal
    max_clusters
    alg
    keep_all

    Returns
    -------

    '''
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from copy import copy

    spike_wf = np.array([sp.annotations['ica_wf'] for sp in sst], dtype=object)
    spike_amps = [sp.annotations['ica_amp'] for sp in sst]
    nclusters = np.zeros(len(spike_amps))
    silhos = np.zeros(len(spike_amps))
    cal_hars = np.zeros(len(spike_amps))

    reduced_amps = []
    reduced_sst = []
    keep_id = []

    if features == 'amp':
        for i, amps in enumerate(spike_amps):
            silho = 0
            cal_har = 0
            keep_going = True

            if len(amps) > 2:
                for k in range(2, max_clusters):
                    if alg == 'kmeans':
                        kmeans_new = KMeans(n_clusters=k, random_state=0)
                        kmeans_new.fit(amps.reshape(-1, 1))
                        labels_new = kmeans_new.predict(amps.reshape(-1, 1))
                    elif alg == 'mog':
                        gmm_new = GaussianMixture(n_components=k, covariance_type='full')
                        gmm_new.fit(amps.reshape(-1, 1))
                        labels_new = gmm_new.predict(amps.reshape(-1, 1))

                    if len(np.unique(labels_new)) > 1:
                        silho_new = silhouette_score(amps.reshape(-1, 1), labels_new)
                        cal_har_new = calinski_harabasz_score(amps.reshape(-1, 1), labels_new)
                        if metric == 'silho':
                            if silho_new > silho:
                                silho = silho_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                        elif metric == 'cal':
                            if cal_har_new > cal_har:
                                cal_har = cal_har_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                    else:
                        keep_going = False
                        nclusters[i] = 1

                    if not keep_going:
                        break

                if nclusters[i] != 1:
                    if metric == 'silho':
                        if silho < min_sihlo:
                            nclusters[i] = 1
                            reduced_sst.append(sst[i])
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    reduced_sst.append(sst[i][idxs])
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                highest_clust = np.argmin(kmeans.cluster_centers_)
                                highest_idx = np.where(labels == highest_clust)[0]
                                reduced_sst.append(sst[i][highest_idx])
                                reduced_amps.append(amps[highest_idx])
                                keep_id.append(highest_idx)
                    elif metric == 'cal':
                        if cal_har < min_cal:
                            nclusters[i] = 1
                            sst[i].annotate(ica_source=i)
                            reduced_sst.append(sst[i])
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_sst.append(red_spikes)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                if alg == 'kmeans':
                                    highest_clust = np.argmin(kmeans.cluster_centers_)
                                elif alg == 'mog':
                                    highest_clust = np.argmin(gmm.means_)
                                idxs = np.where(labels == highest_clust)[0]
                                if len(idxs) > 0:
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i][idxs].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_sst.append(red_spikes)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                    silhos[i] = silho
                    cal_hars[i] = cal_har
                else:
                    red_spikes = copy(sst[i])
                    red_spikes.annotations = copy(sst[i].annotations)
                    red_spikes.annotate(ica_source=i)
                    reduced_sst.append(red_spikes)
                    reduced_amps.append(amps)
                    keep_id.append(range(len(sst[i])))
            else:
                red_spikes = copy(sst[i])
                red_spikes.annotations = copy(sst[i].annotations)
                red_spikes.annotate(ica_source=i)
                reduced_sst.append(red_spikes)
                reduced_amps.append(amps)
                keep_id.append(range(len(sst[i])))
    # TODO keep cluster with largest amplitude (compute amplitudes)
    elif features == 'pca':
        for i, wf in enumerate(spike_wf):
            # apply pca on ica_wf
            wf_pca, comp = apply_pca(wf.T, n_comp=ncomp)
            wf_pca = wf_pca.T
            amps = spike_amps[i]

            silho = 0
            cal_har = 0
            keep_going = True

            if len(wf_pca) > 2:
                for k in range(2, max_clusters):
                    if alg == 'kmeans':
                        kmeans_new = KMeans(n_clusters=k, random_state=0)
                        kmeans_new.fit(wf_pca)
                        labels_new = kmeans_new.predict(wf_pca)
                    elif alg == 'mog':
                        gmm_new = GaussianMixture(n_components=k, covariance_type='full')
                        gmm_new.fit(wf_pca)
                        labels_new = gmm_new.predict(wf_pca)

                    if len(np.unique(labels_new)) > 1:
                        silho_new = silhouette_score(wf_pca, labels_new)
                        cal_har_new = calinski_harabasz_score(wf_pca, labels_new)
                        if metric == 'silho':
                            if silho_new > silho:
                                silho = silho_new
                                nclusters[i] = k
                                if alg == 'kmeans':
                                    kmeans = kmeans_new
                                elif alg == 'mog':
                                    gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                        elif metric == 'cal':
                            if cal_har_new > cal_har:
                                cal_har = cal_har_new
                                if metric == 'cal':
                                    nclusters[i] = k
                                    if alg == 'kmeans':
                                        kmeans = kmeans_new
                                    elif alg == 'mog':
                                        gmm = gmm_new
                                labels = labels_new
                            else:
                                keep_going = False
                    else:
                        keep_going = False
                        nclusters[i] = 1

                    if not keep_going:
                        break

                if nclusters[i] != 1:
                    if metric == 'silho':
                        if silho < min_sihlo:
                            nclusters[i] = 1
                            red_spikes = sst[i]
                            red_spikes.annotations = copy(sst[i].annotations)
                            red_spikes.annotate(ica_source=i)
                            reduced_sst.append(red_spikes)
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                # TODO keep cluster with largest amplitude (compute amplitudes)
                                highest_clust = np.argmin(kmeans.cluster_centers_)
                                idxs = np.where(labels == highest_clust)[0]
                                red_spikes.annotations = copy(sst[i].annotations)
                                if 'ica_amp' in red_spikes.annotations:
                                    red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                if 'ica_wf' in red_spikes.annotations:
                                    red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                red_spikes.annotate(ica_source=i)
                                reduced_amps.append(amps[highest_clust])
                                keep_id.append(highest_clust)
                    elif metric == 'cal':
                        if cal_har < min_cal:
                            nclusters[i] = 1
                            red_spikes = copy(sst[i])
                            red_spikes.annotations = copy(sst[i].annotations)
                            red_spikes.annotate(ica_source=i)
                            reduced_sst.append(red_spikes)
                            reduced_amps.append(amps)
                            keep_id.append(range(len(sst[i])))
                        else:
                            if keep_all:
                                for clust in np.unique(labels):
                                    idxs = np.where(labels == clust)[0]
                                    red_spikes = sst[i][idxs]
                                    red_spikes.annotations = copy(sst[i].annotations)
                                    if 'ica_amp' in red_spikes.annotations:
                                        red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                    if 'ica_wf' in red_spikes.annotations:
                                        red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                    red_spikes.annotate(ica_source=i)
                                    reduced_sst.append(red_spikes)
                                    reduced_amps.append(amps[idxs])
                                    keep_id.append(idxs)
                            else:
                                # for PCA the sign might be inverted
                                if alg == 'kmeans':
                                    highest_clust = np.argmax(np.abs(kmeans.cluster_centers_))
                                elif alg == 'mog':
                                    highest_clust = np.argmax(np.abs(gmm.means_))
                                idxs = np.where(labels == highest_clust)[0]
                                red_spikes = sst[i][idxs]
                                red_spikes.annotations = copy(sst[i].annotations)
                                if 'ica_amp' in red_spikes.annotations:
                                    red_spikes.annotate(ica_amp=red_spikes.annotations['ica_amp'][idxs])
                                if 'ica_wf' in red_spikes.annotations:
                                    red_spikes.annotate(ica_wf=red_spikes.annotations['ica_wf'][idxs])
                                red_spikes.annotate(ica_source=i)
                                reduced_sst.append(red_spikes)
                                reduced_amps.append(amps[idxs])
                                keep_id.append(idxs)
                    silhos[i] = silho
                    cal_hars[i] = cal_har
                else:
                    red_spikes = copy(sst[i])
                    red_spikes.annotations = copy(sst[i].annotations)
                    red_spikes.annotate(ica_source=i)
                    reduced_sst.append(red_spikes)
                    reduced_amps.append(amps)
                    keep_id.append(range(len(sst[i])))
            else:
                red_spikes = copy(sst[i])
                red_spikes.annotations = copy(sst[i].annotations)
                red_spikes.annotate(ica_source=i)
                reduced_sst.append(red_spikes)
                reduced_amps.append(amps)
                keep_id.append(range(len(sst[i])))

    if metric == 'silho':
        score = silhos
    elif metric == 'cal':
        score = cal_hars

    return reduced_sst, reduced_amps, nclusters, keep_id, score


def template_matching(sources, ic_templates):
    pass


def matcorr(x, y, rmmean=False, weighting=None):
    from scipy.optimize import linear_sum_assignment

    m, n = x.shape
    p, q = y.shape
    m = np.min([m, p])

    if m != n or p != q:
        # print 'matcorr(): Matrices are not square: using max abs corr method (2).'
        method = 2

    if n != q:
        raise Exception('Rows in the two input matrices must be the same length.')

    if rmmean:
        x = x - np.mean(x, axis=1)  # optionally remove means
        y = y - np.mean(y, axis=1)

    dx = np.sum(x ** 2, axis=1)
    dy = np.sum(y ** 2, axis=1)
    dx[np.where(dx == 0)] = 1
    dy[np.where(dy == 0)] = 1
    # raise Exception()
    corrs = np.matmul(x, y.T) / np.sqrt(dx[:, np.newaxis] * dy[np.newaxis, :])

    if weighting != None:
        if any(corrs.shape != weighting.shape):
            print('matcorr(): weighting matrix size must match that of corrs')
        else:
            corrs = corrs * weighting

    cc = np.abs(corrs)

    # Performs Hungarian algorithm matching
    col_ind, row_ind = linear_sum_assignment(-cc.T)

    idx = np.argsort(-cc[row_ind, col_ind])
    corr = corrs[row_ind, col_ind][idx]
    indy = np.arange(m)[idx]
    indx = row_ind[idx]

    return corr, indx, indy, corrs


def evaluate_PI(ic_unmix, gt_mix):
    '''

    Parameters
    ----------
    gt_mix
    ic_mix

    Returns
    -------

    '''
    H = np.matmul(ic_unmix, gt_mix)
    C = H ** 2
    N = np.min([gt_mix.shape[0], ic_unmix.shape[0]])

    PI = (N - 0.5 * (np.sum(np.max(C, axis=0) / np.sum(C, axis=0)) + np.sum(np.max(C, axis=1) / np.sum(C, axis=1)))) / (
            N - 1)

    return PI, C


def evaluate_sum_CC(ic_mix, gt_mix, ic_sources, gt_sources, n_sources):  # ):
    '''

    Parameters
    ----------
    ic_unmix
    gt_mix
    ic_source
    gt_source

    Returns
    -------

    '''
    correlation, idx_truth, idx_, corr_m = matcorr(gt_mix, ic_mix)
    correlation, idx_truth, idx_, corr_s = matcorr(gt_sources, ic_sources)

    # corr_mix = np.corrcoef(ic_mix, gt_mix)
    # corr_sources = np.corrcoef(ic_sources, gt_sources)
    #
    # id_sources = ic_mix.shape[0]
    #
    # corr_cross_mix = corr_mix[id_sources:, :id_sources] ** 2
    # corr_cross_sources = corr_sources[id_sources:, :id_sources] ** 2
    corr_cross_mix = corr_m ** 2
    corr_cross_sources = corr_s ** 2

    mix_CC_mean_gt = np.trace(corr_cross_mix) / n_sources
    mix_CC_mean_id = np.trace(corr_cross_mix) / len(ic_sources)
    sources_CC_mean = np.trace(corr_cross_sources) / n_sources

    return mix_CC_mean_gt, mix_CC_mean_id, sources_CC_mean, corr_cross_mix, corr_cross_sources


def find_consistent_sorces(source_idx, thresh=0.5):
    '''
    Returns sources that appear at least thresh % of the times
    Parameters
    ----------
    source_idx
    thresh

    Returns
    -------

    '''
    len_no_empty = len([s for s in source_idx if len(s) > 0])

    s_dict = {}
    for s in source_idx:
        for id in s:
            if id not in s_dict.keys():
                s_dict.update({id: 1})
            else:
                s_dict[id] += 1

    consistent_sources = []
    for id in s_dict.keys():
        if s_dict[id] >= thresh * len_no_empty:
            consistent_sources.append(id)

    return np.sort(consistent_sources)


def threshold_spike_sorting(recordings, threshold):
    '''

    Parameters
    ----------
    recordings
    threshold

    Returns
    -------

    '''
    spikes = {}
    for i_rec, rec in enumerate(recordings):
        sp_times = []
        if isinstance(threshold, list):
            idx_spikes = np.where(rec < threshold[i_rec])
        else:
            idx_spikes = np.where(rec < threshold)
        if len(idx_spikes[0]) != 0:
            idx_spikes = idx_spikes[0]
            for t, idx in enumerate(idx_spikes):
                # find single waveforms crossing thresholds
                if t == 0:
                    sp_times.append(idx)
                elif idx - idx_spikes[t - 1] > 1:  # or t == len(idx_spike) - 2:  # single spike
                    sp_times.append(idx)

            spikes.update({i_rec: sp_times})

    return spikes


def clean_tests(A_ica, s_ica, recording, method):
    import scipy.stats as ss
    # find closest channels
    max_ids = np.argmax(A_ica, axis=1)
    chan_loc = recording.get_channel_locations()
    closest = []
    for idx in max_ids:
        pos = chan_loc[idx]
        chans = []
        for c in range(32):
            dist = np.sqrt(np.square(pos[0] - chan_loc[c, 0]) + np.square(pos[1] - chan_loc[c, 1]))
            if dist <= 60:
                chans.append(c)
        closest.append(chans)

    if method == 'average':
        av_val = []
        for i, ind in enumerate(max_ids):
            max_val = A_ica[i, ind]
            close_val = A_ica[i, closest[i]]
            av = max_val - (np.sum(close_val) - max_val) / (len(close_val) - 1)
            av_val.append(av)
        av_max = np.average(np.asarray(av_val))
        source_idx = np.where(av_val > av_max*0.66)[0]
        print(av_max)

    if method == 'std':
        tmp_val = []
        for i, ind in enumerate(max_ids):
            std = np.std(A_ica[i, closest[i]])
            tmp_val.append(std)
        av_std = np.average(np.asarray(tmp_val))
        source_idx = np.where(tmp_val > av_std)[0]
        print(av_std)

    cleaned_sources_ica = s_ica[source_idx]
    sk_sp = ss.skew(cleaned_sources_ica, axis=1)
    # invert sources with positive skewness
    cleaned_sources_ica[sk_sp > 0] = -cleaned_sources_ica[sk_sp > 0]

    return cleaned_sources_ica, np.array(source_idx)


from spyica.SpyICASorter.SpyICASorter import SpyICASorter


def localize_sources(sorter, axis=1):
    if not isinstance(sorter, SpyICASorter):
        raise Exception("Import a SpyICASorter object")

    recording = sorter.recording
    idx = sorter.source_idx
    chan_positions = recording.get_channel_locations()
    if axis == 0:
        matrix = np.abs(sorter.A_ica[:, idx])
        s = np.sum(matrix, axis=axis)
        coms = (matrix.T @ chan_positions) / s[:, np.newaxis]
    else:
        matrix = np.abs(sorter.A_ica[idx])
        s = np.sum(matrix, axis=axis)
        coms = (matrix @ chan_positions)/s[:, np.newaxis]
    return coms
