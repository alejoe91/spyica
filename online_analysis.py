from __future__ import print_function

import numpy as np
from numpy import linalg as la
import time
import warnings
import matplotlib.pylab as plt
import quantities as pq
from scipy import stats
from scipy.linalg import sqrtm
from scipy.linalg import eigh
from scipy.linalg import LinAlgError
from sklearn.decomposition import PCA
from spike_sorting import *
import pandas as pd
from tools import *

import orICA as orica


if __name__ == '__main__':
    import sys
    from os.path import join
    from scipy import stats
    import MEAutility as MEA
    from tools import *
    import yaml
    from spike_sorting import plot_mixing

    debug=True

    if '-r' in sys.argv:
        pos = sys.argv.index('-r')
        folder = sys.argv[pos + 1]
    if '-M' in sys.argv:
        pos = sys.argv.index('-M')
        ndim = int(sys.argv[pos + 1])
    else:
        ndim = 'all'

    plot_fig = False
    save_res = False
    save_perf = False
    paper_fig = False
    resfile = 'results.csv'

    # folder = 'recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_physrot_Neuronexus-32-cut-30_10_' \
    #              '10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_10-05-2018:11:37_3002/'
    # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/SqMEA-10-15um/recording_ica_physrot' \
    #          '_SqMEA-10-15um_20_20.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_22-05-2018:14:45_25'
    # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_' \
    #          'physrot_Neuronexus-32-cut-30_15_60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_29-05-2018:16:38_2416'
    # folder = 'recordings/convolution/gtica/SqMEA-10-15um/recording_ica_physrot_Sq' \
    #          'MEA-10-15um_10_20.0s_uncorrelated_5.0_5.0Hz_15.0Hz_modulation_none_13-06-2018:10:22_7593/'

    block = 800
    recordings = np.load(join(folder, 'recordings.npy')) #.astype('int16')
    rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
    with open(join(folder, rec_info), 'r') as f:
        info = yaml.load(f)
    electrode_name = info['General']['electrode name']
    fs = info['General']['fs']
    if isinstance(fs, str):
        fs = pq.Quantity(float(fs.split()[0]), fs.split()[1])
    elif isinstance(fs, pq.Quantity):
        fs = fs
    nsec_window = 10
    nsamples_window = int(fs.rescale('Hz').magnitude * nsec_window)
    mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)
    times = (range(recordings.shape[1]) / fs).rescale('s')

    gtst = np.load(join(folder, 'spiketrains.npy'))
    templates = np.load(join(folder, 'templates.npy'))
    mixing = np.load(join(folder, 'mixing.npy')).T
    sources = np.load(join(folder, 'sources.npy'))
    adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
    n_sources = sources.shape[0]
    # lambda_val = 0.01
    lambda_val = 0.995
    # ff = 'cooling'
    ff = 'adaptive'
    ffdecayrate=0.5

    online = False
    detect = False
    calibPCA = False

    pca_window = 10
    ica_window = 0
    skew_window = 5
    step = 1

    start_time = time.time()
    ori = orica.onlineORICAss(recordings, fs=fs, onlineWhitening=online, calibratePCA=calibPCA, forgetfac=ff,
                              skew_thresh=0.5, ndim=ndim, lambda_0=lambda_val, verbose=True,
                              numpass=1, block=block, step_size=step, ffdecayrate=ffdecayrate,
                              skew_window=skew_window, pca_window=pca_window, ica_window=ica_window,
                              detect_trheshold=10, onlineDetection=False)

    processing_time = time.time() - start_time


    ## Convergence analysis
    a = ori.mixing[-1]

    a_t = np.zeros(a.shape)
    for i, a_ in enumerate(a):
        if np.abs(np.min(a_)) > np.abs(np.max(a_)):
            a_t[i] = -a_ / np.max(np.abs(a_))
        else:
            a_t[i] = a_ / np.max(np.abs(a_))

    last_idxs = find_consistent_sorces(ori.source_idx, thresh=0.5)
    last_idxs = last_idxs[np.argsort(np.abs(stats.skew(ori.y[last_idxs], axis=1)))[::-1]]
    n_id = len(last_idxs)

    a_selected_final = ori.mixing[-1][last_idxs]
    y_selected_final = ori.y[last_idxs]
    y_on_selected_final = ori.y_on[last_idxs]

    correlation, idx_truth, idx_orica, _ = matcorr(mixing.T, a_selected_final)
    sorted_idx = idx_orica[idx_truth.argsort()]
    sorted_corr_idx = idx_truth.argsort()
    sorted_a = np.matmul(np.diag(np.sign(correlation[sorted_corr_idx])), a_selected_final[sorted_idx]).T
    sorted_mixing = mixing[:, np.sort(idx_truth)]
    sorted_y = np.matmul(np.diag(np.sign(correlation[sorted_corr_idx])), y_selected_final[sorted_idx])
    sorted_y_on = np.matmul(np.diag(np.sign(correlation[sorted_corr_idx])), y_on_selected_final[sorted_idx])
    sorted_y_true = sources[np.sort(idx_truth)]

    norm_y_true = sorted_y_true/np.max(np.abs(sorted_y_true), axis=1, keepdims=True)
    norm_y_on = sorted_y_on/np.max(np.abs(sorted_y_on), axis=1, keepdims=True)


    mix_CC_mean_gt, mix_CC_mean_id, sources_CC_mean, \
    corr_cross_mix, corr_cross_sources = evaluate_sum_CC(sorted_a.T, sorted_mixing.T, sorted_y_true, sorted_y,
                                                         n_sources)

    print('Number of GT sources: ', n_sources)
    print('Number of identified sources: ', n_id)
    print('Normalized cumulative correlation GT - mixing: ', mix_CC_mean_gt)
    print('Normalized cumulative correlation ID - mixing: ', mix_CC_mean_id)
    print('Normalized cumulative correlation - sources: ', sources_CC_mean)

    pairs = np.array([np.arange(len(idx_truth)), np.arange(len(idx_truth))]).T
    corr_time = np.zeros((len(ori.mixing), len(correlation)))

    for i, mix in enumerate(ori.mixing):
        a_selected = mix[last_idxs]
        corr, idx_truth, idx_orica, _ = matcorr(mixing.T, a_selected)
        corr_time[i] = np.abs(corr[sorted_corr_idx])

    if plot_fig:
        plt.matshow(corr_time.T)
        plot_mixing(sorted_mixing.T, mea_pos, mea_dim)
        plot_mixing(sorted_a.T, mea_pos, mea_dim)


    ## Non-stationatiry

    # Detection and evaluation
    if detect:
        nsamples = int((2*fs.rescale('Hz')).magnitude)
        ## User defined thresholds and detection
        thresholds = []
        for i, s in enumerate(sorted_y_on):
            fig = plt.figure()
            plt.plot(s[-nsamples:])
            plt.title('IC  ' + str(i+1))
            coord = plt.ginput()
            th = coord[0][1]
            plt.close(fig)
            thresholds.append(th)

        spikes = threshold_spike_sorting(sorted_y_on, thresholds)

        # Convert spikes to neo
        sst = []
        for i, k in enumerate(np.sort(spikes.keys())):
            t = spikes[k]
            tt = t / fs
            st = neo.SpikeTrain(times=tt, t_start=(pca_window+ica_window)*pq.s, t_stop=gtst[0].t_stop)
            st.annotate(ica_source=k)
            sst.append(st)

        if 'ica_wf' not in sst[0].annotations.keys():
            extract_wf(sst, recordings, times, fs, ica=True, sources=sorted_y_on)

        gtst_red = [gt.time_slice(t_start=(pca_window+ica_window)*pq.s, t_stop=gtst[0].t_stop)
                    for gt in np.array(gtst)[np.sort(idx_truth)]]

        # # use pairs from mixing and only detected gtst
        counts, pairs = evaluate_spiketrains(gtst_red, sst, pairs=pairs, t_jitt=3*pq.ms)
        perf = compute_performance(counts)

        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        raster_plots(np.array(gtst_red), color_st=pairs[:, 0], ax=ax1, marker='|')
        raster_plots(np.array(sst), color_st=pairs[:, 1], ax=ax2, marker='|')

        ax1.set_title('GTST', fontsize=15)
        ax2.set_title('SST', fontsize=15)

        simplify_axes([ax1, ax2])

        ax1.set_xlim([100, 102])
        ax2.set_xlim([100, 102])

        fig.tight_layout()

        ## SNR evaluation
        unit_SNR(sst, sorted_y_on, times)
        if plot_fig:
            fig_snr, ax_snr = plt.subplots()
            for (gt, st) in zip(gtst, sst):
                ax_snr.plot(gt.annotations['snr'], st.annotations['ica_snr'], color='black',marker='x')
            ax_snr.plot([0, 30], [0, 30], color='grey', ls='--')



    if ndim == 'all':
        ndim = recordings.shape[0]

    if save_res:
        if not os.path.isfile(join(folder, resfile)):
            df = pd.DataFrame({'block': [block], 'M': ndim, 'ff': [ff], 'lambda': [lambda_val], 'time': [processing_time],
                               'C_gt': [mix_CC_mean_gt], 'C_id': [mix_CC_mean_id], 'C_st': sources_CC_mean,
                               'n_id': [n_id], 'n_sources': [n_sources], 'step_size': step,
                               'pca_window': pca_window, 'ica_window': ica_window, 'skew_window': skew_window})
            print('Saving to ', join(folder, resfile))
            with open(join(folder, resfile), 'w') as f:
                df.to_csv(f)
        else:
            with open(join(folder, resfile), 'r') as f:
                new_index = len(pd.read_csv(f))
            with open(join(folder, resfile), 'a') as f:
                df = pd.DataFrame({'block': [block], 'M': ndim, 'ff': [ff], 'lambda': [lambda_val], 'time': [processing_time],
                                   'C_gt': [mix_CC_mean_gt], 'C_id': [mix_CC_mean_id], 'C_st': sources_CC_mean,
                                   'n_id': [n_id], 'n_sources': [n_sources], 'step_size': step,
                                   'pca_window': pca_window, 'ica_window': ica_window, 'skew_window': skew_window},
                                  index=[new_index])
                print('Appending to ', join(folder, resfile))
                df.to_csv(f, header=False)

    if save_perf:
        # save results
        if not os.path.isdir(join(folder, 'results')):
            os.makedirs(join(folder, 'results'))
        np.save(join(folder, 'results', 'counts'), counts)
        np.save(join(folder, 'results', 'performance'), perf)
        np.save(join(folder, 'results', 'thresholds'), thresholds)
        np.save(join(folder, 'results', 'y_on'), sorted_y_on)

    if paper_fig:
        from matplotlib import gridspec
        gs0 = gridspec.GridSpec(2, 2, wspace=0.2, hspace=0.2, bottom=0.01, top=0.93, left=0.02, right=0.98)

        fig_p = plt.figure(figsize=(15, 15))
        plot_mixing(sorted_mixing.T, mea_pos, mea_dim, gs=gs0[0], fig=fig_p)
        plot_mixing(sorted_a.T, mea_pos, mea_dim, gs=gs0[1], fig=fig_p)
        fig_p.text(0.17, 0.95, 'GT mixing', fontsize=25)
        fig_p.text(0.67, 0.95, 'ORICA mixing', fontsize=25)

        ax_21 = fig_p.add_subplot(gs0[2])
        ax_22 = fig_p.add_subplot(gs0[3])

        raster_plots(np.array(gtst_red), color_st=pairs[:, 0], ax=ax_21, marker='|', markersize=60, mew=2)
        raster_plots(np.array(sst), color_st=pairs[:, 1], ax=ax_22, marker='|', markersize=60, mew=2)

        ax_21.set_title('GT Spike Trains', fontsize=25)
        ax_22.set_title('ORICA Spike Trains', fontsize=25)

        ax_21.axis('off')
        ax_22.axis('off')

        fig_p.text(0.01, 0.95, 'A', fontsize=45, fontweight='demibold')
        fig_p.text(0.51, 0.95, 'B', fontsize=45, fontweight='demibold')
        fig_p.text(0.01, 0.44, 'C', fontsize=45, fontweight='demibold')
        fig_p.text(0.51, 0.44, 'D', fontsize=45, fontweight='demibold')

        ax_21.set_xlim([100, 102])
        ax_22.set_xlim([100, 102])

        fig_p.tight_layout()

