from __future__ import print_function

import numpy as np
from sklearn.decomposition import FastICA


def instICA(X, n_comp='all', n_chunks=1, chunk_size=None, max_iter=200):
    """Performs instantaneous ICA.

    Parameters
    ----------
    X : np.array
        2d array of analog signals (N x T)
    n_comp : int or 'all'
             number of ICA components
    max_iter :  int
                max number of ICA iterations

    Returns
    -------
    sources : sources
    A : mixing matrix
    W : unmixing matrix

    """
    if n_comp == 'all' or n_comp is None:
        n_comp = X.shape[0]
    else:
        n_comp = int(n_comp)

    n_obs = X.shape[1]

    if n_chunks > 1:
        if chunk_size is None:
            raise AttributeError('Chunk size (n_samples) is required')
        else:
            assert chunk_size * n_chunks < n_obs
            chunk_init = []
            idxs = []
            for c in range(n_chunks):
                proceed = False
                i = 0
                while not proceed and i < 1000:
                    c_init = np.random.randint(n_obs - n_chunks - 1)
                    proceed = True
                    for prev_c in chunk_init:
                        if c_init > prev_c and c_init < c_init + chunk_size:
                            proceed = False
                            i += 1
                            print('failed ', i)
                idxs.extend(range(c_init, c_init + chunk_size))
            X_reduced = X[:, idxs]
    else:
        X_reduced = X

    ica = FastICA(n_components=n_comp, max_iter=max_iter)  # , algorithm='deflation')
    ica.fit(np.transpose(X_reduced))
    sources = np.transpose(ica.transform(np.transpose(X)))
    A = np.transpose(ica.mixing_)
    W = ica.components_

    return sources, A, W


def iICAweights(weight, mea_dim=None, axis=None, cmap='viridis', style='mat', origin='lower'):
    import matplotlib.pyplot as plt
    if len(weight.shape) == 3:
        raise AttributeError('Plot one weight at a time!')
    else:
        if axis:
            ax = axis
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if (mea_dim[0] * mea_dim[1]) == weight.shape[0]:
            mea_values = weight.reshape((mea_dim[0], mea_dim[1]))
            if style == 'mat':
                im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
            else:
                im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
        else:
            raise Exception('MEA dimnensions are wrong!')
        ax.axis('off')

        return ax, im


if __name__ == '__main__':
    import time
    import sys, os
    from os.path import join
    from scipy import stats
    import MEAutility as MEA
    from spike_sorting.spyica.spyica.tools import *
    import yaml
    from spike_sorting import plot_mixing

    if len(sys.argv) == 1:
        folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/recording_physrot_Neuronexus-32-cut-30_' \
                 '10_10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulated_24-01-2018_22_00'
        filename = join(folder, 'kilosort/raw.dat')
        recordings = np.fromfile(filename, dtype='int16') \
            .reshape(-1, 30).transpose()

        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)

        electrode_name = info['General']['electrode name']
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        templates = np.load(join(folder, 'templates.npy'))
    else:
        folder = sys.argv[1]
        recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)

        electrode_name = info['General']['electrode name']
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        templates = np.load(join(folder, 'templates.npy'))

    adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)

    t_start = time.time()
    y, a, w = instICA(recordings)
    print('Elapsed time: ', time.time() - t_start)

    skew_thresh = 0.1
    sk = stats.skew(y, axis=1)
    high_sk = np.where(np.abs(sk) >= skew_thresh)

    ku_thresh = 1.5
    ku = stats.kurtosis(y, axis=1)
    high_ku = np.where(np.abs(ku) >= ku_thresh)

    # Smoothness
    a /= np.max(a)
    smooth = np.zeros(a.shape[0])
    for i in range(len(smooth)):
        smooth[i] = (np.mean([1. / len(adj) * np.sum(a[i, j] - a[i, adj]) ** 2
                              for j, adj in enumerate(adj_graph)]))

    print('High skewness: ', np.abs(sk[high_sk]))
    print('Average high skewness: ', np.mean(np.abs(sk[high_sk])))
    print('Number high skewness: ', len(sk[high_sk]))

    print('High kurtosis: ', ku[high_ku])
    print('Average high kurtosis: ', np.mean(ku[high_ku]))
    print('Number high kurtosis: ', len(ku[high_ku]))

    print('Smoothing: ', smooth[high_sk])
    print('Average smoothing: ', np.mean(smooth[high_sk]))

    f = plot_mixing(a[high_sk], mea_pos, mea_dim)
    f.suptitle('fastICA')
    plt.figure();
    plt.plot(y[high_sk].T)
    plt.title('fastICA')
