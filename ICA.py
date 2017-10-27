'''

'''

import numpy as np
from sklearn.decomposition import FastICA

def instICA(X, n_comp='all'):
    """Performs instantaneous ICA.

    Parameters
    ----------
    X : np.array
        2d array of analog signals (N x T)
    n_comp : int or 'all'
             number of ICA components

    Returns
    -------
    sources : sources
    A : mixing matrix
    W : unmixing matrix

    """
    if n_comp == 'all':
        n_comp = X.shape[0]

    ica = FastICA(n_components=n_comp)
    sources = np.transpose(ica.fit_transform(np.transpose(X)))
    A = ica.mixing_
    W = ica.components_

    return sources, A, W


def cICAemb(X, n_comp='all', L=3):
    """Performs convolutive embedded ICA described in:

    Leibig, C., Wachtler, T., & Zeck, G. (2016).
    Unsupervised neural spike sorting for high-density microelectrode arrays with convolutive independent component
    analysis.
    Journal of neuroscience methods, 271, 1-13.


    Parameters
    ----------
    X : np.array
        2d array of analog signals (N x T)
    n_comp : int or 'all'
             number of ICA components
    L : int
        length of filter (number of mixing matrices)

    Returns
    -------
    sources : sources
    A : mixing matrix
    W : unmixing matrix

    """
    n_chan = X.shape[0]
    X_block = np.zeros((L*n_chan, X.shape[1]-L))
    for ch in range(n_chan):
        for lag in range(L):
            if lag == 0:
                X_block[L * ch + lag, :] = X[ch, L-lag:]
            else:
                X_block[L * ch + lag, :] = X[ch, L - lag:-lag]

    if n_comp == 'all':
        n_comp = X_block.shape[0]

    ica = FastICA(n_components=n_comp)
    sources = np.transpose(ica.fit_transform(np.transpose(X_block)))
    A = ica.mixing_
    W = ica.components_

    return sources, A, W



def gFICA(X, dim, n_comp='all'):
    """Performs instantaneous gradient-flow ICA described in:

    Stanacevic, M., Cauwenberghs, G., & Zweig, G. (2002, May).
    Gradient flow adaptive beamforming and signal separation in a miniature microphone array.
    In Acoustics, Speech, and Signal Processing (ICASSP),
    2002 IEEE International Conference on (Vol. 4, pp. IV-4016). IEEE.

    Parameters
    ----------
    X : np.array
        2d array of analog signals (N x T)
    n_comp : int or 'all'
             number of ICA components

    Returns
    -------
    sources : sources
    A : mixing matrix
    W : unmixing matrix

    """
    n_elec = X.shape[0]
    n_samples = X.shape[1]
    if dim[0]*dim[1] != n_elec:
        raise AttributeError('Reshape dimensions are wrong!')
    X_res = np.reshape(X, (dim[0], dim[1], n_samples))
    # X_grad_t = np.zeros((n_elec, n_samples-1))
    X_grad_i = np.zeros((dim[0]-1, dim[1], n_samples-1))
    X_grad_j = np.zeros((dim[0], dim[1]-1, n_samples-1))

    for i in range(dim[0]-1):
        for j in range(dim[1]):
            X_grad_i[i, j] = X_res[i+1, j, 1:]-X_res[i, j, 1:]
    for i in range(dim[0]):
        for j in range(dim[1]-1):
            X_grad_j[i, j] = X_res[i, j+1, 1:]-X_res[i, j, 1:]
    X_grad_t = np.diff(X, axis=1)

    X_gf = np.vstack((X_grad_t,
                      np.reshape(X_grad_i, ((dim[0]-1)*(dim[1]), n_samples-1)),
                      np.reshape(X_grad_j, ((dim[0])*(dim[1]-1), n_samples-1))))

    if n_comp == 'all':
        n_comp = X_gf.shape[0]

    ica = FastICA(n_components=n_comp)
    sources = np.transpose(ica.fit_transform(np.transpose(X_gf)))
    A = ica.mixing_
    W = ica.components_

    return sources, A, W


