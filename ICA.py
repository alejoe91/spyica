'''

'''

import numpy as np
from sklearn.decomposition import FastICA

def instICA(X, n_comp='all', n_chunks=1, chunk_size=None):
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

    n_obs = X.shape[1]

    if n_chunks > 1:
        if chunk_size is None:
            raise AttributeError('Chunk size (n_samples) is required')
        else:
            assert chunk_size*n_chunks < n_obs
            chunk_init = []
            idxs = []
            for c in range(n_chunks):
                proceed = False
                i = 0
                while not proceed and i<1000:
                    c_init = np.random.randint(n_obs-n_chunks-1)
                    proceed = True
                    for prev_c in chunk_init:
                        if c_init > prev_c and c_init < c_init + chunk_size:
                            proceed = False
                            i += 1
                            print 'failed ', i

                idxs.extend(range(c_init, c_init + chunk_size))

            X_reduced = X[:, idxs]
            print X_reduced.shape
    else:
        X_reduced = X

    ica = FastICA(n_components=n_comp) #, algorithm='deflation')
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


def cICAweights(weight, mea_dim=None, cmap='viridis', style='mat', origin='lower'):
    import matplotlib.pyplot as plt
    #  check if number of spike is 1
    if len(weight.shape) == 3:
        raise AttributeError('Plot one weight at a time!')
    else:
        if np.mod(weight.shape[0], (mea_dim[0] * mea_dim[1])) == 0:
            lag = weight.shape[0] / (mea_dim[0] * mea_dim[1])
            fig = plt.figure()
            axes = []
            images = []
            for l in range(lag):
                ax = fig.add_subplot(1, lag, l+1)
                w_l = weight[l::lag]
                mea_values = w_l.reshape((mea_dim[0], mea_dim[1]))
                if style == 'mat':
                    im = ax.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                else:
                    im = ax.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                ax.axis('off')
                axes.append(ax)
                images.append(im)
        else:
            raise Exception('MEA dimnensions are wrong!')
        return axes, images


def gFICA(X, dim, mode='time', n_comp='all'):
    """Performs instantaneous gradient-flow ICA described in:

    Stanacevic, M., Cauwenberghs, G., & Zweig, G. (2002, May).
    Gradient flow adaptive beamforming and signal separation in a miniature microphone array.
    In Acoustics, Speech, and Signal Processing (ICASSP),
    2002 IEEE International Conference on (Vol. 4, pp. IV-4016). IEEE.

    Parameters
    ----------
    X : np.array
        2d array of analog signals (N x T)
    dim :  array dimension to reshape
    mode : 'time' - 'space' - 'spacetime'
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
    X_grad_x_res = np.zeros((dim[0]-1, dim[1], n_samples-1))
    X_grad_y_res = np.zeros((dim[0], dim[1]-1, n_samples-1))

    for i in range(dim[0]-1):
        for j in range(dim[1]):
            X_grad_x_res[i, j] = X_res[i+1, j, 1:]-X_res[i, j, 1:]
    for i in range(dim[0]):
        for j in range(dim[1]-1):
            X_grad_y_res[i, j] = X_res[i, j+1, 1:]-X_res[i, j, 1:]

    X_grad_t = np.diff(X, axis=1)
    X_grad_t = np.pad(X_grad_t, ((0, 0), (0, 1)), 'constant')
    X_grad_x = np.reshape(X_grad_x_res, ((dim[0] - 1) * (dim[1]), n_samples - 1))
    X_grad_y = np.reshape(X_grad_y_res, ((dim[0]) * (dim[1] - 1), n_samples-1))

    if mode == 'time':
        X_gf = np.vstack((X, X_grad_t))
    elif mode == 'space':
        X_gf = np.vstack((X, X_grad_x, X_grad_y))
    elif mode == 'spacetime':
        X_gf = np.vstack((X, X_grad_t, X_grad_x, X_grad_y))
    else:
        raise AttributeError('Gradient flow mode is unknown!')

    if n_comp == 'all':
        n_comp = X_gf.shape[0]

    ica = FastICA(n_components=n_comp)
    sources = np.transpose(ica.fit_transform(np.transpose(X_gf)))
    A = ica.mixing_
    W = ica.components_

    return sources, A, W

def gfICAweights(weight, mea_dim=None, mode='time', cmap='viridis', style='mat', origin='lower'):
    import matplotlib.pyplot as plt
    if len(weight.shape) == 3:
        raise AttributeError('Plot one weight at a time!')
    else:

        if mode == 'time':
            dim = mea_dim[0] * mea_dim[1]

            if 2*dim == weight.shape[0]:
                w = weight[:dim]
                w_t = weight[dim:]
                mea_values = w.reshape((mea_dim[0], mea_dim[1]))
                mea_values_t = w_t.reshape((mea_dim[0], mea_dim[1]))

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)

                if style == 'mat':
                    im1 = ax1.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.matshow(np.transpose(mea_values_t), cmap=cmap, origin=origin)
                else:
                    im1 = ax1.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.imshow(np.transpose(mea_values_t), cmap=cmap, origin=origin)
                ax1.axis('off')
                ax2.axis('off')

                ax1.set_title('time')
                ax2.set_title('grad time')

                axes = [ax1, ax2]
                images = [im1, im2]
            else:
                raise Exception('MEA dimnensions are wrong!')

        elif mode == 'space':
            dim = mea_dim[0] * mea_dim[1]
            dx_dim = (mea_dim[0] - 1) * mea_dim[1]
            dy_dim = mea_dim[0] * (mea_dim[1] - 1)

            if dim + dx_dim + dy_dim == weight.shape[0]:
                w = weight[:dim]
                w_x = weight[dim:dim + dx_dim]
                w_y = weight[dim + dx_dim:dim + dx_dim+dy_dim]

                mea_values = w.reshape((mea_dim[0], mea_dim[1]))
                mea_values_x = w_x.reshape((mea_dim[0] - 1, mea_dim[1]))
                mea_values_y = w_y.reshape((mea_dim[0], mea_dim[1] - 1))

                fig = plt.figure()
                ax1 = fig.add_subplot(1, 3, 1)
                ax2 = fig.add_subplot(1, 3, 2)
                ax3 = fig.add_subplot(1, 3, 3)

                if style == 'mat':
                    im1 = ax1.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.matshow(np.transpose(mea_values_x), cmap=cmap, origin=origin)
                    im3 = ax3.matshow(np.transpose(mea_values_y), cmap=cmap, origin=origin)
                else:
                    im1 = ax1.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.imshow(np.transpose(mea_values_x), cmap=cmap, origin=origin)
                    im3 = ax3.imshow(np.transpose(mea_values_y), cmap=cmap, origin=origin)
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                ax1.set_title('time')
                ax2.set_title('grad x')
                ax3.set_title('grad y')

                axes = [ax1, ax2, ax3]
                images = [im1, im2, im3]
            else:
                raise Exception('MEA dimnensions are wrong!')
        elif mode == 'spacetime':
            dim = mea_dim[0] * mea_dim[1]
            dx_dim = (mea_dim[0] - 1) * mea_dim[1]
            dy_dim = mea_dim[0] * (mea_dim[1] - 1)

            if 2*dim + dx_dim + dy_dim == weight.shape[0]:
                w = weight[:dim]
                w_t = weight[dim:2*dim]
                w_x = weight[2*dim:dim + dx_dim]
                w_y = weight[2*dim + dx_dim:dim + dx_dim + dy_dim]

                mea_values = w.reshape((mea_dim[0], mea_dim[1]))
                mea_values_t = w_t.reshape((mea_dim[0], mea_dim[1]))
                mea_values_x = w_x.reshape((mea_dim[0] - 1, mea_dim[1]))
                mea_values_y = w_y.reshape((mea_dim[0], mea_dim[1] - 1))

                fig = plt.figure()
                ax1 = fig.add_subplot(2, 2, 1)
                ax2 = fig.add_subplot(2, 2, 2)
                ax3 = fig.add_subplot(2, 2, 3)
                ax4 = fig.add_subplot(2, 2, 4)

                if style == 'mat':
                    im1 = ax1.matshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.matshow(np.transpose(mea_values_t), cmap=cmap, origin=origin)
                    im3 = ax3.matshow(np.transpose(mea_values_x), cmap=cmap, origin=origin)
                    im4 = ax4.matshow(np.transpose(mea_values_y), cmap=cmap, origin=origin)
                else:
                    im1 = ax1.imshow(np.transpose(mea_values), cmap=cmap, origin=origin)
                    im2 = ax2.matshow(np.transpose(mea_values_t), cmap=cmap, origin=origin)
                    im3 = ax3.imshow(np.transpose(mea_values_x), cmap=cmap, origin=origin)
                    im4 = ax4.imshow(np.transpose(mea_values_y), cmap=cmap, origin=origin)
                ax1.axis('off')
                ax2.axis('off')
                ax3.axis('off')
                ax4.axis('off')

                axes = [ax1, ax2, ax3, ax4]
                images = [im1, im2, im3, im4]

                ax1.set_title('time')
                ax2.set_title('grad time')
                ax3.set_title('grad x')
                ax4.set_title('grad y')

            else:
                raise Exception('MEA dimnensions are wrong!')
        else:
                raise AttributeError('Gradient flow mode is unknown!')





    return axes, images


