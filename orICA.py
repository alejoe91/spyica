from __future__ import print_function

import numpy as np
from numpy import linalg as la
import time
import warnings
import matplotlib.pylab as plt
from scipy.linalg import sqrtm
from scipy.linalg import eigh
from scipy.linalg import LinAlgError

# warnings.filterwarnings('error')

# np.seterr(all='warn')

class State():
    def __init__(self, icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign):
        self.icaweights = icaweights
        self.icasphere = icasphere
        self.lambda_k = lambda_k
        self.minNonStatIdx = minNonStatIdx
        self.counter = counter
        self.Rn = Rn
        self.nonStatIdx = nonStatIdx
        self.kurtsign = kurtsign

class ORICA():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', block_white=8, block_ica=8, nsub=0,
                 forgetfac='cooling', localstat=np.inf, ffdecayrate=0.6, evalconverg=True, verbose=False, mu=0, eta=0,
                 adjacency=None):
        '''

        Parameters
        ----------
        data:          np.array - input data (chans-by-samples)
        numpass:       number of passes through the data
        weights:       initial weight matrix     (default -> eye())
        sphering:      ['offline' | 'online'] use online RLS whitening method or pre-whitening
        block_white:   block size for online whitening (in samples)
        block_ica:     block size for ORICA (in samples)
        nsub:          number of subgaussian sources in EEG signal (default -> 0)
        forgetfac:     ['cooling'|'constant'|'adaptive'] forgetting factor profiles
                        'cooling': monotonically decreasing, for relatively stationary data
                        'constant': constant, for online tracking non-stationary data.
                        'adaptive': adaptive based on Nonstatinoarity Index (in dev)
                        See reference [2] for more information.
        localstat:     local stationarity (in number of samples) corresponding to
                       constant forgetting factor at steady state
        ffdecayrate:   [0<f<1] decay rate of (cooling) forgetting factor (default -> 0.6)
        evalconverg:   [0|1] evaluate convergence such as Non-Stationarity Index
        verbose:       bool - give ascii messages  (default -> False)
        mu:            coefficient for spatial smothing
        eta:           coefficient for temporal smoothing (when convolutive)
        adjacency:     adjavency matrix (if mu not 0)

        Reference:
          [1] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Real-time
           adaptive EEG source separation using online recursive independent
           component analysis," IEEE Transactions on Neural Systems and
           Rehabilitation Engineering, 2016.

           [2] S.-H. Hsu, L. Pion-Tanachini, T.-P Jung, and G. Cauwenberghs,
           "Tracking non-stationary EEG sources using adaptive online
           recursive independent component analysis," in IEEE EMBS, 2015.

           [3] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Online
           recursive independent component analysis for real-time source
           separation of high-density EEG," in IEEE EMBS, 2014.

         Author:
               Sheng-Hsiou (Shawn) Hsu, SCCN, UCSD.
               shh078@ucsd.edu
        '''

        nChs, nPts = data.shape

        numPass = numpass
        verbose = verbose

        # Parameters for data whitening
        if sphering == 'online':
            onlineWhitening = True
        else:
            onlineWhitening = False
        blockSizeWhite = block_white
        blockSizeICA = block_ica
        numSubgaussian = nsub

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': 0.995, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        self.adjacency = adjacency
        self.mu = mu
        self.eta = eta

        if weights is None:
            weights = np.eye(nChs)

        ##############################
        # initialize state variables #
        ##############################
        icaweights = weights
        if onlineWhitening:
            icasphere = np.eye(nChs)

        lambda_k = np.zeros((1, blockSizeICA))
        minNonStatIdx =[]
        counter       = 1

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-np.exp(-1 / (self.adaptiveFF['tau_const']))

        if self.evalConvergence['profile']:
            Rn =[]
            nonStatIdx =[]

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        kurtsign = np.ones((nChs, 1))
        if numSubgaussian != 0:
            kurtsign[:numSubgaussian] = 0


        ######################
        # sphere-whiten data #
        ######################
        if not onlineWhitening: # pre - whitening
            if verbose:
                print('Use pre-whitening method.')
                icasphere = 2.0 * la.inv(sqrtm(np.cov(data))) # find the "sphering" matrix = spher()
        else: # Online RLS Whitening
            if verbose:
                print('Use online whitening method.')
        # whiten / sphere the data
        data = np.matmul(icasphere, data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / np.min([blockSizeICA, blockSizeWhite])))

        if verbose:
            printflag = 0
            if self.adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif self.adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif self.adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')
        t_start = time.time()

        for it in range(numPass):
            for bi in range(numBlock - 1):
                dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                                  int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
                if onlineWhitening:
                    self.dynamicWhitening(data[:, dataRange], dataRange)
                    data[:, dataRange] = np.matmul(self.state.icasphere, data[:, dataRange])

                # if np.any(np.isnan(data[:, dataRange])):
                #     raise Exception()
                self.dynamicOrica(data[:, dataRange], dataRange)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        self.unmixing = np.matmul(self.state.icaweights, self.sphere)
        self.mixing = la.pinv(np.matmul(self.unmixing, self.sphere)).T

        # self.y = np.transpose(la.pinv(np.matmul(np.matmul(self.unmixing, self.sphere), data)))
        self.y = np.matmul(np.matmul(self.unmixing, self.sphere), data)


    def dynamicWhitening(self, blockdata, dataRange):
        nPts = blockdata.shape[1]

        if self.adaptiveFF['profile'] == 'cooling':
            lambda_ = self.genCoolingFF(self.state.counter+dataRange, self.adaptiveFF['gamma'], self.adaptiveFF['lambda_0'])
            if lambda_[0] < self.adaptiveFF['lambda_const']:
                lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'constant':
            lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'adaptive':
            lambda_ = np.squeeze(np.tile(self.state.lambda_k[-1], (1, nPts)))


        v = np.matmul(self.state.icasphere, blockdata) # pre - whitened data
        lambda_avg = 1 - lambda_[int(np.ceil(len(lambda_) / 2))] # median lambda
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, np.transpose(v))) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, np.transpose(v)) / nPts
                                            / QWhite * self.state.icasphere)


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        # compute source activation using previous weight matrix
        y = np.matmul(self.state.icaweights, blockdata)

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.state.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.state.kurtsign==1), :])  # Supergaussian
            f[np.where(self.state.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.state.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(y, np.transpose(f))/nPts
            variance = blockdata * blockdata
            if len(self.state.Rn) == 0:
                self.state.Rn = modelFitness
            else:
                self.state.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.state.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.state.nonStatIdx = la.norm(self.state.Rn)


        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = self.genCoolingFF(self.state.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-self.state.lambda_k))
        Q = 1 + self.state.lambda_k * (np.einsum('km,km->m', f, y)-1);
        curr_state = self.state.icaweights

        # Compute smoothing factor
        if self.mu != 0:
            smoothing_matrix = np.zeros(self.state.icaweights.shape)
            for i, comp in enumerate(smoothing_matrix):
                for adj in self.adjacency:
                    smoothing_matrix[i] = 1./len(adj)*np.sum(self.state.icaweights[i, adj])

            self.state.icaweights = lambda_prod * ((self.state.icaweights -
                                                   np.matmul(np.matmul(np.matmul(y,  np.diag(self.state.lambda_k / Q)),
                                                             np.transpose(f)), self.state.icaweights)) +
                                                   self.mu*(self.state.icaweights - smoothing_matrix)) #- eta*())
        else:
            self.state.icaweights = lambda_prod * ((self.state.icaweights -
                                                    np.matmul(np.matmul(np.matmul(y, np.diag(self.state.lambda_k / Q)),
                                                                        np.transpose(f)), self.state.icaweights)))


        # orthogonalize weight matrix
        try:
            D, V = eigh(np.matmul(self.state.icaweights, np.transpose(self.state.icaweights)))
        except LinAlgError:
            raise Exception()

        # curr_state = self.state.icaweights
        self.state.icaweights = np.matmul(np.matmul(
            np.transpose(la.solve(np.transpose(np.diag((np.sqrt(np.abs(D)) * np.sign(D)))), np.transpose(V))),
                                                    np.transpose(V)), self.state.icaweights)

    def genCoolingFF(self, t, gamma, lambda_0):
        lambda_ = lambda_0 / (t ** gamma)
        return lambda_

    def genAdaptiveFF(self, dataRange, lambda_, ratioOfNormRn):
        decayRateAlpha = self.adaptiveFF['decayRateAlpha']
        upperBoundBeta = self.adaptiveFF['upperBoundBeta']
        transBandWidthGamma = self.adaptiveFF['transBandWidthGamma']
        transBandCenter = self.adaptiveFF['transBandCenter']

        gainForErrors = upperBoundBeta * 0.5 * (1 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))
        def f(n):
            return  (1 + gainForErrors) ** n * lambda_[-1] - decayRateAlpha * ((1 + gainForErrors) ** (2 * n - 1) -
                                                                               (1 + gainForErrors) ** (n - 1)) / \
                                                                                gainForErrors * lambda_[-1] **2
        lambda_ = f(range(len(dataRange)))

        return lambda_


def instICA(X, n_comp='all', n_chunks=1, chunk_size=None, adjacency_graph=None, mu=0):
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
                            print('failed ', i)

                idxs.extend(range(c_init, c_init + chunk_size))

            X_reduced = X[:, idxs]
            print(X_reduced.shape)
    else:
        X_reduced = X

    orica = ORICA(X_reduced, sphering='offline', verbose=True, numpass=1,
                  block_white=2000, block_ica=2000, adjacency=adjacency_graph, mu=mu)

    sources = orica.y
    A = orica.mixing
    W = orica.unmixing

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

    orica = ORICA(X_block, sphering='offline', verbose=True, numpass=3,
                  block_white=2000, block_ica=2000)

    sources = orica.y
    A = orica.mixing
    W = orica.unmixing

    return sources, A, W

if __name__ == '__main__':

    recordings = np.fromfile('/home/alessio/Documents/Codes/SpyICA/recordings/convolution/recording_physrot'
                             '_Neuronexus-32-cut-30_10_2.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_all_'
                             '08-02-2018:18:08_2904/kilosort/raw.dat', dtype='int16').reshape(64000, 30).transpose()

    ori = ORICA(recordings, sphering='offline', verbose=True, numpass=1, block_white=1000, block_ica=1000)

    y = ori.y
