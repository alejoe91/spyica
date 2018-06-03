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


def whiten_data(X, n_comp=None):
    '''

    Parameters
    ----------
    X: nsa x nfeatures
    n_comp: number of components

    Returns
    -------

    '''
    # whiten data
    if n_comp==None:
        n_comp = np.min(X.shape)
    pca = PCA(n_components=n_comp, whiten=True)
    data = pca.fit_transform(X.T)

    return np.transpose(data), pca.components_, pca.explained_variance_ratio_

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
    def __init__(self, data, numpass=1, weights=None, sphering='offline', ndim='all', lambda_0=0.995, block_white=8, block_ica=8, nsub=0,
                 forgetfac='cooling', localstat=np.inf, ffdecayrate=0.6, evalconverg=True, verbose=False, mu=0, eta=0,
                 adjacency=None, whiten=True, ortho=True):
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
        self.count = 0
        self.whiten = whiten
        self.ortho = ortho
        self.ndim = ndim

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

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        self.adjacency = adjacency
        self.mu = mu
        self.eta = eta

        if weights is None:
            if self.ndim == 'all':
                icaweights = np.eye(nChs)
            else:
                icaweights = np.eye(self.ndim, self.ndim)

        ##############################
        # initialize state variables #
        ##############################
        if onlineWhitening:
            if self.ndim == 'all':
                icasphere = np.eye(nChs)
            else:
                icasphere = np.eye(self.ndim, nChs)

        lambda_k = np.zeros((1, blockSizeICA))
        minNonStatIdx = []
        counter       = 1

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-np.exp(-1 / (self.adaptiveFF['tau_const']))

        if self.evalConvergence['profile']:
            Rn =[]
            nonStatIdx =[]

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        if self.ndim == 'all':
            kurtsign = np.ones((nChs, 1))
        else:
            kurtsign = np.ones((self.ndim, 1))
        if numSubgaussian != 0:
            kurtsign[:numSubgaussian] = 0


        ######################
        # sphere-whiten data #
        ######################
        if not onlineWhitening: # pre - whitening
            if verbose:
                print('Use pre-whitening method.')
            # data_w, comp, exp = whiten_data(data, self.ndim)
            # icasphere = comp

            icasphere = 2.0 * la.inv(sqrtm(np.cov(data))) # find the "sphering" matrix = sphere()
        else: # Online RLS Whitening
            if verbose:
                print('Use online whitening method.')

        # whiten / sphere the data
        if self.whiten:
            if self.ndim == 'all':
                data_w = np.matmul(icasphere, data)
            else:
                # icasphere = icasphere[:self.ndim]
                # data_w = np.matmul(icasphere, data)
                data_w, comp, exp = whiten_data(data, self.ndim)
                # data_w1_all, comp_all, exp_al = whiten_data(data)
                icasphere = comp
                # raise Exception()
        else:
            print('Initializing weights to sphering matrix')
            data_w = data
            icaweights = icasphere

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icasphere_1 = la.pinv(self.state.icasphere)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / np.min([blockSizeICA, blockSizeWhite])))

        self.NSI = []

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
            for bi in range(numBlock):
                dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                                  int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
                if onlineWhitening:
                    self.dynamicWhitening(data_w[:, dataRange], dataRange)
                    data_w[:, dataRange] = np.matmul(self.state.icasphere, data_w[:, dataRange])
                self.dynamicOrica(data_w[:, dataRange], dataRange)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        if self.whiten:
            self.unmixing = np.matmul(self.state.icaweights, self.sphere)
        else:
            self.unmixing = self.state.icaweights
        self.mixing = la.pinv(self.unmixing).T
        self.y = np.matmul(self.unmixing, data)


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
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
                                            / QWhite * self.state.icasphere)


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        # compute source activation using previous weight matrix
        y = np.matmul(self.state.icaweights, blockdata)

        # raise Exception()

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.state.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.state.kurtsign==1), :])  # Supergaussian
            f[np.where(self.state.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.state.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(y, f.T)/nPts
            variance = blockdata * blockdata
            if len(self.state.Rn) == 0:
                self.state.Rn = modelFitness
            else:
                self.state.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.state.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.state.nonStatIdx = la.norm(self.state.Rn)
            self.NSI.append(self.state.nonStatIdx)


        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = self.genCoolingFF(self.state.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-self.state.lambda_k))
        # Q = 1 + self.state.lambda_k * (np.einsum('km,km->m', f, y)-1);
        Q = 1 + self.state.lambda_k * (np.sum(f * y, axis=0) - 1);
        curr_state = self.state.icaweights

        self.count += 1

        # Compute smoothing factor
        if self.mu != 0:
            smoothing_matrix = np.zeros(self.state.icaweights.shape)
            for i, comp in enumerate(smoothing_matrix):
                for adj in self.adjacency:
                    smoothing_matrix[i] = 1./len(adj)*np.sum(self.state.icaweights[i, adj])

            self.state.icaweights = lambda_prod * ((self.state.icaweights -
                                                   np.matmul(np.matmul(np.matmul(y,  np.diag(self.state.lambda_k / Q)),
                                                             f.T), self.state.icaweights)) -
                                                   self.mu*(self.state.icaweights - smoothing_matrix)) #- eta*())
        else:
            self.state.icaweights = lambda_prod * (self.state.icaweights -
                                                   np.matmul(np.matmul(np.matmul(y, np.diag(self.state.lambda_k / Q)),
                                                                    f.T), self.state.icaweights))


        # orthogonalize weight matrix
        if self.ortho:
            try:
                D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
            except LinAlgError:
                raise Exception()

            # curr_state = self.state.icaweights
            self.state.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                        V.T), self.state.icaweights)

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


class onlineORICAss():
    def __init__(self, data, fs, ndim='all', forgetfac='cooling', skew_thresh=0.5, lambda_0=0.995, block=8, nsub=0,
                 ffdecayrate=0.6, verbose=False, weights=None, step_size=1, window=20, initial_window=10,
                 detect_trheshold=8, onlineDetection=True, evalconverg=True):
        '''

        Parameters
        ----------
        data
        fs
        forgetfac
        skew_thresh
        lambda_0
        block
        nsub
        ffdecayrate
        verbose
        weights
        steps
        window
        initial_window
        '''

        nChs, nPts = data.shape
        verbose = verbose

        if isinstance(fs, pq.Quantity):
            fs = int(fs.rescale('Hz').magnitude)
        else:
            fs = int(fs)

        print(fs)

        self.n_window = int(fs*window)
        self.n_init_window = int(fs*initial_window)
        self.n_step_size = int(fs*step_size)

        self.detect_thresh = detect_trheshold
        self.block = block

        # Parameters for data whitening
        onlineWhitening = True
        numSubgaussian = nsub

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}


        if weights is None:
            weights = np.eye(nChs)

        ##############################
        # initialize state variables #
        ##############################
        icaweights = weights
        icasphere = np.eye(nChs)

        lambda_k = np.zeros((1, block))
        minNonStatIdx = []
        nonStatIdx = []
        Rn = []
        counter       = 1
        self.window = window

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-np.exp(-1 / (self.adaptiveFF['tau_const']))

        if self.evalConvergence['profile']:
            Rn =[]
            nonStatIdx =[]

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        kurtsign = np.ones((nChs, 1))
        if numSubgaussian != 0:
            kurtsign[:numSubgaussian] = 0

        # online estimation
        self.N = 0
        self.mus = []
        self.vars = []
        self.sigmas = []
        self.skews = []

        ######################
        # sphere-whiten data #
        ######################
        # Online RLS Whitening
        if verbose:
            print('Use online whitening method.')

        data_w = np.zeros_like(data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.skew_thresh = skew_thresh

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / block))

        self.w = []
        self.m = []
        self.idx_sources = []

        if verbose:
            printflag = 0
            if self.adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif self.adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')

        t_start = time.time()
        self.init = False
        self.spikes = {}
        self.all_sources = np.array([])
        self.nskews = []
        self.NSI = []
        for bi in range(numBlock):
            dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                              int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
            if onlineWhitening:
                self.dynamicWhitening(data[:, dataRange], dataRange)
                data_w[:, dataRange] = np.matmul(self.state.icasphere, data[:, dataRange])

            self.N += len(dataRange)
            self.dynamicOrica(data_w[:, dataRange], dataRange)
            # self.updateSourceStats(data_w[:, dataRange])

            # select sources
            if not np.mod(self.N, self.n_step_size):
                if self.N > self.n_init_window:
                    self.computeStats(data_w)
                    idx_sources = np.where(np.abs(self.skew) > self.skew_thresh)
                    self.w.append(self.state.icaweights)
                    self.m.append(self.state.icasphere)
                    if len(idx_sources) != 0:
                        self.idx_sources.append(idx_sources[0])
                        self.nskews.append(len(idx_sources[0]))

                        self.all_sources = np.concatenate((self.all_sources, idx_sources[0]))
                        self.all_sources = np.sort(np.unique(self.all_sources)).astype('int')
                        if onlineDetection:
                            self.detectSpikes(idx_sources)
                    else:
                        self.idx_sources.append([])
                else:
                    idx_sources = []
                    self.w.append(self.state.icaweights)
                    self.m.append(self.state.icasphere)
                    self.idx_sources.append([])

            if verbose:
                if printflag < np.floor(10 * bi / numBlock):
                    printflag = printflag + 1
                    print(10 * printflag, '%')
                    if self.N > self.n_init_window:
                        print('Sources: ', idx_sources)


        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.m
        self.unmixing = []
        self.mixing = []
        self.source_idx = self.idx_sources

        if not onlineDetection:
            self.spikes = []

        # print(i)
        for (w, m) in zip(self.w, self.m):
            unmixing = np.matmul(w, m)
            mixing = la.pinv(unmixing).T
            self.unmixing.append(unmixing)
            self.mixing.append(mixing)

        self.unmixing = np.array(self.unmixing)
        self.mixing = np.array(self.mixing)
        self.sphere = np.array(self.sphere)
        self.y = np.matmul(self.unmixing[-1], data)
        print('Done')


    def computeStats(self, data):

        if self.N < self.n_window:
            self.y = np.matmul(self.state.icaweights, data[:, :self.N])
            # skewness for source selection
            self.skew = stats.skew(self.y, axis=1)
            # sigma for spike detection
            self.sigma = np.std(self.y, axis=1)
        else:
            self.y = np.matmul(self.state.icaweights, data[:, self.N-self.n_window:self.N])
            # skewness for source selection
            self.skew = stats.skew(self.y, axis=1)
            # sigma for spike detection
            self.sigma = np.std(self.y, axis=1)

        self.sigmas.append(self.sigma)
        self.skews.append(self.skew)


    def detectSpikes(self, idx_sources):

        self.y[self.skew > 0] = -self.y[self.skew > 0]
        if self.init:
            # first time detect spikes on all past signals
            sources = self.y[self.idx_sources[-1]]
        else:
            # then append spikes from single blocks
            sources = self.y[self.idx_sources[-1], -self.n_step_size:]

        for i, (s, s_idx) in enumerate(zip(sources, self.idx_sources[-1])):
            if s_idx in self.spikes.keys():
                times = self.spikes[s_idx]
            else:
                times = np.array([])

            sp_times = []
            # thresh = -self.detect_thresh * np.median(np.abs(s) / 0.6745)
            thresh = -self.detect_thresh * np.std(s)
            # print s_idx, thresh
            idx_spikes = np.where(s < thresh)
            if len(idx_spikes[0]) != 0:
                idx_spikes = idx_spikes[0]
                for t, idx in enumerate(idx_spikes):
                    # find single waveforms crossing thresholds
                    if t == 0:
                        if self.init:
                            if self.N < self.n_window:
                                sp_times.append(idx)
                            else:
                                sp_times.append(idx)
                        else:
                            sp_times.append(self.N - self.n_step_size + idx)
                    elif idx - idx_spikes[t - 1] > 1: # or t == len(idx_spike) - 2:  # single spike
                        # append crossing time
                        if self.init:
                            if self.N < self.n_window:
                                sp_times.append(idx)
                            else:
                                sp_times.append(idx)
                        else:
                            sp_times.append(self.N - self.n_step_size + idx)
                self.spikes.update({s_idx: np.concatenate((times, np.array(sp_times)))})

        if self.init:
            self.init = False


        # if self.N < self.n_window:
        #     idx_spikes.append(idx_spike[0][0])
        # if idx - n_pad > 0 and idx + n_pad < len(s):
        #     spike = s[idx - n_pad:idx + n_pad]
        #     t_spike = times[idx - n_pad:idx + n_pad]
        #     spike_rec = recordings[:, idx - n_pad:idx + n_pad]





    # def updateSourceStats(self, blockdata):
    #
    #     y = np.matmul(self.state.icaweights, blockdata).T
    #
    #     if self.N == 0:
    #         self.N = blockdata.shape[1]
    #
    #         self.mu = 1. / self.N * np.sum(y, axis=0)
    #         self.var = 1. / (self.N - 1) * np.sum((y - self.mu) ** 2, axis=0)
    #         self.sigma = np.sqrt(self.var)
    #         self.skew = 1. / (self.N - 2) * np.sum(((y - self.mu)/self.sigma)**3, axis=0)
    #
    #         self.sumx = np.sum(y, axis=0)
    #         self.sumx2 = np.sum(y ** 2, axis=0)
    #         self.sumx3 = np.sum(y ** 3, axis=0)
    #     else:
    #         self.N += blockdata.shape[1]
    #
    #         self.mu = 1. / self.N * (self.sumx + np.sum(y, axis=0))
    #         self.var = 1. / (self.N - 1) * (self.sumx2 + self.mu ** 2 - 2 * self.sumx * self.mu +
    #                                         np.sum((y - self.mu) ** 2, axis=0))
    #         self.sigma = np.sqrt(self.var)
    #         self.skew = 1. / ((self.N - 2) * self.sigma ** 3) * \
    #                     (self.sumx3 - 3 * self.sumx2 * self.mu + 3 * self.sumx * self.mu ** 2 +
    #                      np.sum((y - self.mu) ** 3, axis=0))
    #
    #         self.sumx += np.sum(y, axis=0)
    #         self.sumx2 += np.sum(y**2, axis=0)
    #         self.sumx3 += np.sum(y**3, axis=0)
    #
    #     self.mus.append(self.mu)
    #     self.vars.append(self.var)
    #     self.sigmas.append(self.sigma)
    #     self.skews.append(self.skew)



    def dynamicWhitening(self, blockdata, dataRange):
        nPts = blockdata.shape[1]

        if self.adaptiveFF['profile'] == 'cooling':
            lambda_ = self.genCoolingFF(self.state.counter+dataRange, self.adaptiveFF['gamma'], self.adaptiveFF['lambda_0'])
            if lambda_[0] < self.adaptiveFF['lambda_const']:
                lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'constant':
            lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))


        v = np.matmul(self.state.icasphere, blockdata) # pre - whitened data
        lambda_avg = 1 - lambda_[int(np.ceil(len(lambda_) / 2))] # median lambda
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
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
            modelFitness = np.eye(nChs) + np.matmul(y, f.T) / nPts
            variance = blockdata * blockdata
            if len(self.state.Rn) == 0:
                self.state.Rn = modelFitness
            else:
                self.state.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.state.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                # !!! this does not account for block update!
            self.state.nonStatIdx = la.norm(self.state.Rn)
            self.NSI.append(self.state.nonStatIdx)

        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = self.genCoolingFF(self.state.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-self.state.lambda_k))
        Q = 1 + self.state.lambda_k * (np.sum(f * y, axis=0) - 1);
        curr_state = self.state.icaweights

        self.state.icaweights = lambda_prod * (self.state.icaweights -
                                               np.matmul(np.matmul(np.matmul(y, np.diag(self.state.lambda_k / Q)),
                                                                f.T), self.state.icaweights))

        # orthogonalize weight matrix
        try:
            D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
        except LinAlgError:
            raise Exception()

        # curr_state = self.state.icaweights
        self.state.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                    V.T), self.state.icaweights)

    def genCoolingFF(self, t, gamma, lambda_0):
        lambda_ = lambda_0 / (t ** gamma)
        return lambda_



class ORICA_W():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', lambda_0=0.995, block_white=8, block_ica=8, nsub=0,
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

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
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
        data_w = np.matmul(icasphere, data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icasphere_1 = la.inv(self.state.icasphere)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts))

        if verbose:
            printflag = 0
            if self.adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif self.adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif self.adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')
        t_start = time.time()

        dataRange = np.arange(nPts)
        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = self.genCoolingFF(self.state.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
                print('Whatever')
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        # icaweights is A. Assumption: W is A.T
        for it in range(numPass):
            for bi in range(nPts):
                A = self.state.icaweights.T
                W = self.state.icaweights
                v = np.expand_dims(data_w[:, bi], axis=1)
                y = np.expand_dims(np.matmul(W, data_w[:, bi]), axis=1)
                f = np.zeros((nChs, 1))

                # choose nonlinear functions for super- vs. sub-gaussian
                f[np.where(self.state.kurtsign == 1)] = -2 * np.tanh(
                    y[np.where(self.state.kurtsign == 1)])  # Supergaussian
                f[np.where(self.state.kurtsign == 0)] = 2 * np.tanh(
                    y[np.where(self.state.kurtsign == 0)])  # Subgaussian


                # update unmixing matrix using online recursive ICA block update rule

                # # Compute smoothing factor
                if self.mu != 0:
                    dS = self.computeSmoothingFactor().T
                    u = np.matmul(f, v.T) + self.mu * dS
                else:
                    u = np.matmul(f, v.T)

                self.state.icaweights = (1 - self.state.lambda_k[bi]) * W + \
                                        self.state.lambda_k[bi] * u

                # orthogonalize weight matrix
                if not np.mod(bi, block_ica):
                    try:
                        D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
                        # diagonalize matrix (maybe every n steps)
                        self.state.icaweights = np.matmul(np.matmul(
                            la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                      V.T), self.state.icaweights)
                    except LinAlgError:
                        raise Exception()

                    if verbose:
                        if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                            printflag = printflag + 1
                            print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        self.unmixing = np.matmul(self.state.icaweights, self.sphere)
        self.mixing = la.pinv(np.matmul(self.unmixing, self.sphere)).T
        self.y = np.matmul(self.unmixing, data)


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
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
                                            / QWhite * self.state.icasphere)

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

class ORICA_A():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', lambda_0=0.995, block_white=8, block_ica=8, nsub=0,
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

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
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
        data_w = np.matmul(icasphere, data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icasphere_1 = la.inv(self.state.icasphere)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts))

        if verbose:
            printflag = 0
            if self.adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif self.adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif self.adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')
        t_start = time.time()

        dataRange = np.arange(nPts)
        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = self.genCoolingFF(self.state.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.state.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
                print('Whatever')
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        # icaweights is A. Assumption: W is A.T
        for it in range(numPass):
            for bi in range(nPts):
                A = self.state.icaweights
                W = self.state.icaweights.T
                v = np.expand_dims(data_w[:, bi], axis=1)
                y = np.expand_dims(np.matmul(W, data_w[:, bi]), axis=1)
                f = np.zeros((nChs, 1))

                # choose nonlinear functions for super- vs. sub-gaussian
                f[np.where(self.state.kurtsign == 1)] = -2 * np.tanh(
                    y[np.where(self.state.kurtsign == 1)])  # Supergaussian
                f[np.where(self.state.kurtsign == 0)] = 2 * np.tanh(
                    y[np.where(self.state.kurtsign == 0)])  # Subgaussian


                # update weight matrix using online recursive ICA block update rule

                # # Compute smoothing factor
                if self.mu != 0:
                    dS = computeRegularizationFactor(A, self.icasphere_1, self.neighbor_mask)
                    u = np.matmul(v, f.T) + self.mu * dS
                else:
                    u = np.matmul(v, f.T)

                self.state.icaweights = (1 - self.state.lambda_k[bi]) * A + \
                                        self.state.lambda_k[bi] * u

                # orthogonalize weight matrix
                if not np.mod(bi, block_ica):
                    try:
                        D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
                    except LinAlgError:
                        raise Exception()

                    # diagonalize matrix (maybe every n steps)
                    self.state.icaweights = np.matmul(
                        np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                  V.T), self.state.icaweights)
                    if verbose:
                        if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                            printflag = printflag + 1
                            print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        # mixing from data to y
        self.unmixing = np.matmul(la.pinv(self.state.icaweights), self.sphere)
        self.mixing = la.pinv(self.unmixing).T
        self.y = np.matmul(self.unmixing, data)


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
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
                                            / QWhite * self.state.icasphere)


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


class ORICA_W_block():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', lambda_0=0.995, block_white=8, block_ica=8, nsub=0,
                 forgetfac='cooling', localstat=np.inf, ffdecayrate=0.6, evalconverg=True, verbose=False, mu=0, eta=0,
                 adjacency=None, regmode='L1'):
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

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        self.adjacency = adjacency
        self.mu = mu
        self.eta = eta
        self.smooth_count = 0
        self.reg = []
        self.regmode = regmode
        self.lambdas = np.array([])


        if weights is None:
            weights = np.eye(nChs)

        if self.mu != 0:
            self.neighbor_masks = []
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    mask = []
                    for k in range(weights.shape[0]):
                        for l in range(weights.shape[1]):
                            for n in adjacency[l]:
                                if l == i or n == i:
                                    mask.append([i, j, k, l, n])
                    self.neighbor_masks.append(mask)

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
        data_w = np.matmul(icasphere, data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icasphere_1 = la.inv(self.state.icasphere)

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

        self.glob_count = 0

        for it in range(numPass):
            for bi in range(numBlock):
                dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                                  int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
                if onlineWhitening:
                    self.dynamicWhitening(data_w[:, dataRange], dataRange)
                    data_w[:, dataRange] = np.matmul(self.state.icasphere, data_w[:, dataRange])

                # if np.any(np.isnan(data[:, dataRange])):
                #     raise Exception()
                self.dynamicOrica(data_w[:, dataRange], dataRange)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        self.unmixing = np.matmul(self.state.icaweights, self.sphere)
        self.mixing = la.pinv(np.matmul(self.unmixing, self.sphere)).T
        self.y = np.matmul(self.unmixing, data)
        self.reg = np.array(self.reg)


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
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
                                            / QWhite * self.state.icasphere)


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        A = self.state.icaweights.T
        W = self.state.icaweights
        # compute source activation using previous weight matrix
        y = np.matmul(self.state.icaweights, blockdata)
        v = blockdata

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.state.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.state.kurtsign==1), :])  # Supergaussian
            f[np.where(self.state.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.state.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(y, f.T)/nPts
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
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        self.lambdas = np.concatenate((self.lambdas, self.state.lambda_k))


        coeff = np.append(1, self.state.lambda_k) * np.append(1,np.cumprod(1 - self.state.lambda_k[::-1]))[::-1]

        # Compute smoothing factor
        if self.mu != 0:
            self.smooth_count += 1

            dS, S = computeRegularizationFactor(W, M_1 = self.icasphere_1, mode=self.regmode, adj_graph=self.adjacency)
            self.smooth_count = 0
            self.glob_count += 1
            self.reg.append(S)
            self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(f, np.diag(coeff[1:])), v.T)\
                                    + self.mu * np.sum(coeff[1:]) * dS #np.sum(np.einsum('i,jk->ijk', coeff, dS), axis=0)
        else:
            self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(f, np.diag(coeff[1:])), v.T)

        # orthogonalize weight matrix
        try:
            D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
        except LinAlgError:
            raise Exception()

        # curr_state = self.state.icaweights
        self.state.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                    V.T), self.state.icaweights)

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


class ORICA_A_block():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', lambda_0=0.995, block_white=8, block_ica=8, nsub=0,
                 forgetfac='cooling', localstat=np.inf, ffdecayrate=0.6, evalconverg=True, verbose=False, mu=0, eta=0,
                 adjacency=None, regmode='L1'):
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

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        self.adjacency = adjacency
        self.mu = mu
        self.eta = eta
        self.smooth_count = 0
        self.reg = []
        self.regmode = regmode
        self.lambdas = np.array([])

        if weights is None:
            weights = np.eye(nChs)

        if self.mu != 0:
            self.neighbor_masks = []
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    mask = []
                    for k in range(weights.shape[0]):
                        for l in range(weights.shape[1]):
                            for n in adjacency[l]:
                                if l == i or n == i:
                                    mask.append([i, j, k, l, n])
                    self.neighbor_masks.append(mask)

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
        data_w = np.matmul(icasphere, data)

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icasphere_1 = la.inv(self.state.icasphere)

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
            for bi in range(numBlock):
                dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                                  int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
                if onlineWhitening:
                    self.dynamicWhitening(data_w[:, dataRange], dataRange)
                    data_w[:, dataRange] = np.matmul(self.state.icasphere, data_w[:, dataRange])

                self.dynamicOrica(data_w[:, dataRange], dataRange)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')


        # output weights and sphere matrices
        self.sphere = self.state.icasphere
        # mixing from data to y
        self.unmixing = np.matmul(la.pinv(self.state.icaweights), self.sphere)
        self.mixing = la.pinv(self.unmixing).T
        self.y = np.matmul(self.unmixing, data)


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
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(v, v.T) / nPts
                                            / QWhite * self.state.icasphere)

    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        A = self.state.icaweights
        W = self.state.icaweights.T
        y = np.matmul(W, blockdata)
        v = blockdata

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.state.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.state.kurtsign==1), :])  # Supergaussian
            f[np.where(self.state.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.state.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(y, f.T)/nPts
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
            self.state.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = self.genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)


        coeff = np.append(1, self.state.lambda_k) * np.append(1, np.cumprod(1 - self.state.lambda_k[::-1]))[::-1]

        if self.mu != 0:
            self.smooth_count += 1

            dS, S = computeRegularizationFactor(W, M_1 = self.icasphere_1, mode=self.regmode, adj_graph=self.adjacency)
            self.smooth_count = 0
            # self.glob_count += 1
            self.reg.append(S)
            self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(v, np.diag(coeff[1:])), f.T)\
                                    + self.mu * np.sum(coeff[1:]) * dS.T #np.sum(np.einsum('i,jk->ijk', coeff, dS), axis=0)
        else:
            self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(v, np.diag(coeff[1:])), f.T)


        # orthogonalize weight matrix (A)
        try:
            D, V = eigh(np.matmul(self.state.icaweights, self.state.icaweights.T))
        except LinAlgError:
            raise Exception()

        # curr_state = self.state.icaweights
        self.state.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                    V.T), self.state.icaweights)

        # print('ciao')

        # # orthogonalize weight matrix (A.T)
        # try:
        #     D, V = eigh(np.matmul(self.state.icaweights.T, self.state.icaweights))
        # except LinAlgError:
        #     raise Exception()
        #
        # # curr_state = self.state.icaweights
        # self.state.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
        #                                             V.T), self.state.icaweights.T).T


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


def computeRegularizationFactor(W, mode='L1', return_value=True, **kwargs):
    '''

    Parameters
    ----------
    W:      unmixing matrix
    mode:   L1 - L2 - smooth - simple_smooth
    return_value
    kwargs

    Returns
    -------
    dS: reg matrix for W
    S: reg value

    '''
    if mode == 'L1' or mode == 'L2':
        if 'M_1' not in kwargs.keys():
            raise Exception('Pass M_1 term for L1 or L2 norm')
        else:
            M_1 = kwargs['M_1']
    elif mode == 'smooth' or mode == 'smooth_simple':
        if 'adj_graph' not in kwargs.keys():
            raise Exception('Pass adj_graph term for smoothing')
        else:
            adj_graph = kwargs['adj_graph']

    # W: rows = sources, cols = measurements
    # A_cap = M_1 * W.T: rows = sources, cols = measurements
    # i-j is cicle over W, k-l is cycle over A_cap for each w_ij

    if mode == 'L1':
        A_cap = np.matmul(M_1, W.T).T
        dS = 2 * np.matmul(np.sign(A_cap.T), M_1)
        if return_value:
            S = np.sum(A_cap)

    elif mode == 'L2':
        A_cap = np.matmul(M_1, W.T).T
        dS = 2 * np.matmul(A_cap.T, M_1)
        if return_value:
            S = np.sum(A_cap**2)



    elif mode == 'smooth':
        A_cap = np.matmul(M_1, W.T).T
        A_der = np.zeros(A_cap.shape)

        for i in range(A_der.shape[0]):
            for j in range(A_der.shape[0]):
                for k in range(A_der.shape[0]):
                    for l in range(A_der.shape[1]):
                        for n in adj_graph[l]:
                            if l == i or n == i:
                                A_der[i, j] += (A_cap[k, l] - A_cap[k, n])*M_1[k, j]
        A_adj = np.zeros(W.shape)
        S = []
        for i, comp in enumerate(A_adj):
            ss = 0
            for j, adj in enumerate(adj_graph):
                s_mat[i, j] = 1. / len(adj) * np.sum(A_cap[i, adj])
                if return_value:
                    ss += 1. / len(adj) * np.sum([(A_cap[i, j] - A_cap[i, ad]) ** 2 for ad in adj])
            S.append(ss)
        dS = 2. * (A_cap - A_adj)  * (M_1 - A_der)

    elif mode == 'smooth_simple':
        W_mat = np.zeros(W.shape)
        S = []
        for i, comp in enumerate(W_mat):
            ss = 0
            for j, adj in enumerate(adj_graph):
                W_mat[i, j] = 1. / len(adj) * np.sum(W[i, adj])
                if return_value:
                    ss += 1. / len(adj) * np.sum([(W[i, j] - W[i, ad]) ** 2 for ad in adj])
            S.append(ss)
        dS = 2. * (W - W_mat)

    return dS.T, S



def instICA(X, n_comp='all', n_chunks=1, chunk_size=None, numpass=1, block_size=2000, mode='original',
            adjacency_graph=None, mu=0):
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

    if mode == 'original':
        orica = ORICA(X_reduced, ndim=n_comp, sphering='offline', verbose=True, numpass=numpass,
                      block_white=block_size, block_ica=block_size, adjacency=adjacency_graph, mu=mu)
    elif mode == 'W_block':
        orica = ORICA(X_reduced, ndim=n_comp, sphering='offline', verbose=True, numpass=numpass,
                      block_white=block_size, block_ica=block_size, adjacency=adjacency_graph, mu=mu)
    elif mode == 'A_block':
        orica = ORICA(X_reduced, ndim=n_comp, sphering='offline', verbose=True, numpass=numpass,
                      block_white=block_size, block_ica=block_size, adjacency=adjacency_graph, mu=mu)
    else:
        raise Exception('Unrecognized orica type')

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
    #TODO reduce number of sources to n_chan (not L*n_chan)
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
    import sys
    from os.path import join
    from scipy import stats
    import MEAutility as MEA
    from tools import *
    import yaml
    from spike_sorting import plot_mixing

    debug=True

    if debug:
        # folder = 'recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_physrot_Neuronexus-32-cut-30_10_' \
        #              '10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_10-05-2018:11:37_3002/'
        # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/SqMEA-10-15um/recording_ica_physrot' \
        #          '_SqMEA-10-15um_20_20.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_22-05-2018:14:45_25'
        folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_' \
                 'physrot_Neuronexus-32-cut-30_15_60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_29-05-2018:16:38_2416'
        block = 1000
        recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)
        electrode_name = info['General']['electrode name']
        fs = info['General']['fs']
        nsec_window = 10
        nsamples_window = int(fs.rescale('Hz').magnitude * nsec_window)
        mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)

        templates = np.load(join(folder, 'templates.npy'))
        mixing = np.load(join(folder, 'mixing.npy')).T
        sources = np.load(join(folder, 'sources.npy'))
        adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
        n_sources = sources.shape[0]
        # lambda_val = 0.0001
        lambda_val = 0.995

        ori = onlineORICAss(recordings, fs=fs, forgetfac='cooling', skew_thresh=0.8, lambda_0=lambda_val, verbose=True,
                            block=block, step_size=1, window=5, initial_window=0, detect_trheshold=10,
                            onlineDetection=False)

        # ori = ORICA(recordings, sphering='offline', forgetfac='cooling',
        #             verbose=True, numpass=1, block_white=block, block_ica=block)

    # if len(sys.argv) == 1:
    #     folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/recording_physrot_Neuronexus-32-cut-30_' \
    #              '10_10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulated_24-01-2018_22_00'
    #     filename = join(folder, 'kilosort/raw.dat')
    #     recordings = np.fromfile(filename, dtype='int16')\
    #         .reshape(-1, 30).transpose()
    #
    #     rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
    #     with open(join(folder, rec_info), 'r') as f:
    #         info = yaml.load(f)
    #
    #     electrode_name =info['General']['electrode name']
    #     mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)
    #
    #     templates = np.load(join(folder, 'templates.npy'))
    #
    # else:
    #     folder = sys.argv[1]
    #     recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
    #     rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
    #     with open(join(folder, rec_info), 'r') as f:
    #         info = yaml.load(f)
    #
    #     electrode_name = info['General']['electrode name']
    #     mea_pos, mea_dim, mea_pitch = MEA.return_mea(electrode_name)
    #
    #     templates = np.load(join(folder, 'templates.npy'))
    #
    # block_size = 200
    # orica_type = 'A_block' # original - W - A -  W_block - A_block
    # forgetfac='constant'
    # lambda_val = 1. / recordings.shape[1] # 0.995
    # # lambda_val = 0.995
    #
    # adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
    # # ori = ORICAsimple(recordings, sphering='offline', forgetfac='constant',
    # #             verbose=True, numpass=5, block_white=1000, block_ica=1000)
    # if orica_type == 'original':
    #     ori = ORICA(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
    #                 mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    # elif orica_type == 'W':
    #     ori = ORICA_W(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
    #                       mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    # elif orica_type == 'A':
    #     ori = ORICA_A(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
    #                 mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    # elif orica_type == 'W_block':
    #     ori = ORICA_W_block(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
    #                 mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    # elif orica_type == 'A_block':
    #     ori = ORICA_A_block(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
    #                 mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    # else:
    #     raise Exception('ORICA type not understood')
    #
    # y = ori.y
    # w = ori.unmixing
    # m = ori.sphere
    # a = ori.mixing
    #
    # # Skewness
    # skew_thresh = 0.1
    # sk = stats.skew(y, axis=1)
    # high_sk = np.where(np.abs(sk) >= skew_thresh)
    #
    # # Kurtosis
    # ku_thresh = 1
    # ku = stats.kurtosis(y, axis=1)
    # high_ku = np.where(np.abs(ku) >= ku_thresh)
    #
    # # Smoothness
    # a /= np.max(a)
    # smooth = np.zeros(a.shape[0])
    # for i in range(len(smooth)):
    #    smooth[i] = (np.mean([1. / len(adj) * np.sum(a[i, j] - a[i, adj]) ** 2
    #                                          for j, adj in enumerate(adj_graph)]))
    #
    # print('High skewness: ', np.abs(sk[high_sk]))
    # print('Average high skewness: ', np.mean(np.abs(sk[high_sk])))
    # print('Number high skewness: ', len(sk[high_sk]))
    #
    # print('High kurtosis: ', ku[high_ku])
    # print('Average high kurtosis: ', np.mean(ku[high_ku]))
    # print('Number high kurtosis: ', len(ku[high_ku]))
    #
    # print('Smoothing: ', smooth[high_sk])
    # print('Average smoothing: ', np.mean(smooth[high_sk]))
    #
    # f = plot_mixing(a[high_sk], mea_pos, mea_dim)
    # f.suptitle('ORICA ' + orica_type + ' ' + str(block_size))
    # plt.figure();
    # plt.plot(y[high_sk].T)
    # plt.title('ORICA ' + orica_type + ' ' + str(block_size))
