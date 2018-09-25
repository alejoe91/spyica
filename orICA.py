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
from tools import whiten_data


def gha_step(lambd, U, x, gamma, q='all', center=False, sort=True):

    U_n1 = U + gamma * np.matmul(np.dot(x[:, np.newaxis], x[np.newaxis, :]), U)
    lambd_n1 = np.zeros_like(lambd)
    # phi = np.zeros_like(lambd)
    # for i, (l, u) in enumerate(zip(lambd, U_n1)):
    #     phi[i] = np.dot(x.T, u)
    #     sum = np.sum([phi[j] * U[j] for j in range(i-1)], axis=0)
    #     U_n1[i] = u + gamma * phi[i] * (x - phi[i] * u - sum)
    #     lambd_n1[i] =  l + gamma * (phi[i] ** 2 - l)

    return U_n1, lambd_n1


# warnings.filterwarnings('error')

# np.seterr(all='warn')

class State():
    def __init__(self, icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign,
                 Vn=None, whiteIdx=None):
        self.icaweights = icaweights
        self.icasphere = icasphere
        self.lambda_k = lambda_k
        self.minNonStatIdx = minNonStatIdx
        self.counter = counter
        self.Rn = Rn
        self.nonStatIdx = nonStatIdx
        self.kurtsign = kurtsign

class ORICA():
    def __init__(self, data, numpass=1, weights=None, onlineWhitening=False, ndim='all', lambda_0=0.995, block_white=8,
                 block_ica=8, nsub=0, white_mode='pca', pcaonly=False,
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
        self.tracking = True

        numPass = numpass
        verbose = verbose

        # Parameters for data whitening
        blockSizeWhite = block_white
        blockSizeICA = block_ica
        numSubgaussian = nsub

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        self.adjacency = adjacency
        self.mu = mu
        self.eta = eta

        self.lambdas = np.array([])


        ##############################
        # initialize state variables #
        ##############################
        if weights is None:
            if self.ndim == 'all':
                icaweights = np.eye(nChs)
            else:
                icaweights = np.eye(self.ndim, self.ndim)

        if onlineWhitening:
            if self.ndim == 'all':
                icasphere = np.eye(nChs)
            else:
                icasphere = np.eye(self.ndim, nChs)

        self.counter       = 1

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-np.exp(-1 / (self.adaptiveFF['tau_const']))
            self.lambda_k = np.zeros(blockSizeICA)
        else: # adaptive
            self.lambda_k = self.adaptiveFF['lambdaInitial'] * np.ones(blockSizeICA)

        if self.evalConvergence['profile']:
            self.Rn =[]
            self.nonStatIdx = []
            self.minNonStatIdx = -1
            self.Vn=[]
            self.whiteIdx=[]

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        if self.ndim == 'all':
            self.kurtsign = np.ones((nChs, 1))
        else:
            self.kurtsign = np.ones((self.ndim, 1))
        if numSubgaussian != 0:
            self.kurtsign[:numSubgaussian] = 0


        ######################
        # sphere-whiten data #
        ######################
        if not onlineWhitening: # pre - whitening
            if verbose:
                print('Use pre-whitening method.')
            if self.whiten:
                if self.ndim == 'all':
                    if white_mode == 'pca':
                        print('PCA whitening')
                        data_w, eigvecs, eigvals, sphere = whiten_data(data)
                        icasphere = sphere
                    elif white_mode == 'zca':
                        print('ZCA whitening')
                        # TODO use SVD and compute ZCA VS PCA
                        icasphere = la.inv(sqrtm(np.cov(data)))
                        data_w = np.matmul(icasphere, data)
                else:
                    # data_w = np.matmul(icasphere, data)
                    data_w, eigvecs, eigvals, sphere = whiten_data(data, self.ndim)
                    icasphere = sphere
            else:
                print('Initializing weights to sphering matrix')
                data_w = np.zeros_like(data, dtype=dtype)
                icaweights = icasphere
        else: # Online RLS Whitening
            if verbose:
                print('Use online whitening method.')
            if self.ndim == 'all':
                data_w = np.zeros_like(data)
                n_samples = int(0.1 * 32000)

                print('initializing eigenvectors and values')
                data_init = data[:, :n_samples]
                _, eigvecs, eigvals, sphere = whiten_data(data_init)

                self.eigvecs = eigenvecs
                self.eigvals = eigenvals
            else:
                data_w = np.zeros((self.ndim, nPts))


        # whiten / sphere the data


        # self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, counter, Rn, nonStatIdx, kurtsign)
        self.icaweights = icaweights
        self.icasphere = icasphere
        self.icasphere_1 = la.pinv(self.icasphere)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / np.min([blockSizeICA, blockSizeWhite])))

        self.NSI = []
        self.WI = []

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
                    self.dynamicWhitening(data[:, dataRange], dataRange)
                    #self.dynamicPCA(data[:, dataRange], dataRange)
                    data_w[:, dataRange] = np.matmul(self.icasphere, data[:, dataRange])

                if not pcaonly:
                    self.dynamicOrica(data_w[:, dataRange], dataRange)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        self.sphere = self.icasphere
        if not pcaonly:
            if self.whiten:
                self.unmixing = np.matmul(self.icaweights, self.sphere)
            else:
                self.unmixing = self.icaweights
            self.mixing = la.pinv(self.unmixing).T
            self.y = np.matmul(self.unmixing, data)


    def dynamicWhitening(self, blockdata, dataRange):
        nChs, nPts = blockdata.shape

        if self.adaptiveFF['profile'] == 'cooling':
            lambda_ = self.genCoolingFF(self.counter+dataRange, self.adaptiveFF['gamma'], self.adaptiveFF['lambda_0'])
            if lambda_[0] < self.adaptiveFF['lambda_const']:
                lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'constant':
            lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'adaptive':
            lambda_ = np.squeeze(np.tile(self.lambda_k[-1], (1, nPts)))

        v = np.matmul(self.icasphere, blockdata) # pre - whitened data

        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(v, v.T)/nPts
            if len(self.Vn) == 0:
                self.Vn = modelFitness
            else:
                self.Vn = (1 - self.evalConvergence['leakyAvgDelta']) * self.Vn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.whiteIdx = la.norm(self.Vn)
            self.WI.append(self.whiteIdx)

        lambda_avg = 1 - lambda_[int(np.ceil((len(lambda_)-1) / 2))] # median lambda
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
        self.icasphere = 1 / lambda_avg * (self.icasphere - np.matmul(np.matmul(v, v.T) / nPts / QWhite, self.icasphere))

    def dynamicPCA(self, blockdata, dataRange, method='gha'):
        nChs, nPts = blockdata.shape

        if self.adaptiveFF['profile'] == 'cooling':
            lambda_ = self.genCoolingFF(self.counter+dataRange, self.adaptiveFF['gamma'], self.adaptiveFF['lambda_0'])
            if lambda_[0] < self.adaptiveFF['lambda_const']:
                lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'constant':
            lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
        elif self.adaptiveFF['profile'] == 'adaptive':
            lambda_ = np.squeeze(np.tile(self.lambda_k[-1], (1, nPts)))

        v = np.matmul(self.icasphere, blockdata) # pre - whitened data

        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(v, v.T)/nPts
            if len(self.Vn) == 0:
                self.Vn = modelFitness
            else:
                self.Vn = (1 - self.evalConvergence['leakyAvgDelta']) * self.Vn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.whiteIdx = la.norm(self.Vn)
            self.WI.append(self.whiteIdx)

        if method == 'gha':
            U = self.eigvecs
            U_n1 = U + lambda_[0] * np.matmul(np.dot(blockdata, blockdata.T)/nPts, U)
            self.eigvecs = U_n1
            # for (x, g) in zip(blockdata.T, lambda_):
            #     self.eigvecs, self.eigvals = gha_step(self.eigvals, self.eigvecs, x, g)

        self.icasphere = np.matmul(np.diag(1./np.sqrt(self.eigvals)), self.eigvecs.T)


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        # compute source activation using previous weight matrix
        y = np.matmul(self.icaweights, blockdata)

        # raise Exception()

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.kurtsign==1), :])  # Supergaussian
            f[np.where(self.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            # modelFitness = np.eye(nChs) + np.matmul(y, f.T)/nPts
            modelFitness = np.eye(nChs) + np.matmul(y, f.T)/nPts
            if len(self.Rn) == 0:
                self.Rn = modelFitness
            else:
                self.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.nonStatIdx = la.norm(self.Rn)
            self.NSI.append(self.nonStatIdx)


        if self.adaptiveFF['profile'] == 'cooling':
            self.lambda_k = self.genCoolingFF(self.counter + dataRange, self.adaptiveFF['gamma'],
                                                    self.adaptiveFF['lambda_0'])
            if self.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            self.counter = self.counter + nPts
        elif self.adaptiveFF['profile'] == 'constant':
            self.lambda_k = np.arange(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if self.minNonStatIdx != -1:
                self.minNonStatIdx = self.nonStatIdx
            self.minNonStatIdx = np.max([np.min([self.minNonStatIdx, self.nonStatIdx]), 1])
            ratioOfNormRn = self.nonStatIdx / self.minNonStatIdx
            self.lambda_k = self.genAdaptiveFF(dataRange, self.lambda_k, ratioOfNormRn)

        self.lambdas = np.concatenate((self.lambdas, self.lambda_k))

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-self.lambda_k))
        # Q = 1 + self.lambda_k * (np.einsum('km,km->m', f, y)-1);
        Q = 1 + self.lambda_k * (np.sum(f * y, axis=0) - 1);
        curr_state = self.icaweights

        self.count += 1

        # Compute smoothing factor
        if self.mu != 0:
            smoothing_matrix = np.zeros(self.icaweights.shape)
            for i, comp in enumerate(smoothing_matrix):
                for adj in self.adjacency:
                    smoothing_matrix[i] = 1./len(adj)*np.sum(self.icaweights[i, adj])

            self.icaweights = lambda_prod * ((self.icaweights -
                                                   np.matmul(np.matmul(np.matmul(y,  np.diag(self.lambda_k / Q)),
                                                             f.T), self.icaweights)) -
                                                   self.mu*(self.icaweights - smoothing_matrix)) #- eta*())
        else:
            self.icaweights = lambda_prod * (self.icaweights -
                                                   np.matmul(np.matmul(np.matmul(y, np.diag(self.lambda_k / Q)),
                                                                    f.T), self.icaweights))


        # orthogonalize weight matrix
        if self.ortho:
            try:
                D, V = eigh(np.matmul(self.icaweights, self.icaweights.T))
            except LinAlgError:
                raise Exception()

            # curr_state = self.icaweights
            self.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                        V.T), self.icaweights)
        else:
            self.icaweights = self.icaweights / np.max(np.abs(self.icaweights))

    def genCoolingFF(self, t, gamma, lambda_0):
        lambda_ = lambda_0 / (t ** gamma)
        return lambda_

    def genAdaptiveFF(self, dataRange, lambda_in, ratioOfNormRn):
        decayRateAlpha = self.adaptiveFF['decayRateAlpha']
        upperBoundBeta = self.adaptiveFF['upperBoundBeta']
        transBandWidthGamma = self.adaptiveFF['transBandWidthGamma']
        transBandCenter = self.adaptiveFF['transBandCenter']

        gainForErrors = upperBoundBeta * 0.5 * (1 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))
        nrange = np.arange(len(dataRange))
        lambda_ = np.zeros(len(dataRange))
        lam_pr = lambda_in[-1]
        for i in nrange:
            lam_new = (1 + gainForErrors) * lam_pr - decayRateAlpha * lam_pr **2
            lambda_[i] = lam_new
            lam_pr = lam_new
        return lambda_


class onlineORICAss():
    def __init__(self, data, fs, ndim='all', onlineWhitening=True, calibratePCA=True, forgetfac='cooling',
                 skew_thresh=0.5, lambda_0=0.995, min_lambda=0, block=8, nsub=0, ffdecayrate=0.6, verbose=False,
                 pcaweights=[], weights=[], step_size=1, skew_window=20,
                 pca_window=10, ica_window=0, detect_trheshold=8, onlineDetection=True, evalconverg=True, numpass=1):
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

        if numpass > 1:
            data = np.tile(data, numpass)

        nChs, nPts = data.shape
        verbose = verbose

        if isinstance(fs, pq.Quantity):
            fs = int(fs.rescale('Hz').magnitude)
        else:
            fs = int(fs)

        self.n_window = int(fs*skew_window)
        self.n_pca_window = int(fs*pca_window)
        self.n_ica_window = int(fs*ica_window)
        self.n_step_size = int(fs*step_size)
        self.n_trackin_window = int(fs * 10. / block)
        self.tracking = True

        self.detect_thresh = detect_trheshold
        self.block = block
        self.ndim = ndim

        # Parameters for data whitening
        # onlineWhitening = True
        numSubgaussian = nsub

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': ffdecayrate, 'lambda_0': lambda_0,
                           'decayRateAlpha': 0.02, 'upperBoundBeta': 1e-3, 'transBandWidthGamma': 0.3,
                           'transBandCenter': 0, 'lambdaInitial': 0.1, 'ff': 0.001, 'min_lambda': min_lambda}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.005, 'leakyAvgDeltaVar': 1e-3}


        ##############################
        # initialize state variables #
        ##############################
        if len(weights) == 0:
            if self.ndim == 'all':
                icaweights = np.eye(nChs)
            else:
                icaweights = np.eye(self.ndim, self.ndim)
        else:
            icaweights = weights

        # if len(pcaweights) == 0:
        # if onlineWhitening:
        icasphere = np.eye(nChs)
        # else:
        #     onlineWhitening = False

        if self.adaptiveFF['profile'] == 'adaptive':
            self.lambda_k = self.adaptiveFF['lambdaInitial'] * np.ones(block)
        else:
            self.lambda_k = np.zeros(block)

        if self.evalConvergence['profile']:
            self.Rn =[]
            self.nonStatIdx =[]
            self.minNonStatIdx = -1
            self.Vn=[]
            self.whiteIdx=[]
        self.counter       = 1

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-np.exp(-1 / (self.adaptiveFF['tau_const']))

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        if self.ndim == 'all':
            self.kurtsign = np.ones((nChs, 1))
        else:
            self.kurtsign = np.ones((self.ndim, 1))
        if numSubgaussian != 0:
            self.kurtsign[:numSubgaussian] = 0

        # online estimation
        self.N = 0
        self.mus = []
        self.vars = []
        self.sigmas = []
        self.skews = []

        ######################
        # sphere-whiten data #
        ######################
        if not calibratePCA:
            if not onlineWhitening:  # pre - whitening
                if verbose:
                    print('Use pre-whitening method.')
                if self.ndim == 'all':
                    _, eigvecs, eigvals, sphere = whiten_data(data)
                    icasphere = sphere
                else:
                    _, eigvecs, eigvals, sphere = whiten_data(data, self.ndim)
                    icasphere = sphere
                    # data_w = np.zeros((self.ndim, nPts), dtype=float)
                self.n_pca_window = 0
            else:  # Online RLS Whitening
                if verbose:
                    print('Use online whitening method.')
        else:
            if verbose:
                print('Use initial PCA calibration method.')
            self.pca_calibrated = False

        self.icaweights = icaweights
        self.icasphere = icasphere
        self.icasphere_1 = la.pinv(self.icasphere)
        self.skew_thresh = skew_thresh
        if self.ndim == 'all':
            self.y_on = np.zeros_like(data, dtype=float)
        else:
            self.y_on = np.zeros((self.ndim, nPts), dtype=float)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / block))

        self.w = []
        self.m = []
        self.idx_sources = []
        self.means = np.zeros(nChs)
        self.sumx = np.zeros(nChs)
        self.ratiosumx = 0
        self.ratiosumx2 = 0
        self.iter = 0
        self.tracking_iter = 0

        if verbose:
            printflag = 0
            if self.adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif self.adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif self.adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')

        t_start = time.time()
        self.init = False
        self.spikes = {}
        self.all_sources = np.array([])
        self.nskews = []
        self.NSI = []
        self.WI = []
        self.lambdas = np.array([])
        self.ratios = np.array([])
        self.normNSI = np.array([])
        self.NSImean = 0
        self.NSIvar = 0

        for bi in range(numBlock):
            dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                              int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
            self.N += len(dataRange)
            self.onlineMean(data[:, dataRange])

            if calibratePCA:
                self.calibratePCA(data)

            if onlineWhitening:
                self.dynamicWhitening(data[:, dataRange], dataRange)
            else:
                # compute WI
                data_cent = data[:, dataRange] - self.means
                self.dynamicWhitening(data_cent, dataRange, whitening=False)

            if self.N > self.n_pca_window:
                data_cent = data[:, dataRange] - self.means
                if self.ndim != 'all':
                    data_white = np.matmul(self.icasphere[:self.ndim], data_cent)
                else:
                    data_white = np.matmul(self.icasphere, data_cent)
                self.dynamicOrica(data_white, dataRange)

            if self.N > self.n_pca_window + self.n_ica_window:
                # online sources
                self.y_on[:, dataRange] = np.matmul(self.icaweights, np.matmul(self.icasphere, data_cent))

            # select sources
            if not np.mod(self.N, self.n_step_size):
                if self.N > self.n_pca_window + self.n_ica_window:
                    self.computeSkew()
                    idx_sources = np.where(np.abs(self.skew) > self.skew_thresh)
                    self.w.append(self.icaweights)
                    if self.ndim != 'all':
                        self.m.append(self.icasphere[:self.ndim])
                    else:
                        self.m.append(self.icasphere)

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
                    self.w.append(self.icaweights)
                    if self.ndim != 'all':
                        self.m.append(self.icasphere[:self.ndim])
                    else:
                        self.m.append(self.icasphere)
                    self.idx_sources.append([])

            if verbose:
                if printflag < np.floor(10 * bi / numBlock):
                    printflag = printflag + 1
                    print(10 * printflag, '%')
                    if self.N > self.n_pca_window + self.n_ica_window and len(self.idx_sources) != 0:
                        print('Sources: ', self.idx_sources[-1])


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


    def computeSkew(self):
        if self.N < self.n_window:
            y = self.y_on[:, :self.N]
            # skewness for source selection
            self.skew = stats.skew(y, axis=1)
            # self.sigma = np.std(y, axis=1)
        else:
            # skewness for source selection
            y = self.y_on[:, self.N-self.n_window:self.N]
            self.skew = stats.skew(y, axis=1)
            # self.sigma = np.std(self.y, axis=1)
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


    def onlineMean(self, blockdata):
        self.means = 1. / self.N * (self.sumx[:, np.newaxis] + np.sum(blockdata, axis=1, keepdims=True))
        self.sumx += np.sum(blockdata, axis=1)

    def onlineNSIUpdate(self, newNSI, ff=None):
        if ff == None:
            if self.iter > 1:
                ff = 1. /(self.iter - 1)
            else:
                ff = 1.

        if self.iter > 1:
            self.NSImean = (1 - ff) * self.NSImean + ff * newNSI
            self.NSIvar = (1 - ff) * self.NSIvar + ff * (1 - ff) * (newNSI - self.NSImean) ** 2
        else:
            self.NSImean = newNSI
            self.NSIvar = 1.

        normNSI = (newNSI - self.NSImean) / np.sqrt(self.NSIvar)
        # if normNSI > self.adaptiveFF['transBandCenter']:
        #     self.tracking = True
        # else:
        #     if self.tracking:
        #         self.tracking = False
        #         self.tracking_iter = 0
        #         print('STOP TRACKING')

        return normNSI


    def calibratePCA(self, data):
        if self.N >= self.n_pca_window and not self.pca_calibrated:
            data_init = data[:, :self.n_pca_window]
            print('PCA calibration')
            if self.ndim == 'all':
                _, eigvecs, eigvals, sphere = whiten_data(data_init)
            else:
                _, eigvecs, eigvals, sphere = whiten_data(data_init, self.ndim)
            self.icasphere = sphere
            self.pca_calibrated = True


    def dynamicWhitening(self, blockdata, dataRange, whitening=True):
        nChs, nPts = blockdata.shape

        v = np.matmul(self.icasphere, blockdata) # pre - whitened data

        if self.evalConvergence['profile']:
            modelFitness = np.eye(self.icaweights.shape[0]) - np.matmul(v, v.T)/nPts
            if len(self.Vn) == 0:
                self.Vn = modelFitness
            else:
                self.Vn = (1 - self.evalConvergence['leakyAvgDelta']) * self.Vn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.whiteIdx = la.norm(self.Vn)
            self.WI.append(self.whiteIdx)

        if whitening:
            if self.adaptiveFF['profile'] == 'cooling':
                lambda_ = self.genCoolingFF(self.counter + dataRange, self.adaptiveFF['gamma'],
                                            self.adaptiveFF['lambda_0'],
                                            self.adaptiveFF['min_lambda'])
                if lambda_[0] < self.adaptiveFF['lambda_const']:
                    lambda_ = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            elif self.adaptiveFF['profile'] == 'constant':
                lambda_ = np.ones(nPts) * self.adaptiveFF['lambda_0']

            lambda_avg = 1 - lambda_[int(np.ceil(len(lambda_) / 2))] # median lambda
            QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
            self.icasphere = 1 / lambda_avg * (self.icasphere - np.matmul(np.matmul(v, v.T) / nPts
                                                / QWhite,self.icasphere))


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        # compute source activation using previous weight matrix
        y = np.matmul(self.icaweights, blockdata)

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(self.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.kurtsign==1), :])  # Supergaussian
            f[np.where(self.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) - np.matmul(y, f.T) / nPts
            # modelFitness = (np.eye(nChs) - np.matmul(y, f.T) / nPts) / nChs
            # variance = blockdata * blockdata
            if len(self.Rn) == 0:
                self.Rn = modelFitness
            else:
                self.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                # !!! this does not account for block update!
            self.nonStatIdx = la.norm(self.Rn)
            # self.nonStatIdx = la.norm(modelFitness)
            self.NSI.append(self.nonStatIdx)
            if self.N > self.n_pca_window + self.n_ica_window:
                self.iter += 1
                nsinorm = self.onlineNSIUpdate(self.nonStatIdx, ff=self.adaptiveFF['ff'])
                self.normNSI = np.append(self.normNSI, nsinorm)
            else:
                self.normNSI= np.append(self.normNSI, 0)

        if self.adaptiveFF['profile'] == 'cooling':
            self.lambda_k = self.genCoolingFF(self.counter + dataRange, self.adaptiveFF['gamma'],
                                              self.adaptiveFF['lambda_0'], self.adaptiveFF['min_lambda'])
            if self.lambda_k[0] < self.adaptiveFF['lambda_const']:
                self.lambda_k = np.squeeze(np.tile(self.adaptiveFF['lambda_const'], (1, nPts)))
            self.counter = self.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.lambda_k = np.ones(nPts) * self.adaptiveFF['lambda_0']
        elif self.adaptiveFF['profile'] == 'adaptive':
            if self.minNonStatIdx == -1:
                self.minNonStatIdx = self.nonStatIdx
            self.minNonStatIdx = np.max([np.min([self.minNonStatIdx, self.nonStatIdx]), 1])
            ratioOfNormRn = self.nonStatIdx / self.minNonStatIdx
            # self.lambda_k = self.genAdaptiveFF(dataRange, self.lambda_k, ratioOfNormRn)
            self.lambda_k = self.genAdaptiveFF(dataRange, self.lambda_k, self.normNSI[-1])
            self.ratios = np.append(self.ratios, ratioOfNormRn)

        self.lambdas = np.concatenate((self.lambdas, self.lambda_k))


        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-self.lambda_k))
        Q = 1 + self.lambda_k * (np.sum(f * y, axis=0) - 1);
        curr_state = self.icaweights

        self.icaweights = lambda_prod * (self.icaweights -
                                               np.matmul(np.matmul(np.matmul(y, np.diag(self.lambda_k / Q)),
                                                                f.T), self.icaweights))

        if np.any(np.isnan(np.matmul(self.icaweights, self.icaweights.T))) or \
                np.any(np.isinf(np.matmul(self.icaweights, self.icaweights.T))):
            raise Exception()

        # orthogonalize weight matrix
        try:
            D, V = eigh(np.matmul(self.icaweights, self.icaweights.T))
        except LinAlgError:
            raise Exception()

        # curr_state = self.icaweights
        self.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                    V.T), self.icaweights)

    def genCoolingFF(self, t, gamma, lambda_0, min_lambda):
        lambda_ = lambda_0 / (t ** gamma)
        if min_lambda != 0:
            lambda_[lambda_<min_lambda] = min_lambda
        return lambda_

    def genAdaptiveFF(self, dataRange, lambda_in, ratioOfNormRn):
        decayRateAlpha = self.adaptiveFF['decayRateAlpha']
        upperBoundBeta = self.adaptiveFF['upperBoundBeta']
        transBandWidthGamma = self.adaptiveFF['transBandWidthGamma']
        transBandCenter = self.adaptiveFF['transBandCenter']

        gainForErrors = upperBoundBeta * 0.5 * (1 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))
        nrange = np.arange(len(dataRange))

        lambda_ = np.zeros(len(dataRange))
        lam_pr = lambda_in[-1]
        for i in nrange:
            lam_new = (1+gainForErrors) * lam_pr - decayRateAlpha * lam_pr ** 2
            # lam_new = (1./((i+1)**self.adaptiveFF['gamma']) + gainForErrors) * lam_pr
            lambda_[i] = lam_new
            lam_pr = lam_new

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
        self.state.icasphere = 1 / lambda_avg * (self.state.icasphere - np.matmul(np.matmul(v, v.T) / nPts
                                            / QWhite, self.state.icasphere))

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
        orica = ORICA(X_reduced, ndim=n_comp, onlineWhitening=False, verbose=True, numpass=numpass,
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
        # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/Neuronexus-32-cut-30/recording_ica_' \
        #          'physrot_Neuronexus-32-cut-30_15_60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_noise-all_29-05-2018:16:38_2416'
        # folder= 'recordings/convolution/gtica/SqMEA-10-15um/recording_ica_physrot_SqMEA-10-15um_10_20.0s_' \
        #         'uncorrelated_5.0_5.0Hz_15.0Hz_modulation_none_13-06-2018:10:22_7593/'
        # folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/gtica/Neuronexus-32-cut-30/recording_' \
        #          'ica_physrot_Neuronexus-32-cut-30_10_60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_none_04-06-2018:13:29_4236'
        folder = 'recordings/convolution/intermittent/Neuronexus-32-cut-30/recording_physrot_Neuronexus-32-cut-30_10_' \
                 '60.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulation_none_27-06-2018:15:56_1514_intermittent'
        block = 800
        recordings = np.load(join(folder, 'recordings.npy')).astype('int16')
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

        templates = np.load(join(folder, 'templates.npy'))
        mixing = np.load(join(folder, 'mixing.npy')).T
        sources = np.load(join(folder, 'sources.npy'))
        adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
        n_sources = sources.shape[0]
        # lambda_val = 0.0001
        lambda_val = 0.995
        lambda_val = 0.0078

        raise Exception()

        orio = onlineORICAss(recordings, fs=fs, forgetfac='constant', skew_thresh=0.8, lambda_0=lambda_val, verbose=True,
                             block=block, step_size=1, skew_window=5, detect_trheshold=10, onlineWhitening=False,
                             calibratePCA=False, ica_window=15, pca_window=0,
                             onlineDetection=False, numpass=1)

        # ori = ORICA(recordings, onlineWhitening=False, forgetfac='adaptive', pcaonly=False, lambda_0=lambda_val,
        #             verbose=True, numpass=1, block_white=block, block_ica=block)


        print('ciao')
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
