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
        minNonStatIdx = []
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
        self.unmixing = np.matmul(self.state.icaweights, self.sphere)
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
                    dS = computeSmoothingFactor(A, self.icasphere_1, self.neighbor_mask)
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

    def computeSmoothingFactor(self):
        A = self.state.icaweights
        M_1 = self.icasphere_1
        s_der = np.zeros(W.shape)
        for i in range(s_der.shape[0]):
            for j, adj in enumerate(adj_graph):
                if i in adj:
                    # print(i, j, adj_graph[j])
                    s_der[i, j] = 1. / len(adj) * M_1[i, j]

        s_mat = np.zeros(W.shape)
        for i, comp in enumerate(s_mat):
            for j, adj in enumerate(adj_graph):
                s_mat[i, j] = 1. / len(adj) * np.sum(A[i, adj])

        dS = 2. * (A - s_mat) * (M_1 - s_der)
        dS_T = dS.T

        return dS_T

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
        self.smooth_count = 0
        self.S = []

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
        self.S = np.array(self.S)


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


        coeff = np.append(1, self.state.lambda_k) * np.append(1,np.cumprod(1 - self.state.lambda_k[::-1]))[::-1]

        # Compute smoothing factor
        if self.mu != 0:
            self.smooth_count += 1

            # if not np.mod(self.smooth_count, 20):
                # A_cap = np.matmul(self.icasphere_1, A)
            dS, S = computeSmoothingFactor(W, self.adjacency)
            self.smooth_count = 0
            self.glob_count += 1
            self.S.append(S)
            # dS = np.zeros(A.shape)
                # print(self.glob_count)
                # if self.glob_count == 50:
                #     raise Exception()
            # else:
            #     dS = np.zeros(A.shape)
            self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(f, np.diag(coeff[1:])), v.T)\
                                    + self.mu * np.sum(coeff[1:]) * dS #np.sum(np.einsum('i,jk->ijk', coeff, dS), axis=0)

            # if not np.mod(self.smooth_count, 100):
            #     ica_coeff = coeff[0] * self.state.icaweights + np.matmul(np.matmul(f, np.diag(coeff[1:])), v.T)
            #     smooth_coeff = self.mu * np.sum(coeff[1:]) * dS
            #     print('ICA ', np.max(ica_coeff), np.min(ica_coeff))
            #     print('Smooth ', np.max(smooth_coeff), np.min(smooth_coeff))

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
        self.state.icaweights = coeff[0] * self.state.icaweights + np.matmul(np.matmul(v, np.diag(coeff[1:])), f.T)

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


def computeSmoothingFactor(W, adj_graph, mode='L1', simple=True, return_smoothing=True):
    '''

    Parameters
    ----------
    A
    M_1

    Returns
    -------

    '''
    # # i-j is cicle over W, k-l is cycle over A for each w_ij
    # t1 = time.time()
    # s_der = np.zeros(A.shape)
    # for i in range(s_der.shape[0]):
    #     for j in range(s_der.shape[0]):
    #         for k in range(s_der.shape[0]):
    #             for l in range(s_der.shape[1]):
    #                 for n in adj_graph[l]:
    #                     if l == i or n == i:
    #                         s_der[i, j] += (A[k, l] - A[k, n])*M_1[k, j]
    # print (time.time() - t1)
    #
    #
    #
    # dS = np.zeros(A.shape)
    # for nn in masks:
    #     for (i, j, k, l, n) in nn:
    #         dS[i, j] += (A[k, l] - A[k, n])*M_1[k, j]

    s_mat = np.zeros(W.shape)
    for i, comp in enumerate(s_mat):
        for j, adj in enumerate(adj_graph):
            s_mat[i, j] = 1. / len(adj) * np.sum(W[i, adj])

    dS = 2. * (W - s_mat) #* (M_1 - s_der)
    S = []
    if return_smoothing:
        for i, comp in enumerate(s_mat):
            ss = 0
            for j, adj in enumerate(adj_graph):
                ss += 1. / len(adj) * np.sum([(W[i,j] - W[i, ad])**2 for ad in adj])
            S.append(ss)

    return dS, S



def instICA(X, n_comp='all', n_chunks=1, chunk_size=None, numpass=1, block_size=2000, adjacency_graph=None, mu=0):
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

    orica = ORICA(X_reduced, sphering='offline', verbose=True, numpass=numpass,
                  block_white=block_size, block_ica=block_size, adjacency=adjacency_graph, mu=mu)

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

    if len(sys.argv) == 1:
        folder = '/home/alessio/Documents/Codes/SpyICA/recordings/convolution/recording_physrot_Neuronexus-32-cut-30_' \
                 '10_10.0s_uncorrelated_10.0_5.0Hz_15.0Hz_modulated_24-01-2018_22_00'
        filename = join(folder, 'kilosort/raw.dat')
        recordings = np.fromfile(filename, dtype='int16')\
            .reshape(-1, 30).transpose()

        rec_info = [f for f in os.listdir(folder) if '.yaml' in f or '.yml' in f][0]
        with open(join(folder, rec_info), 'r') as f:
            info = yaml.load(f)

        electrode_name =info['General']['electrode name']
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

    block_size = 200
    orica_type = 'A_block' # original - W - A -  W_block - A_block
    forgetfac='constant'
    lambda_val = 1. / recordings.shape[1] # 0.995
    # lambda_val = 0.995

    adj_graph = extract_adjacency(mea_pos, np.max(mea_pitch) + 5)
    # ori = ORICAsimple(recordings, sphering='offline', forgetfac='constant',
    #             verbose=True, numpass=5, block_white=1000, block_ica=1000)
    if orica_type == 'original':
        ori = ORICA(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
                    mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    elif orica_type == 'W':
        ori = ORICA_W(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
                          mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    elif orica_type == 'A':
        ori = ORICA_A(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
                    mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    elif orica_type == 'W_block':
        ori = ORICA_W_block(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
                    mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    elif orica_type == 'A_block':
        ori = ORICA_A_block(recordings, sphering='offline', forgetfac=forgetfac, lambda_0=lambda_val,
                    mu=0, verbose=True, numpass=1, block_white=block_size, block_ica=block_size)
    else:
        raise Exception('ORICA type not understood')

    y = ori.y
    w = ori.unmixing
    m = ori.sphere
    a = ori.mixing

    # Skewness
    skew_thresh = 0.1
    sk = stats.skew(y, axis=1)
    high_sk = np.where(np.abs(sk) >= skew_thresh)

    # Kurtosis
    ku_thresh = 1
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
    f.suptitle('ORICA ' + orica_type + ' ' + str(block_size))
    plt.figure();
    plt.plot(y[high_sk].T)
    plt.title('ORICA ' + orica_type + ' ' + str(block_size))