# % orica()  - Perform Online Recursive Independent Component Analysis (ORICA) decomposition
# %            of input data with optional Online Recursive Least Square Whitening.
# % Usage:
# %         >> [weights,sphere] = orica(data); % train using defaults
# %    else
# %         >> [weights,sphere] = orica(data,'Key1',Value1',...);
# % Input:
# %   data     = input data (chans-by-samples)
# %
# % Optional Keywords [argument]:
# % 'weights'     = [W] initial weight matrix     (default -> eye())
# % 'sphering'    = ['online'|'offline'] use online RLS whitening method or pre-whitening
# % 'numpass'     = [N] number of passes over input data
# % 'block_ica'   = [N] block size for ORICA (in samples)
# % 'block_white' = [N] block size for online whitening (in samples)
# % 'forgetfac'   = ['cooling'|'constant'|'adaptive'] forgetting factor profiles
# %                 'cooling': monotonically decreasing, for relatively stationary data
# %                 'constant': constant, for online tracking non-stationary data.
# %                 'adaptive': adaptive based on Nonstatinoarity Index (in dev)
# %                 See reference [2] for more information.
# % 'localstat'   = [f] local stationarity (in number of samples) corresponding to
# %                 constant forgetting factor at steady state
# % 'ffdecayrate' = [0<f<1] decay rate of (cooling) forgetting factor (default -> 0.6)
# % 'nsub'        = [N] number of subgaussian sources in EEG signal (default -> 0)
# %                 EEG brain sources are usually supergaussian
# %                 Subgaussian sources are motstly artifact or noise
# % 'evalconverg' = [0|1] evaluate convergence such as Non-Stationarity Index
# % 'verbose'     = ['on'|'off'] give ascii messages  (default -> 'off')
# %
# % Output:
# %   weights  = ICA weight matrix (comps,chans)
# %   sphere   = data sphering matrix (chans,chans)
# %              Note that unmixing_matrix = weights*sphere
# %
# % Reference:
# %       [1] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Real-time
# %       adaptive EEG source separation using online recursive independent
# %       component analysis," IEEE Transactions on Neural Systems and
# %       Rehabilitation Engineering, 2016.
# %
# %       [2] S.-H. Hsu, L. Pion-Tanachini, T.-P Jung, and G. Cauwenberghs,
# %       "Tracking non-stationary EEG sources using adaptive online
# %       recursive independent component analysis," in IEEE EMBS, 2015.
# %
# %       [3] S.-H. Hsu, T. Mullen, T.-P Jung, and G. Cauwenberghs, "Online
# %       recursive independent component analysis for real-time source
# %       separation of high-density EEG," in IEEE EMBS, 2014.
# %
# % Author:
# %       Sheng-Hsiou (Shawn) Hsu, SCCN, UCSD.
# %       shh078@ucsd.edu

import numpy as np
from numpy import linalg as la
import time
from __future__ import print_function


class State():
    def __init__(self, icaweights, icasphere, lambda_k, minNonStatIdx, count, Rn, nonStatIdx, kurtsign):
        self.icaweights = icaweights
        self.icasphere = icasphere
        self.lambda_k = lambda_k
        self.minNonStatIdx = minNonStatIdx
        self.count = count
        self.Rn = Rn
        self.nonStatIdx = nonStatIdx
        self.kurtsign = kurtsign

class ORICA():
    def __init__(self, data, numpass=1, weights=None, sphering='offline', block_white=8, block_ica=8, nsub=0,
                 forgetfac='cooling', localstat=Inf, ffdecayrate=0.6, evalconverg=True, verbose=False):
        '''

        Parameters
        ----------
        data
        numpass
        sphering
        online
        block_white
        block_ica
        nsub
        forgetfac
        cooling
        localstat
        ffdecayrate
        evalconverg
        verbose

        Returns
        -------

        '''

        nChs, nPts = data.shape

        numPass = numPass
        verbose = verbose

        # Parameters for data whitening
        onlineWhitening = online
        blockSizeWhite = block_white
        blockSizeICA = block_ica
        numSubgaussian = nsub

        self.adaptiveFF = {'profile': forgetfac, 'tau_const': Inf, 'gamma': 0.6, 'lambda_0': 0.995, 'decayRateAlpha': 0.02,
                      'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1, 'transBandCenter': 5, 'lambdaInitial': 0.1}
        self.evalConvergence = {'profile': evalconverg, 'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

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
        counter       = 0

        if self.adaptiveFF['profile'] == 'cooling' or  self.adaptiveFF['profile'] == 'constant':
            self.adaptiveFF['lambda_const']  = 1-exp(-1 / (adaptiveFF['tau_const']))

        if self.evalConvergence['profile']:
            Rn =[]
            nonStatIdx =[]

        # sign of kurtosis for each component: true(supergaussian), false(subgaussian)
        kurtsign = np.ones((nChs, 1))
        if numSubgaussian != 0:
            kurtsign[:numSubgaussian] = 0

        self.state = State(icaweights, icasphere, lambda_k, minNonStatIdx, count, Rn, nonStatIdx, kurtsign)

        ######################
        # sphere-whiten data #
        ######################
        if not onlineWhitening: # pre - whitening
            if verbose:
                print('Use pre-whitening method.')
                icasphere = 2.0 * np.invert(la.sqrtm(float(np.cov(np.transpose(data))))) # find the "sphering" matrix = spher()
        else: # Online RLS Whitening
            if verbose:
                print('Use online whitening method.')
        # whiten / sphere the data
        data = icasphere * data

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = no.floor(nPts / np.min(blockSizeICA, blockSizeWhite))

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

                dataRange = range(np.floor(bi * nPts / numBlock), min(nPts, floor((bi + 1) * nPts / numBlock)))
                if onlineWhitening:
                    dynamicWhitening(data[:, dataRange], dataRange)
                    data[:, dataRange] = self.state.icasphere * data[:, dataRange]
                dynamicOrica(data[:, dataRange], dataRange)

                if verbose:
                    if printflag < floor(10 * ((it - 1) * numBlock + bi) / numPass / numBlock):
                        printflag = printflag + 1
                        print(' %d%% ', 10 * printflag)

        if verbose:
            processing_time = time.time() - t_start
            print('Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        weights = self.state.icaweights
        sphere = self.state.icasphere

        return weights, sphere



    def dynamicWhitening(self, blockdata, dataRange):
        nPts = blockdata.shape[1]

        if self.adaptiveFF['profile'] == 'cooling':
            lambda_ = genCoolingFF(state.counter+dataRange, self.adaptiveFF['gamma'], self.adaptiveFF['lambda_0'])
            if lambda_[0] < self.adaptiveFF['lambda_const']:
                lambda_ = np.repeat(adaptiveFF['lambda_const'], 1, nPts)
        elif self.adaptiveFF['profile'] == 'constant':
            lambda_ = np.repeat(adaptiveFF['lambda_const'], 1, nPts)
        elif self.adaptiveFF['profile'] == 'adaptive':
            lambda_ = np.repeat(state.lambda_k[-1], 1, nPts)

        v = self.state.icasphere * blockdata # pre - whitened data
        lambda_avg = 1 - lambda_[np.ceil(len(lambda_) / 2)] # median lambda
        QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(np.transpose(v), v)) / nPts
        self.state.icasphere = 1 / lambda_avg * (state.icasphere - np.matmul(np.transpose(v), v) / nPts
                                            / QWhite * self.state.icasphere)


    def dynamicOrica(self, blockdata, dataRange, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros(nChs, nPts);
        # compute source activation using previous weight matrix
        y = self.state.icaweights * blockdata

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is not None:
            f[np.where(self.state.kurtsign==1), :]  = -2 * np.tanh(y[np.where(self.state.kurtsign==1), :])  # Supergaussian
            f[np.where(self.state.kurtsign==0), :] = 2 * np.tanh(y[np.where(self.state.kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (nonStatIdx) and variance of source dynamics (Var)
        if self.evalConvergence['profile']:
            modelFitness = np.eye(nChs) + np.matmul(y, np.transpose(f))/nPts
            variance = blockdata * blockdata
            if len(self.state.Rn) != 0:
                self.state.Rn = modelFitness
            else:
                self.state.Rn = (1 - self.evalConvergence['leakyAvgDelta']) * self.state.Rn + \
                                self.evalConvergence['leakyAvgDelta'] * modelFitness
                #!!! this does not account for block update!
            self.state.nonStatIdx = la.norm(self.state.Rn)


        if self.adaptiveFF['profile'] == 'cooling':
            self.state.lambda_k = genCoolingFF(self.state.counter + dataRange, adaptiveFF['gamma'], adaptiveFF['lambda_0'])
            if self.state.lambda_k[0] < adaptiveFF['lambda_const']:
                self.state.lambda_k = np.repeat(adaptiveFF['lambda_const'], 1, nPts)
            self.state.counter = self.state.counter + nPts;
        elif self.adaptiveFF['profile'] == 'constant':
            self.state.lambda_k = np.repeat(adaptiveFF['lambda_const'], 1, nPts)
        elif self.adaptiveFF['profile'] == 'adaptive':
            if len(self.state.minNonStatIdx) != 0:
                self.state.minNonStatIdx = self.state.nonStatIdx
            self.state.minNonStatIdx = np.max([np.min(self.state.minNonStatIdx, self.state.nonStatIdx), 1])
            ratioOfNormRn = self.state.nonStatIdx / self.state.minNonStatIdx
            self.state.lambda_k = genAdaptiveFF(dataRange, self.state.lambda_k, ratioOfNormRn)

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1 / (1-self.state.lambda_k))
        Q = 1 + self.state.lambda_k * (np.dot(f,y,1)-1);
        self.state.icaweights = lambda_prod * (self.state.icaweights - y * diag(self.state.lambda_k / Q)
                                          * np.transpose(f) * self.state.icaweights)

        # orthogonalize weight matrix
        D, V = la.eig(self.state.icaweights * np.transpose(self.state.icaweights))
        self.state.icaweights = V/sqrt(D)*np.transpose(V)*self.state.icaweights


    def genCoolingFF(self, t, gamma, lambda_0):
        lambda_ = lambda_0 / (t ** gamma)
        return lambda_

    def genAdaptiveFF(self, dataRange, lambda_, ratioOfNormRn):
        decayRateAlpha = adaptiveFF['decayRateAlpha']
        upperBoundBeta = adaptiveFF['upperBoundBeta']
        transBandWidthGamma = adaptiveFF['transBandWidthGamma']
        transBandCenter = adaptiveFF['transBandCenter']
        gainForErrors = upperBoundBeta * 0.5 * (1 + tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))
        def f(n):
            return  (1 + gainForErrors) ** n * lambda_[-1] - decayRateAlpha * ((1 + gainForErrors) ** (2 * n - 1) -
                                                                               (1 + gainForErrors) ** (n - 1)) / \
                    gainForErrors * lambda_[-1] **2
        lambda_ = f(range(len(dataRange)))
        return lambda_