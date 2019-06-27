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
from spyica.tools import whiten_data


class ORICA():
    def __init__(self, data, ndim='all', block_size=800, white_mode='pca', forgetfac='cooling', lambda_0=0.995,
                 pcaonly=False, numpass=1, weights=[], nsub=0, ffdecayrate=0.6, computeNSI=True, verbose=False):
        '''Runs Online Recursive ICA on the data with pre-whitening.

        Parameters
        ----------
        data: numpy array
            Recording data Nchs x Nframes
        ndim: int or 'all'
            Number of components to keep after whitening (default='all')
        block_size: int
            Number of samples in block processing (defaulr=800)
        white_mode: str
            'pca' or 'zca' (default='pca')
        forgetfac: str
            'constant' | 'cooling' | 'adaptive' (default='cooling')
        lambda_0: float
            Initial forgetting factor value (default=0.995)
        pcaonly: bool
            If True only pca is run (not orica)
        numpass: int
            Specifies number of passes through the data (default=1)
        weights: numpy array
            Initial ICA weights (default=[])
        nsub: int
            Number of subgaussian sources (default=0)
        ffdecayrate: float
            Decay rate for 'cooling' forgetting factor (default=0.6)
        computeNSI: bool
            If True the Non-Stationarity-Index is compute (default=True)
        verbose: bool
            If True print statements are shown (default=True)


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
        if ndim == 'all':
            ndim = nChs

        # Parameters for adaptive weights
        adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0,
                      'decayRateAlpha': 0.02, 'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1,
                      'transBandCenter': 5, 'lambdaInitial': 0.1}

        ##############################
        # initialize state variables #
        ##############################
        if len(weights) == 0:
            icaweights = np.eye(ndim)
        self.counter = 1
        self.lambdas = np.array([])

        if adaptiveFF['profile'] == 'cooling' or  adaptiveFF['profile'] == 'constant':
            adaptiveFF['lambda_const']  = 1-np.exp(-1 / (adaptiveFF['tau_const']))

        if computeNSI:
            self.NSI = []
            NSIparams = {'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        # sign of kurtosis for each component: 1 (supergaussian), 0 (subgaussian)
        kurtsign = np.ones((ndim, 1))
        if nsub != 0:
            kurtsign[:nsub] = 0


        ######################
        # sphere-whiten data #
        ######################
        if verbose:
            print('Use pre-whitening method.')
        if ndim == nChs:
            if white_mode == 'pca':
                if verbose:
                    print('PCA whitening')
                data_w, eigvecs, eigvals, sphere = whiten_data(data)
                icasphere = sphere
            elif white_mode == 'zca':
                if verbose:
                    print('ZCA whitening')
                # TODO use SVD and compute ZCA VS PCA
                icasphere = la.inv(sqrtm(np.cov(data)))
                data_w = np.matmul(icasphere, data)
        else:
            if verbose:
                print('Reducing dimension to ', ndim)
            data_w, eigvecs, eigvals, sphere = whiten_data(data, ndim)
            icasphere = sphere

        self.icaweights = icaweights
        self.icasphere = icasphere
        # self.icasphere_1 = la.pinv(self.icasphere)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        numBlock = int(np.floor(nPts / block_size))

        if verbose:
            printflag = 0
            if adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')
        t_start = time.time()

        for it in range(numpass):
            for bi in range(numBlock):
                dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
                                  int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))

                if not pcaonly:
                    self._dynamicOrica(data_w[:, dataRange], dataRange, kurtsign, adaptiveFF, computeNSI, NSIparams)

                if verbose:
                    if printflag < np.floor(10 * (it * numBlock + bi) / numpass / numBlock):
                        printflag = printflag + 1
                        print(10 * printflag, '%')

        if verbose:
            processing_time = time.time() - t_start
            print('ORICA Finished. Elapsed time: ', processing_time, ' sec.')

        # output weights and sphere matrices
        if not pcaonly:
            self.unmixing = np.matmul(self.icaweights, self.icasphere)
            self.mixing = la.pinv(self.unmixing).T
            self.y = np.matmul(self.unmixing, data)


    def _dynamicOrica(self, blockdata, dataRange, kurtsign, adaptiveFF, computeNSI, NSIparams, nlfunc=None):

        # initialize
        nChs, nPts = blockdata.shape
        f = np.zeros((nChs, nPts))
        # compute source activation using previous weight matrix
        y = np.matmul(self.icaweights, blockdata)

        # choose nonlinear functions for super- vs. sub-gaussian
        if nlfunc is None:
            f[np.where(kurtsign==1), :]  = -2 * np.tanh(y[np.where(kurtsign==1), :])  # Supergaussian
            f[np.where(kurtsign==0), :] = 2 * np.tanh(y[np.where(kurtsign==0), :])  # Subgaussian
        else:
            f = nlfunc(y);

        # compute Non-Stationarity Index (NSI)
        if computeNSI:
            modelFitness = np.eye(nChs) + np.matmul(y, f.T) / nPts
            if len(self.NSI) == 0:
                self.Rn = modelFitness
            else:
                self.Rn = (1 - NSIparams['leakyAvgDelta']) * self.Rn + NSIparams['leakyAvgDelta'] * modelFitness
            # print(self.Rn, la.norm(self.Rn), modelFitness.shape)
            self.NSI.append(la.norm(self.Rn))

        # compute block forgetting factors
        if adaptiveFF['profile'] == 'cooling':
            lambda_k = self._genCoolingFF(self.counter + dataRange, adaptiveFF['gamma'],
                                                    adaptiveFF['lambda_0'])
            if lambda_k[0] < adaptiveFF['lambda_const']:
                lambda_k = np.squeeze(np.tile(adaptiveFF['lambda_const'], (1, nPts)))
            self.counter = self.counter + nPts
        elif adaptiveFF['profile'] == 'constant':
            lambda_k = np.arange(nPts) * adaptiveFF['lambda_0']
        # elif adaptiveFF['profile'] == 'adaptive':
        #     if self.minNonStatIdx != -1:
        #         self.minNonStatIdx = self.nonStatIdx
        #     self.minNonStatIdx = np.max([np.min([self.minNonStatIdx, self.nonStatIdx]), 1])
        #     ratioOfNormRn = self.nonStatIdx / self.minNonStatIdx
        #     lambda_k = self.genAdaptiveFF(dataRange, lambda_k, ratioOfNormRn, adaptiveFF)

        # concatenate current weights
        self.lambdas = np.concatenate((self.lambdas, lambda_k))

        # update weight matrix using online recursive ICA block update rule
        lambda_prod = np.prod(1. / (1.-lambda_k))
        Q = 1 + lambda_k * (np.sum(f * y, axis=0) - 1);
        self.icaweights = lambda_prod * (self.icaweights - np.matmul(np.matmul(np.matmul(y, np.diag(lambda_k / Q)),
                                                                               f.T), self.icaweights))
        # orthogonalize weight matrix
        try:
            D, V = eigh(np.matmul(self.icaweights, self.icaweights.T))
        except LinAlgError:
            raise Exception()
        self.icaweights = np.matmul(np.matmul(la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))).T, V.T).T,
                                                    V.T), self.icaweights)

    def _genCoolingFF(self, t, gamma, lambda_0):
        lambda_ = lambda_0 / (t ** gamma)
        return lambda_

    def _genAdaptiveFF(self, dataRange, lambda_in, ratioOfNormRn, adaptiveFF):
        decayRateAlpha = adaptiveFF['decayRateAlpha']
        upperBoundBeta = adaptiveFF['upperBoundBeta']
        transBandWidthGamma = adaptiveFF['transBandWidthGamma']
        transBandCenter = adaptiveFF['transBandCenter']

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
    def __init__(self, data, fs, ndim='all', onlineWhitening=True, calibratePCA=False, forgetfac='cooling',
                 skew_thresh=0.5, lambda_0=0.995, min_lambda=0, pca_block=2000, ica_block=800,
                 nsub=0, ffdecayrate=0.6, save_step_state = False,
                 pcaweights=[], icaweights=[], step_size=1, skew_window=20, white_mode='pca',
                 pca_window=0, ica_window=0, detect_trheshold=8, onlineDetection=True, computeNSI=True, numpass=1,
                 computeWI=True, verbose=True):
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
        if ndim == 'all':
            ndim = nChs

        if isinstance(fs, pq.Quantity):
            fs = int(fs.rescale('Hz').magnitude)
        else:
            fs = int(fs)

        n_skew_window = int(fs*skew_window)
        n_pca_window = int(fs*pca_window)
        n_ica_window = int(fs*ica_window)
        n_step_size = int(fs*step_size)

        # Parameters for adaptive weights
        adaptiveFF = {'profile': forgetfac, 'tau_const': np.inf, 'gamma': 0.6, 'lambda_0': lambda_0,
                      'decayRateAlpha': 0.02, 'upperBoundBeta': 1e-3, 'transBandWidthGamma': 1,
                      'transBandCenter': 5, 'lambdaInitial': 0.1}

        ##############################
        # initialize state variables #
        ##############################
        if len(icaweights) == 0:
            icaweights = np.eye(ndim)

        self.counter = 1
        self.lambdas = np.array([])

        if adaptiveFF['profile'] == 'cooling' or adaptiveFF['profile'] == 'constant':
            adaptiveFF['lambda_const'] = 1 - np.exp(-1 / (adaptiveFF['tau_const']))

        if computeNSI:
            self.NSI = []
            NSIparams = {'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}
        
        if computeWI:
            self.WI = []
            WIparams = {'leakyAvgDelta': 0.01, 'leakyAvgDeltaVar': 1e-3}

        # sign of kurtosis for each component: 1 (supergaussian), 0 (subgaussian)
        kurtsign = np.ones((ndim, 1))
        if nsub != 0:
            kurtsign[:nsub] = 0

        if adaptiveFF['profile'] == 'cooling' or  adaptiveFF['profile'] == 'constant':
            adaptiveFF['lambda_const']  = 1-np.exp(-1 / (adaptiveFF['tau_const']))


        ######################
        # sphere-whiten data #
        ######################

        # initialize
        self.cov = np.zeros((nChs, nChs))
        self.mu = np.zeros((nChs, 1))
        self.pca_counter = 1

        self.pca_mode = 'stationary'
        # self.pca_lambda_const = 1.5e-4
        self.pca_gamma = 0.92;
        self.pca_lambda_0 = 0.6;

        if not calibratePCA:
            if not onlineWhitening:  # pre - whitening
                if verbose:
                    print('Use pre-whitening method.')
                _, eigvecs, eigvals, sphere = whiten_data(data, ndim)
                icasphere = sphere
                n_pca_window = 0
            else:  # Online RLS Whitening
                if verbose:
                    print('Use online whitening method.')
        else:
            if verbose:
                print('Use initial PCA calibration method.')
            self.pca_calibrated = False
            icasphere = np.zeros(ndim)

        self.icaweights = icaweights
        self.icasphere = icasphere
        self.icasphere_1 = la.pinv(self.icasphere)
        self.skew_thresh = skew_thresh
        self.y_on = np.zeros((ndim, nPts), dtype=float)

        #########
        # ORICA #
        #########

        # divide data into blocks for online block update
        # numBlock = int(np.floor(nPts / block))
        if save_step_state:
            self.skews = []
            self.w = []
            self.m = []
        self.idx_sources = []
        iter = 0

        if verbose:
            printflag = 0
            if adaptiveFF['profile'] == 'cooling':
                print('Running ORICA with cooling forgetting factor...')
            elif adaptiveFF['profile'] == 'constant':
                print('Running ORICA with constant forgetting factor...')
            elif adaptiveFF['profile'] == 'adaptive':
                print('Running ORICA with adaptive forgetting factor...')

        t_start = time.time()
        self.init = False
        self.spikes = {}
        self.all_sources = np.array([])
        self.nskews = []
        # self.lambdas = np.array([])
        # self.ratios = np.array([])
        # self.normNSI = np.array([])
        # self.NSImean = 0
        # self.NSIvar = 0

        first_pca=False
        first_ica=False
        # for bi in range(numBlock):
        #     dataRange = np.arange(int(np.floor(bi * nPts / numBlock)),
        #                       int(np.min([nPts, np.floor((bi + 1) * nPts / numBlock)])))
        for i in range(nPts):
            self.N = i
            # self.N += len(dataRange)
            # self.onlineMean(data[:, dataRange])

            if calibratePCA:
                if np.mod(self.N, n_pca_window) == 0 and not self.pca_calibrated:
                    self.calibratePCA(data[:, self.N], ndim)

            #TODO FIX new implementation

            if np.mod(self.N, pca_block) == 0 and self.N != 0:
                dataRange = np.arange(self.N - pca_block, self.N)
                if onlineWhitening:
                    # self.dynamicWhitening(data[:, dataRange], dataRange)
                    if white_mode == 'pca':
                        self.dynamicPCA(data[:, dataRange], dataRange)
                    elif white_mode == 'zca':
                        self.dynamicZCA(data[:, dataRange], dataRange)
                else:
                    # compute WI
                    data_cent = data[:, dataRange] - self.mu
                    self.dynamicWhitening(data_cent, dataRange, whitening=False)
                if not first_pca:
                    print(self.N, ' First PCA done')
                    first_pca = True
                print('PCA_block: ', self.N)

            if self.N > n_pca_window:
                if np.mod(self.N, ica_block) == 0 and first_pca:
                    dataRange = np.arange(self.N - ica_block, self.N)
                    data_cent = data[:, dataRange] - self.mu
                    data_white = np.matmul(self.icasphere[:ndim], data_cent)
                    self.dynamicOrica(data_white, dataRange)
                    if not first_ica:
                        print(self.N, ' First ICA done')
                        first_ica = True
                    print('ICA_block: ', self.N)

                    if self.N > n_pca_window + n_ica_window:
                        # online sources
                        if first_pca and first_ica:
                            self.y_on[:, dataRange] = np.matmul(self.icaweights, data_white)

            # select sources
            if not np.mod(self.N, n_step_size):
                if self.N > n_pca_window + n_ica_window:
                    self.computeSkew()
                    idx_sources = np.where(np.abs(self.skew) > self.skew_thresh)
                    self.w.append(self.icaweights)
                    self.m.append(self.icasphere[:ndim])
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
                    self.m.append(self.icasphere[:ndim])
                    self.idx_sources.append([])

            if verbose:
                if printflag < np.floor(10 * self.N / float(nPts)):
                    printflag = printflag + 1
                    print(10 * printflag, '%')
                    if self.N > n_pca_window + n_ica_window and len(self.idx_sources) != 0:
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


    def computeSkew(self, n_skew_window):
        if self.N < n_skew_window:
            y = self.y_on[:, :self.N]
            # skewness for source selection
            self.skew = stats.skew(y, axis=1)
            # self.sigma = np.std(y, axis=1)
        else:
            # skewness for source selection
            y = self.y_on[:, self.N - n_skew_window:self.N]
            self.skew = stats.skew(y, axis=1)
            # self.sigma = np.std(self.y, axis=1)
        self.skews.append(self.skew)


    def detectSpikes(self, idx_sources, n_step_size):

        self.y[self.skew > 0] = -self.y[self.skew > 0]
        if self.init:
            # first time detect spikes on all past signals
            sources = self.y[self.idx_sources[-1]]
        else:
            # then append spikes from single blocks
            sources = self.y[self.idx_sources[-1], -n_step_size:]

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
                            sp_times.append(self.N - n_step_size + idx)
                    elif idx - idx_spikes[t - 1] > 1: # or t == len(idx_spike) - 2:  # single spike
                        # append crossing time
                        if self.init:
                            if self.N < self.n_window:
                                sp_times.append(idx)
                            else:
                                sp_times.append(idx)
                        else:
                            sp_times.append(self.N - n_step_size + idx)
                self.spikes.update({s_idx: np.concatenate((times, np.array(sp_times)))})

        if self.init:
            self.init = False
    #
    #
    # def onlineMean(self, blockdata):
    #     self.means = 1. / self.N * (self.sumx[:, np.newaxis] + np.sum(blockdata, axis=1, keepdims=True))
    #     self.sumx += np.sum(blockdata, axis=1)

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
        # if normNSI > adaptiveFF['transBandCenter']:
        #     self.tracking = True
        # else:
        #     if self.tracking:
        #         self.tracking = False
        #         self.tracking_iter = 0
        #         print('STOP TRACKING')

        return normNSI


    def calibratePCA(self, data, ndim):
        print('PCA calibration')
        _, eigvecs, eigvals, sphere = whiten_data(data, ndim)
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
            if adaptiveFF['profile'] == 'cooling':
                lambda_ = self.genCoolingFF(self.counter + dataRange, adaptiveFF['gamma'],
                                            adaptiveFF['lambda_0'],
                                            adaptiveFF['min_lambda'])
                if lambda_[0] < adaptiveFF['lambda_const']:
                    lambda_ = np.squeeze(np.tile(adaptiveFF['lambda_const'], (1, nPts)))
            elif adaptiveFF['profile'] == 'constant':
                lambda_ = np.ones(nPts) * adaptiveFF['lambda_0']

            lambda_avg = 1 - lambda_[int(np.ceil(len(lambda_) / 2))] # median lambda
            QWhite = lambda_avg / (1-lambda_avg) + np.trace(np.matmul(v, v.T)) / nPts
            self.icasphere = 1 / lambda_avg * (self.icasphere - np.matmul(np.matmul(v, v.T) / nPts
                                                / QWhite,self.icasphere))


    def dynamicPCA(self, blockdata, dataRange, whitening=True):
        nChs, nPts = blockdata.shape
        self.pca_counter += 1

        if self.pca_mode == 'stationary':
            ff = self.genStationaryFF()
        elif self.pca_mode == 'cooling':
            pass
        elif self.pca_mode == 'constant':
            pass

        self.mu = (1 - ff) * self.mu + ff * np.mean(blockdata, axis=1, keepdims=True)
        self.cov = (1 - ff) * self.cov + ff * (1-ff) * np.matmul((blockdata - self.mu), (blockdata - self.mu).T) / float(nPts)

        # D, V = eigh(self.cov)
        D, V = la.eig(self.cov)

        sort_idxs = np.argsort(D)[::-1]
        V = V[:, sort_idxs]
        D = D[sort_idxs]

        self.icasphere =  la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))), V.T)

    def dynamicZCA(self, blockdata, dataRange, whitening=True):
        nChs, nPts = blockdata.shape
        self.pca_counter += 1

        if self.pca_mode == 'stationary':
            ff = self.genStationaryFF()
        elif self.pca_mode == 'cooling':
            pass
        elif self.pca_mode == 'constant':
            pass

        self.mu = (1 - ff) * self.mu + ff * np.mean(blockdata, axis=0)
        self.cov = (1 - ff) * self.cov + ff * (1-ff) * np.matmul((blockdata - self.mu), (blockdata - self.mu).T) / float(nPts)

        D, V = la.eig(self.cov)
        sort_idxs = np.argsort(D)[::-1]
        V = V[:, sort_idxs]
        D = D[sort_idxs]

        self.icasphere =  np.matmul(V, la.solve(np.diag((np.sqrt(np.abs(D)) * np.sign(D))), V.T))

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
            # if self.N > self.n_pca_window + self.n_ica_window:
            #     self.iter += 1
            #     nsinorm = self.onlineNSIUpdate(self.nonStatIdx, ff=adaptiveFF['ff'])
            #     self.normNSI = np.append(self.normNSI, nsinorm)
            # else:
            #     self.normNSI= np.append(self.normNSI, 0)

        if adaptiveFF['profile'] == 'cooling':
            self.lambda_k = self.genCoolingFF(self.counter + dataRange, adaptiveFF['gamma'],
                                              adaptiveFF['lambda_0'], adaptiveFF['min_lambda'])
            if self.lambda_k[0] < adaptiveFF['lambda_const']:
                self.lambda_k = np.squeeze(np.tile(adaptiveFF['lambda_const'], (1, nPts)))
            self.counter = self.counter + nPts;
        elif adaptiveFF['profile'] == 'constant':
            self.lambda_k = np.ones(nPts) * adaptiveFF['lambda_0']
        elif adaptiveFF['profile'] == 'adaptive':
            if self.minNonStatIdx == -1:
                self.minNonStatIdx = self.nonStatIdx
            self.minNonStatIdx = np.max([np.min([self.minNonStatIdx, self.nonStatIdx]), 1])
            ratioOfNormRn = self.nonStatIdx / self.minNonStatIdx
            # self.lambda_k = self.genAdaptiveFF(dataRange, self.lambda_k, ratioOfNormRn)
            self.lambda_k = self.genAdaptiveFF(dataRange, self.lambda_k, self.normNSI[-1], adaptiveFF)
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
        decayRateAlpha = adaptiveFF['decayRateAlpha']
        upperBoundBeta = adaptiveFF['upperBoundBeta']
        transBandWidthGamma = adaptiveFF['transBandWidthGamma']
        transBandCenter = adaptiveFF['transBandCenter']

        gainForErrors = upperBoundBeta * 0.5 * (1 + np.tanh((ratioOfNormRn - transBandCenter) / transBandWidthGamma))
        nrange = np.arange(len(dataRange))

        lambda_ = np.zeros(len(dataRange))
        lam_pr = lambda_in[-1]
        for i in nrange:
            lam_new = (1+gainForErrors) * lam_pr - decayRateAlpha * lam_pr ** 2
            # lam_new = (1./((i+1)**adaptiveFF['gamma']) + gainForErrors) * lam_pr
            lambda_[i] = lam_new
            lam_pr = lam_new

        return lambda_

    def genStationaryFF(self):
        return 1. / self.pca_counter


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
        orica = ORICA(X_reduced, ndim=n_comp, verbose=True, numpass=numpass)
    else:
        raise Exception('Unrecognized orica type')

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
