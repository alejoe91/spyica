'''

'''

import numpy as np
import neo
import elephant.spike_train_generation as stg
import elephant.conversion as conv
import elephant.statistics as stat
import matplotlib.pylab as plt
import quantities as pq
from quantities import Quantity

class SpikeTrainGenerator:
    def __init__(self, n_exc=70, n_inh=30, f_exc=10*pq.Hz, f_inh=40*pq.Hz, st_exc=2*pq.Hz, st_inh=5*pq.Hz,
                 process='poisson', gamma_shape=2.0, t_start=0*pq.s, t_stop=10*pq.s, ref_period=2*pq.ms, n_add=0,
                 t_add=0, n_remove=0, t_remove=0):
        '''
        Spike Train Generator: class to create poisson or gamma spike trains

        Parameters
        ----------
        n_exc: number of excitatory cells
        n_inh: number of inhibitory cells
        f_exc: mean firing rate of excitatory cells
        f_inh: mean firing rate of inhibitory cells
        st_exc: firing rate standard deviation of excitatory cells
        st_inh: firing rate standard deviation of inhibitory cells
        process: 'poisson' - 'gamma'
        gamma_shape: shape param for gamma distribution
        t_start: starting time (s)
        t_stop: stopping time (s)
        ref_period: refractory period to remove spike violation
        n_add: number of units to add at t_add time
        t_add: time to add units
        n_remove: number of units to remove at t_remove time
        t_remove: time to remove units
        '''

        self.n_exc = n_exc
        self.n_inh = n_inh
        self.f_exc = f_exc
        self.f_inh = f_inh
        self.st_exc = st_exc
        self.st_inh = st_inh
        self.process=process
        self.t_start = t_start
        self.t_stop = t_stop
        self.exc_st = None
        self.inh_st = None
        self.min_rate = 0.5 * pq.Hz
        self.ref_period = ref_period
        self.all_spiketrains = []
        self.n_add = n_add
        self.n_remove = n_remove
        self.t_add = int(t_add) * pq.s
        self.t_remove = int(t_remove) * pq.s
        self.intermittent = False
        self.idx = 0

        if n_add != 0:
            if t_add == 0:
                raise Exception('Provide time to add units')
            else:
                self.intermittent = True
        if n_remove != 0:
            if t_remove == 0:
                raise Exception('Provide time to remove units')
            else:
                self.intermittent = True


        if self.intermittent:
            n_tot = n_exc + n_inh
            perm_idxs = np.random.permutation(np.arange(n_tot))
            self.idxs_add = perm_idxs[:self.n_add]
            self.idxs_remove = perm_idxs[-self.n_remove:]
        else:
            self.idxs_add = []
            self.idxs_remove = []

    def set_spiketrain(self, idx, spiketrain):
        '''
        Sets spike train idx to new spiketrain
        Parameters
        ----------
        idx: index of spike train to set
        spiketrain: new spike train

        Returns
        -------

        '''
        self.all_spiketrains[idx] = spiketrain

    def generate_spikes(self):
        '''
        Generate spike trains based on params of the SpikeTrainGenerator class.
        self.all_spiketrains contains the newly generated spike trains

        Returns
        -------

        '''

        self.all_spiketrains = []
        for exc in range(self.n_exc):
            if self.process == 'poisson':
                rate = self.st_exc * np.random.randn() + self.f_exc
                if rate < self.min_rate:
                    rate = self.min_rate
                if self.intermittent:
                    if self.idx in self.idxs_add:
                        st = stg.homogeneous_poisson_process(rate, self.t_add, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_add, self.t_stop)
                        st.t_start = self.t_start
                    elif self.idx in self.idxs_remove:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_remove)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_remove)
                        st.t_stop = self.t_stop
                    else:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                else:
                    st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                    while len(st) == 0:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            elif self.process == 'gamma':
                rate = self.st_exc * np.random.randn() + self.f_exc
                if rate < self.min_rate:
                    rate = self.min_rate
                if self.intermittent:
                    if self.idx in self.idxs_add:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_add, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_add, self.t_stop)
                        st.t_start = self.t_start
                    elif self.idx in self.idxs_remove:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_remove)
                        while len(st) == 0:
                            st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_remove)
                        st.t_stop = self.t_stop
                    else:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                else:
                    st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_stop)
                    while len(st) == 0:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)
            self.idx += 1

        for inh in range(self.n_inh):
            if self.process == 'poisson':
                rate = self.st_inh * np.random.randn() + self.f_inh
                if rate < self.min_rate:
                    rate = self.min_rate
                if self.intermittent:
                    if self.idx in self.idxs_add:
                        st = stg.homogeneous_poisson_process(rate, self.t_add, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_add, self.t_stop)
                        st.t_start = self.t_start
                    elif self.idx in self.idxs_remove:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_remove)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_remove)
                        st.t_stop = self.t_stop
                    else:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                else:
                    st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                    while len(st) == 0:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            elif self.process == 'gamma':
                rate = self.st_inh * np.random.randn() + self.f_inh
                if rate < self.min_rate:
                    rate = self.min_rate
                if self.intermittent:
                    if self.idxs in self.idxs_add:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_add, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_add, self.t_stop)
                        st.t_start = self.t_start
                    elif self.idxs in self.idxs_remove:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_remove)
                        while len(st) == 0:
                            st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_remove)
                        st.t_stop = self.t_stop
                    else:
                        st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                        while len(st) == 0:
                            st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                else:
                    st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_stop)
                    while len(st) == 0:
                        st = stg.homogeneous_gamma_process(gamma_shape, rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)
            self.idx += 1

        # check consistency and remove spikes below refractory period
        for idx, st in enumerate(self.all_spiketrains):
            isi = stat.isi(st)
            idx_remove = np.where(isi < self.ref_period)[0]
            spikes_to_remove = len(idx_remove)
            unit = st.times.units

            while spikes_to_remove > 0:
                new_times = np.delete(st.times, idx_remove[0]) * unit
                st = neo.SpikeTrain(new_times, t_start=self.t_start, t_stop=self.t_stop)
                isi = stat.isi(st)
                idx_remove = np.where(isi < self.ref_period)[0]
                spikes_to_remove = len(idx_remove)

            self.set_spiketrain(idx, st)


    def raster_plots(self, marker='|', markersize=5, mew=2):
        '''
        Plots raster plots of spike trains

        Parameters
        ----------
        marker: marker type (def='|')
        markersize: marker size (def=5)
        mew: marker edge width (def=2)

        Returns
        -------
        ax: matplotlib axes

        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, spiketrain in enumerate(self.all_spiketrains):
            t = spiketrain.rescale(pq.s)
            if i < self.n_exc:
                ax.plot(t, i * np.ones_like(t), color='b', marker=marker, ls='', markersize=markersize, mew=mew)
            else:
                ax.plot(t, i * np.ones_like(t), color='r', marker=marker, ls='', markersize=markersize, mew=mew)
        ax.axis('tight')
        ax.set_xlim([self.t_start.rescale(pq.s), self.t_stop.rescale(pq.s)])
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)

        return ax

    def resample_spiketrains(self, fs=None, T=None):
        '''
        Resamples spike trains. Provide either fs or T parameters
        Parameters
        ----------
        fs: new sampling frequency (quantity)
        T: new period (quantity)

        Returns
        -------
        matrix with resampled binned spike trains

        '''
        resampled_mat = []
        if not fs and not T:
            print 'Provide either sampling frequency fs or time period T'
        elif fs:
            if not isinstance(fs, Quantity):
                raise ValueError("fs must be of type pq.Quantity")
            binsize = 1./fs
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        elif T:
            binsize = T
            if not isinstance(T, Quantity):
                raise ValueError("T must be of type pq.Quantity")
            binsize.rescale('ms')
            resampled_mat = []
            for sts in self.all_spiketrains:
                spikes = conv.BinnedSpikeTrain(sts, binsize=binsize).to_array()
                resampled_mat.append(np.squeeze(spikes))
        return np.array(resampled_mat)

    def add_synchrony(self, idxs, rate=0.05):
        '''
        Adds synchronous spikes between pairs of spike trains at a certain rate
        Parameters
        ----------
        idxs: list or array with the 2 indices
        rate: probability of adding a synchronous spike to spike train idxs[1] for each spike of idxs[0]

        Returns
        -------

        '''
        idx1 = idxs[0]
        idx2 = idxs[1]
        st1 = self.all_spiketrains[idx1]
        st2 = self.all_spiketrains[idx2]
        times2 = st2.times
        t_start = st2.t_start
        t_stop = st2.t_stop
        unit = times2.units
        for t1 in st1:
            rand = np.random.rand()
            if rand <= rate:
                # check time difference
                t_diff = np.abs(t1.rescale(pq.ms).magnitude-times2.rescale(pq.ms).magnitude)
                if np.all(t_diff > self.ref_period):
                    times2 = np.sort(np.concatenate((np.array(times2), np.array([t1]))))
                    times2 = times2 * unit
                    st2 = neo.SpikeTrain(times2, t_start=t_start, t_stop=t_stop)
                    self.set_spiketrain(idx2, st2)


    def bursting_st(self, freq=None, min_burst=3, max_burst=10):
        pass
