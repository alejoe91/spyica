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
                 process='poisson', t_start=0*pq.s, t_stop=10*pq.s, ref_period=2*pq.ms):
        '''

        Parameters
        ----------
        n_exc
        n_inh
        f_exc
        f_inh
        process
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

    # @property
    # def all_spiketrains(self):
    #     all_spikes = []
    #     if self.exc_st:
    #         all_spikes.extend(self.exc_st)
    #     if self.inh_st:
    #         all_spikes.extend(self.inh_st)
    #     return all_spikes

    def set_spiketrain(self, idx, spiketrain):
        self.all_spiketrains[idx] = spiketrain
        # if idx < self.n_exc:
        #     self.exc_st[idx] = spiketrain
        # else:
        #     self.inh_st[idx - self.n_exc] = spiketrain

    def generate_spikes(self):

        self.all_spiketrains = []
        for exc in range(self.n_exc):
            if self.process == 'poisson':
                rate = self.st_exc * np.random.randn() + self.f_exc
                if rate < self.min_rate:
                    rate = self.min_rate
                st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                while len(st) == 0:
                    st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            elif self.process == 'gamma':
                rate = self.st_exc * np.random.randn() + self.f_exc
                if rate < self.min_rate:
                    rate = self.min_rate
                st = stg.homogeneous_gamma_process(2.0, rate, self.t_start, self.t_stop)
                while len(st) == 0:
                    st = stg.homogeneous_gamma_process(2.0, rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)

        for inh in range(self.n_inh):
            if self.process == 'poisson':
                rate = self.st_inh * np.random.randn() + self.f_inh
                if rate < self.min_rate:
                    rate = self.min_rate
                st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                while len(st) == 0:
                    st = stg.homogeneous_poisson_process(rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            elif self.process == 'gamma':
                rate = self.st_inh * np.random.randn() + self.f_inh
                if rate < self.min_rate:
                    rate = self.min_rate
                st = stg.homogeneous_gamma_process(2.0, rate, self.t_start, self.t_stop)
                while len(st) == 0:
                    st = stg.homogeneous_gamma_process(2.0, rate, self.t_start, self.t_stop)
                self.all_spiketrains.append(st)
            self.all_spiketrains[-1].annotate(freq=rate)

        # check consistency and remove spikes below refractory period
        for idx, st in enumerate(self.all_spiketrains):
            isi = stat.isi(st)
            idx_remove = np.where(isi < self.ref_period)[0]
            spikes_to_remove = len(idx_remove)
            unit = st.times.units

            while spikes_to_remove > 0:
                new_times = np.delete(st.times, idx_remove[0]) * unit
                st = neo.SpikeTrain(new_times, tstart=self.t_start, t_stop=self.t_stop)
                isi = stat.isi(st)
                idx_remove = np.where(isi < self.ref_period)[0]
                spikes_to_remove = len(idx_remove)

            self.set_spiketrain(idx, st)
        #
        # if self.exc_st:
        #     self.all_spiketrains.extend(self.exc_st)
        # if self.inh_st:
        #     self.all_spiketrains.extend(self.inh_st)


    def raster_plots(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i, spiketrain in enumerate(self.all_spiketrains):
            t = spiketrain.rescale(pq.s)
            if i < self.n_exc:
                ax.plot(t, i * np.ones_like(t), 'b.', markersize=5)
            else:
                ax.plot(t, i * np.ones_like(t), 'r.', markersize=5)
        ax.axis('tight')
        ax.set_xlim([self.t_start.rescale(pq.s), self.t_stop.rescale(pq.s)])
        ax.set_xlabel('Time (ms)', fontsize=16)
        ax.set_ylabel('Spike Train Index', fontsize=16)
        plt.gca().tick_params(axis='both', which='major', labelsize=14)

        return ax

    def resample_spiketrains(self, fs=None, T=None):
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
