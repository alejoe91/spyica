import numpy as np
from spikeinterface import NumpySorting
from .preProcessing import LinearMapFilter
from ..SpyICASorter.SpyICASorter import SpyICASorter


class ICAFilter(LinearMapFilter):

    def __init__(self, recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
                 max_num_spikes=None):
        if isinstance(recording, NumpySorting):
            raise Exception("import a NumpySorting recording")
        sorter = SpyICASorter(recording)
        sorter.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                           balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes)
        sorter.compute_ica('all')
        self.A = sorter.A_ica

        LinearMapFilter.__init__(self, recording, sorter.W_ica)

    def find_closest(self, recording):
        # find closest channels
        max_ids = np.argmax(self.A, axis=1)
        chan_loc = recording.get_channel_locations()
        closest = []
        for idx in max_ids:
            pos = chan_loc[idx]
            chans = []
            for c in range(32):
                dist = np.sqrt(np.square(pos[0] - chan_loc[c, 0]) + np.square(pos[1] - chan_loc[c, 1]))
                if dist <= 50:
                    chans.append(c)
            closest.append(chans)
        return closest, max_ids

    def compute_smoothness(self, closest, max_ids):
        av_val = []
        for i, ind in enumerate(max_ids):
            max_val = self.A[i, ind]
            close_val = self.A[i, closest[i]]
            av = max_val - (np.sum(close_val) - max_val) / (len(close_val) - 1)
            if av >= 1800:
                av_val.append(av)
                print(av, i)


def ica_filter(recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
               max_num_spikes=None):
    filt = ICAFilter(recording, sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                     balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes)
    return filt


ica_filter.__doc__ = ICAFilter.__doc__
