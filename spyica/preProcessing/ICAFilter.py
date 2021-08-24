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
        source_idx = []
        chan_loc = recording.get_channel_locations()
        num_channels = recording.get_num_channels()
        for chan in range(num_channels):
            if np.abs(np.sum(sorter.A_ica[chan])) > max(np.max(sorter.A_ica[chan]), np.abs(np.min(sorter.A_ica[chan]))):
                source_idx.append(chan)

        LinearMapFilter.__init__(self, recording, sorter.W_ica)


def ica_filter(recording, sample_window_ms=1, percent_spikes=None, balance_spikes_on_channel=False,
               max_num_spikes=None):
    filt = ICAFilter(recording, sample_window_ms=sample_window_ms, percent_spikes=percent_spikes,
                     balance_spikes_on_channel=balance_spikes_on_channel, max_num_spikes=max_num_spikes)
    return filt


ica_filter.__doc__ = ICAFilter.__doc__
