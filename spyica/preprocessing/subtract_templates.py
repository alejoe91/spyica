import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from tqdm import tqdm


class SubtractTemplates(BasePreprocessor):

    def __init__(self, recording, sorting, templates_dict, n_before, good_units=None):
        """

        Parameters
        ----------
        recording: RecordingExtractor
            RecordingExtractor object to be processed.
        sorting: SortingExtractor
            Output of spike sorting algorithm.
        templates_dict: dict
            Dictionary containing the templates of units to be removed from the recording.
            The keys are the ids of the units.
        n_before: int
            Number of samples before the spike time used to build the templates.
        good_units: list
            list of unit ids to be removed from the recording. All the ids must match the keys
            of templates_dict

        Returns
        ---------
        SubtractTemplates object. A new recording without spike of good_units
        """

        BasePreprocessor.__init__(self, recording)
        if good_units is None:
            unit_ids = sorting.get_unit_ids()
        else:
            unit_ids = good_units
            
        for unit in templates_dict.keys():
            assert int(unit) in unit_ids

        for i, segment in enumerate(recording._recording_segments):
            all_spikes, all_labels = sorting.get_all_spike_trains()[i]
            new_segment = SubtractRecordingSegment(segment, all_spikes, all_labels, templates_dict, n_before, unit_ids)
            self.add_recording_segment(new_segment)
        self._kwargs = dict(recording=recording.to_dict(), sorting=sorting.to_dict(), templates_dict=templates_dict,
                            n_before=n_before, good_units=good_units)


class SubtractRecordingSegment(BasePreprocessorSegment):

    def __init__(self, segment, all_spikes, all_labels, templates_dict, n_before, unit_ids):
        BasePreprocessorSegment.__init__(self, segment)
        self.all_spikes = all_spikes
        self.all_labels = all_labels
        self.templates = templates_dict
        self.n_before = n_before
        self.unit_ids = unit_ids

    def get_traces(self, start_frame, end_frame, channel_indices):
        t = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        traces = t.copy()  # needed by parallel in unitsrecovery
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = traces.shape[0]
        
        mask = np.where((start_frame <= self.all_spikes) & (self.all_spikes < end_frame))[0]
        spikes_window = self.all_spikes[mask]
        labels_window = self.all_labels[mask]
        
        for i in tqdm(range(len(spikes_window)), ascii=True, desc="subtracting spikes"):
            if labels_window[i] in self.unit_ids:
                st = spikes_window[i]
                temp = np.array(self.templates[str(int(labels_window[i]))])
                start = int(st - self.n_before - start_frame)
                end = start + temp.shape[0]
                if start < 0:
                    tmp = temp[-start:]
                    traces[:end] -= tmp
                elif end > traces.shape[0]:
                    end = traces.shape[0] - 1
                    traces[start:end] -= temp[:end - start]
                else:
                    traces[start:end] -= temp

        traces = traces[:, channel_indices]
        return traces


def subtract_templates(*args):
    return SubtractTemplates(*args)


subtract_templates.__doc__ = SubtractTemplates.__doc__
