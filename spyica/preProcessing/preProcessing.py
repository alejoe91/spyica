import numpy as np
from spikeinterface.toolkit.preprocessing.basepreprocessor import BasePreprocessor, BasePreprocessorSegment
from ..tools import clean_sources

clean_idx = None


class LinearMapFilter(BasePreprocessor):
    name = 'Filter'

    def __init__(self, recording, matrix):
        self.recording = recording
        if isinstance(matrix, np.ndarray):
            self.M = [matrix]
        else:
            self.M = matrix
        if not recording.get_num_channels() == self.M[0].shape[0]:
            raise ArithmeticError(
                f"Matrix first dimension must be equal to number of channels: {recording.get_num_channels()}"
                f"It is: {self.M.shape[0]}")

        BasePreprocessor.__init__(self, recording)
        for i, parent_segment in enumerate(recording._recording_segments):
            segment = FilterRecordingSegment(parent_segment, self.M[i])
            self.add_recording_segment(segment)

        self._kwargs = dict(recording=recording.to_dict())


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, M):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.M = M

    def get_traces(self, start_frame, end_frame, channel_indices):
        global clean_idx
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None))
        filtered_traces = traces @ self.M.T
        if channel_indices is not None:
            filtered_traces = filtered_traces[:, channel_indices]
        return filtered_traces


def lin_filter(*args):
    return LinearMapFilter(*args)


lin_filter.__doc__ = LinearMapFilter.__doc__
