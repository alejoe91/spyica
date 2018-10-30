import spikeinterface as si
import spiketoolkit as st
import spyica
import matplotlib.pylab as plt
import numpy as np

seed=2308
np.random.seed(seed)

recording_path = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_Neuronexus-32_10.0_10.0uV_13-10-2018:10:55'

recording = si.MEArecRecordingExtractor(recording_path)
fs = recording.getSamplingFrequency()
sorting_true = si.MEArecSortingExtractor(recording_path)

sorting_ica, sources_ica = spyica.ica_spike_sorting(recording)
sorting_orica, sources_ica = spyica.orica_spike_sorting(recording)
sorting_online, sources_online = spyica.online_orica_spike_sorting(recording)

true_ica = st.comparison.SortingComparison(sorting_ica, sorting_true)
true_orica = st.comparison.SortingComparison(sorting_orica, sorting_true)
true_online = st.comparison.SortingComparison(sorting_online, sorting_true)
ica_orica = st.comparison.SortingComparison(sorting_ica, sorting_orica)

true_online.plotConfusionMatrix()

plt.ion()
plt.show()