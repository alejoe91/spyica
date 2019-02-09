import spikeinterface as si
import spiketoolkit as st
import spyica
import matplotlib.pylab as plt
import numpy as np
from scipy.linalg import sqrtm


from spyica.tools import whiten_data

seed=2308
np.random.seed(seed)

recording_path = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_Neuronexus-32_60.0_10.0uV_29-10-2018:14:40'
# recording_path = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_SqMEA-10-15um_60.0_10.0uV_02-11-2018:13:29'
# recording_path = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_SqMEA-10-15um_30.0_10.0uV_02-11-2018:13:19'
# recording_path = '/home/alessio/Documents/Codes/MEArec/data/recordings/recordings_20cells_SqMEA-10-15um_10.0_10.0uV_02-11-2018:13:12'
recording = si.MEArecRecordingExtractor(recording_path)
fs = recording.getSamplingFrequency()
# sorting_true = si.MEArecSortingExtractor(recording_path)
#
# traces = recording.getTraces()
#
# orio = spyica.online_orica_alg(recording, online=True, n_comp='all',
#                               calibPCA=False, pca_window=10, ica_window=0, ica_block=1000, pca_block=3000)
#
# ori = spyica.orica_alg(recording, verbose=True)

# sorting_online, sources_online = spyica.online_orica_spike_sorting(recording, n_comp='all', white_mode='pca',
#                                                                    pca_window=0, ica_window=0,
#                                                                    online=True, calibPCA=False)
#
# true_online = st.comparison.SortingComparison(sorting_online, sorting_true)
# true_online.plotConfusionMatrix()
#
# print('ZCA whitening')
# zca = np.linalg.inv(sqrtm(np.cov(traces)))
# _, eigvecs, eigvals, pca = whiten_data(traces)

# true_ica = st.comparison.SortingComparison(sorting_ica, sorting_true)
# true_orica = st.comparison.SortingComparison(sorting_orica, sorting_true)
# true_online = st.comparison.SortingComparison(sorting_online, sorting_true)
# ica_orica = st.comparison.SortingComparison(sorting_ica, sorting_orica)
#
# true_online.plotConfusionMatrix()

plt.ion()
plt.show()