import numpy as np
import pylab as plt
import MEAutility as MEA
from neuroplot import *
from os.path import join
import quantities as pq
from tools import filter_analog_signals

filter = True
folder = '/home/alessio/Documents/Data/Neuropixel'

ch_pos = np.load(join(folder, 'channel_positions.npy'))
ch_map = np.load(join(folder, 'channel_map.npy'))

ch_pos_red = ch_pos[:128]
ch_map_red = ch_map[:128]

nchan = 385
fs = 30000 * pq.Hz
data = np.fromfile(join(folder, 'rawDataSample.bin'), dtype='int16')
nsamples = int(len(data)/nchan)

nsec = 30

data = data.reshape((nchan, nsamples))
recordings = np.squeeze(data[ch_map_red, :nsec*int(fs)])

mea_pos, mea_pitch, mea_dim = MEA.return_mea('NeuroPixels-128-v1')
mea_pos = np.hstack((np.zeros((128,1)), ch_pos_red))

if filter:
    bp = [300, 6000] * pq.Hz
    recordings_f = filter_analog_signals(recordings, freq=bp, fs=fs)
else:
    recordings_f = recordings

plot_mea_recording(recordings_f, mea_pos, mea_pitch)

plt.ion()
plt.show()
