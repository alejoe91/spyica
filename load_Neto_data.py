import numpy as np
import os
import matplotlib.pylab as plt
from os.path import join
import quantities as pq
import MEAutility as mea
from tools import *
from neuroplot import *
import ipdb
import yaml

# Probe_numChannels = 32
Probe_numChannels = 128

Probe_dtype = np.uint16
Probe_voltage_step_size = 0.195
Probe_y_digitization = 32768.
fs = 30000. * pq.Hz
Juxta_numChannels = 8
Juxta_dtype = np.uint16
Juxta_ADC_used_channel = 0
Juxta_Gain = 100.
Juxta_y_digitization=65536.
Juxta_y_range=10.

# electrode_name = 'Neuronexus-32-Kampff'
electrode_name = 'NeuroSeeker-128-Kampff'
elec_root = electrode_name.split('-')[0]

split_probe = True

nsec_cut = 120
filter=True

# folder_name = '/home/alessio/Documents/Data/Neto/2014_11_25_Pair_3_0'
folder_name = '/home/alessio/Documents/Data/Neto/2015_09_03_Pair_9_0'

exp_name = os.path.split(folder_name)[-1]

file_extra = [join(folder_name, f) for f in os.listdir(folder_name) if 'amplifier' in f][0]
file_juxta = [join(folder_name, f) for f in os.listdir(folder_name) if 'adc' in f][0]

# Load data from bin files into a matrix of n channels x m samples 
def loadRawData(filename,numChannels,dtype):
    fdata = np.fromfile(filename,dtype=dtype)
    numsamples = len(fdata) / numChannels
    data = np.reshape(fdata,(numsamples,numChannels))
    return (np.transpose(data))

# Open Amplifier file 
extra = loadRawData(file_extra, numChannels = Probe_numChannels, dtype = Probe_dtype)
extra_Volts = (np.float32(extra) - Probe_y_digitization) * Probe_voltage_step_size

#Open ADC file 
# juxta = loadRawData(file_juxta, numChannels = Juxta_numChannels, dtype = Juxta_dtype)
# juxta_channel_Volts= juxta [Juxta_ADC_used_channel, :] * (Juxta_y_range /( Juxta_y_digitization * Juxta_Gain))

extra_times = np.arange(extra_Volts.shape[1])/fs

extra_v_cut = extra_Volts[:, :int(nsec_cut*fs)]

mea_pos, mea_dim, mea_pitch = mea.return_mea(electrode_name, sortlist=None)
mea_info = mea.return_mea_info(electrode_name)

if 'sortlist' in mea_info.keys():
    recordings = extra_v_cut[mea_info['sortlist']]

if split_probe:
    nsplit = 4.
    step = (np.max(mea_pos[:, 2]) - np.min(mea_pos[:, 2]))/nsplit

    rec = []
    pos = []
    for i in np.arange(nsplit):
        pos_idx = np.where((mea_pos[:, 2] >= i*step) & (mea_pos[:, 2] < i+1*step))
        rec.append(recordings[pos_idx, :])
        pos.append(mea_pos[pos_idx])


times = np.arange(recordings.shape[1])/fs

del extra, extra_Volts, extra_times, extra_v_cut

if filter:
    bp = [300, 6000]*pq.Hz
    recordings_f = filter_analog_signals(recordings, freq=bp, fs=fs)
else:
    recordings_f = recordings

plot_mea_recording(recordings_f[:, :100000], mea_pos, mea_pitch)

save_folder = join('recordings', 'exp', 'neto', elec_root, exp_name)

if not os.path.isdir(save_folder):
    os.makedirs(save_folder)

np.save(join(save_folder, 'recordings'), recordings_f)

with open(join(save_folder, 'rec_info.yaml'), 'w') as f:
    general = {'spike_folder': folder_name, 'rotation': '',
               'pitch': mea_pitch, 'electrode name': str(electrode_name),
               'MEA dimension': mea_dim, 'fs': fs, 'duration': str(nsec_cut),
               'seed': ''}
    filter = {'filter': filter, 'bp': str(bp)}
    if filter:
        info = {'General': general, 'Filter': filter}
    else:
        info = {'General': general}
    yaml.dump(info, f)


