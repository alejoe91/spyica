import numpy as np
import os
import matplotlib.pylab as plt
from os.path import join
import quantities as pq
from tools import *
import MEAutility as MEA
from neuroplot import *
from statsmodels.tsa.arima_model import ARMA, arma_generate_sample
import scipy.signal as ss

Probe_numChannels = 32
Probe_dtype = np.uint16
Probe_voltage_step_size = 0.195e-6
Probe_y_digitization = 32768.
fs = 30000.*pq.Hz
Juxta_numChannels = 8
Juxta_dtype = np.uint16
Juxta_ADC_used_channel = 0
Juxta_Gain = 100.
Juxta_y_digitization=65536.
Juxta_y_range=10.

nsec_cut = 30. * pq.s
nsamples = int((nsec_cut*fs).magnitude)
load_juxta = False

folder_name = '/home/alessio/Documents/Data/Neto/2014_11_25_Pair_3_0'

file_extra = [join(folder_name, f) for f in os.listdir(folder_name) if 'adc' in f][0]
file_juxta = [join(folder_name, f) for f in os.listdir(folder_name) if 'amplifier' in f][0  ]

# Load data from bin files into a matrix of n channels x m samples 
def loadRawData(filename,numChannels,dtype):
    fdata = np.fromfile(filename,dtype=dtype)
    numsamples = len(fdata) / numChannels
    data = np.reshape(fdata,(numsamples,numChannels))
    return (np.transpose(data))

# Open Amplifier file 
extra = loadRawData(file_extra, numChannels = Probe_numChannels, dtype = Probe_dtype)
extra_Volts = (extra - Probe_y_digitization) * Probe_voltage_step_size * pq.uV

if load_juxta:
    #Open ADC file
    juxta = loadRawData(file_juxta, numChannels = Juxta_numChannels, dtype = Juxta_dtype)
    juxta_channel_Volts= juxta [Juxta_ADC_used_channel, :] * (Juxta_y_range /( Juxta_y_digitization * Juxta_Gain))

extra_times = np.arange(extra_Volts.shape[1])/fs

extra_v_cut = extra_Volts[:, :nsamples]
times_cut = extra_times[:nsamples]

del extra_Volts, extra
# filter
bp = [300, 6000]*pq.Hz
if fs/2. < bp[1]:
    recordings = filter_analog_signals(extra_v_cut, freq=bp[0], fs=fs,
                                            filter_type='highpass')
else:
    recordings = filter_analog_signals(extra_v_cut, freq=bp, fs=fs)

mea_pos, mea_dim, mea_pitch = MEA.return_mea('Neuronexus-32-Kampff')

plot_mea_recording(recordings, mea_pos, mea_pitch, lw=0.3)


plt.figure(); plt.plot(times_cut, recordings.T)

points = plt.ginput(2)
t1 = points[0][0]*pq.s
t2 = points[1][0]*pq.s

idxs_noise = np.where((times_cut > t1) & (times_cut < t2))

times_noise = times_cut[idxs_noise]
v_noise = recordings[:, idxs_noise[0]]

noise_corr = np.cov(v_noise)
plt.matshow(noise_corr)


f, psd = ss.welch(v_noise, fs=fs)
plt.figure()
plt.plot(f, psd.T)

# choose 1 channel
noise = v_noise[5]
aic = []
pd = []

for p in range(5):
    for d in range(5):
        try:
            model = ARMA(noise, (p, d)).fit()

            x  =model.aic
            x1 = p,d
            print (x1,x)

            aic.append(x)
            pd.append(x1)
        except:
            print 'Pass'
            pass

# print model.summary()

plt.ion()
plt.show()



