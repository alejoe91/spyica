import numpy as np
import os, sys
import matplotlib.pylab as plt
from os.path import join
import quantities as pq
import MEAutility as mea
from tools import *
from neuroplot import *
import neo
import ipdb
import yaml
from docx import Document


# Load data from bin files into a matrix of n channels x m samples 
def loadRawData(filename,numChannels,dtype):
    fdata = np.fromfile(filename,dtype=dtype)
    numsamples = len(fdata) / numChannels
    data = np.reshape(fdata,(numsamples,numChannels))
    return (np.transpose(data))


def find_peaks(data, threshold, minstep=0):
    derivative = np.diff(np.sign(np.diff(data)))
    if threshold > 0:
        derivative = derivative < 0
    else:
        derivative = derivative > 0

    peaks = derivative.nonzero()[0] + 1  # local max
    if threshold > 0:
        peaks = peaks[data[peaks] > threshold]
    else:
        peaks = peaks[data[peaks] < threshold]

    if minstep > 0:
        gpeaks = split_list_pairwise(peaks, lambda x, p: x - p > minstep)
        peaks = np.array([g[np.argmax([data[i] for i in g])] for g in gpeaks])
    return peaks

def normalize(data):
    return data - np.reshape(np.mean(data,1),[8,1])

def split_list_pairwise(l,p):
    groups = []
    prev = None
    group = None
    for x in l:
        if prev is None or p(x,prev):
            group = []
            groups.append(group)
        group.append(x)
        prev = x
    return groups


def loadReadMe(filename):
    document = Document(filename)
    pars = {}
    for para in document.paragraphs:
        print para.text
        para_split = para.text.split()
        if len(para_split) > 1:
            if '=' in para_split:
                if 'Sampling' in para_split[0]:
                    freq = int(para_split[-2] + para_split[-1]) * pq.Hz
                    pars.update({para_split[0]: freq})
                elif 'dtype' in para_split[0]:
                    pars.update({para_split[0]: para_split[-1]})
                elif 'step_size' in para_split[0].lower() or 'digitization' in para_split[0].lower()\
                        or 'gain' in para_split[0].lower() or 'range' in para_split[0].lower():
                    pars.update({para_split[0]: float(para_split[-1])})
                else:
                    pars.update({para_split[0]: int(para_split[-1])})
        elif '=' in para.text:
            para_split = para.text.split('=')
            if 'Sampling' in para_split[0]:
                freq = int(para_split[-2] + para_split[-1]) * pq.Hz
                pars.update({para_split[0]: freq})
            elif 'dtype' in para_split[0]:
                pars.update({para_split[0]: para_split[-1]})
            elif 'step_size' in para_split[0].lower() or 'digitization' in para_split[0].lower() \
                    or 'gain' in para_split[0].lower() or 'range' in para_split[0].lower():
                pars.update({para_split[0]: float(para_split[-1])})
            else:
                pars.update({para_split[0]: int(para_split[-1])})
    return pars


if __name__ == '__main__':
    if '-r' in sys.argv:
        pos = sys.argv.index('-r')
        folder_name = sys.argv[pos + 1]
    else:
        raise Exception('Provide experimental data folder')
    if '-nosave' in sys.argv:
        save = False
    else:
        save = True

    pars = loadReadMe(join(folder_name, 'READ ME.docx'))

    Probe_numChannels = pars['Probe_numChannels']
    if Probe_numChannels == 32:
        electrode_name = 'Neuronexus-32-Kampff'
    elif Probe_numChannels == 128:
        electrode_name = 'NeuroSeeker-128-Kampff'
        Probe_voltage_step_size = pars['Probe_voltage_step_size']*1e6
        Probe_y_digitization = pars['Probe_y_digitization']

    fs = pars['Sampling_frequency']
    Probe_dtype = str(pars['Probe_dtype'].split('.')[-1])
    Juxta_numChannels = pars['Juxta_numChannels']
    Juxta_dtype =  str(pars['Juxta_dtype'].split('.')[-1])
    Juxta_ADC_used_channel = pars['Juxta_ADC_used_channel']
    Juxta_Gain = pars['Juxta_Gain']
    Juxta_y_digitization = pars['Juxta_y_digitization']
    Juxta_y_range = pars['Juxta_y_range']
    elec_root = electrode_name.split('-')[0]

    split_probe = False
    nsec_cut = 120
    filter = True
    threshold = 1
    minstep = 100

    exp_name = os.path.split(folder_name)[-1]

    file_extra = [join(folder_name, f) for f in os.listdir(folder_name) if 'amplifier' in f][0]
    file_juxta = [join(folder_name, f) for f in os.listdir(folder_name) if 'adc' in f][0]

    # Open Amplifier file
    extra = loadRawData(file_extra, numChannels=Probe_numChannels, dtype=Probe_dtype)
    if 'NeuroSeeker' in electrode_name:
        extra_Volts = (np.float32(extra) - Probe_y_digitization) * Probe_voltage_step_size
    else:
        extra_Volts = np.float32(extra)
    # Open ADC file
    juxta = loadRawData(file_juxta, numChannels = Juxta_numChannels, dtype = Juxta_dtype)

    extra_v_cut = extra_Volts[:, :int(nsec_cut * fs)]
    juxta_cut = juxta[:, :int(nsec_cut * fs)]

    triggerchannel = Juxta_ADC_used_channel
    juxta_factor_scale = Juxta_y_range / (Juxta_y_digitization * float(Juxta_Gain))
    juxta_scaled_mV_offset = normalize(juxta_cut[:, :])
    juxta_scaled_mV = (juxta_scaled_mV_offset[:, :]) * juxta_factor_scale * 1000.
    trigchannel = juxta_scaled_mV[triggerchannel, :]
    triggers = find_peaks(trigchannel, threshold, minstep)  # Trigger on spikes

    spiketrain = [neo.SpikeTrain(triggers/fs, t_start=0*pq.s, t_stop=nsec_cut*pq.s)]

    mea_pos, mea_dim, mea_pitch = mea.return_mea(electrode_name, sortlist=None)
    mea_info = mea.return_mea_info(electrode_name)
    if 'sortlist' in mea_info.keys():
        recordings = extra_v_cut[mea_info['sortlist']]

    if split_probe:
        nsplit = 4.
        step = (np.max(mea_pos[:, 2]) - np.min(mea_pos[:, 2])) / nsplit

        rec = []
        pos = []
        for i in np.arange(nsplit):
            pos_idx = np.where((mea_pos[:, 2] >= i * step) & (mea_pos[:, 2] < i + 1 * step))
            rec.append(recordings[pos_idx, :])
            pos.append(mea_pos[pos_idx])

    times = np.arange(recordings.shape[1]) / fs

    del extra, extra_Volts, extra_v_cut, juxta



    if filter:
        bp = [300, 6000] * pq.Hz
        recordings_f = filter_analog_signals(recordings, freq=bp, fs=fs)
    else:
        recordings_f = recordings

    plot_mea_recording(recordings_f, mea_pos, mea_pitch)

    save_folder = join('recordings', 'exp', 'neto', elec_root, exp_name)

    if save:
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        np.save(join(save_folder, 'recordings'), recordings_f)
        np.save(join(save_folder, 'spiketrains'), spiketrain)
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