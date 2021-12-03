# SpyICA - offline and online ICA-based spike sorting toolbox

SpyICA is a python package that implements ICA-based spike sorting algorithms.
The core algorithms are based on FastICA and Online Recursive ICA and can be used offline or tested as if the data were coming as an online stream.

### Installation

In order to install SpyICA clone the repo and run the python installer:

```
git clone https://github.com/alejoe91/spyica.git
cd spyica
python setup.py install (or develop)
```

### Basic usage

SpyICA is integrated with SpikeInterface (https://github.com/SpikeInterface). The spike sorting functions take a `RecordingExtractor` object as input and return a `SortingExtractor`.

```
import spikeextractors as se
import spyica

recording = se.SomeRecordingExtractor('path-to-file')

# offline ICA-based spike sorting
sorting = spyica.ica_spike_sorting(recording)


# offline ORICA-based spike sorting
sorting = spyica.orica_spike_sorting(recording)

# online ORICA-based spike sorting
sorting = spyica.online_orica_spike_sorting(recording)
```

### Units recovery

SpyICA can apply ICA to improve the performance of other sorters. The recovery function takes a list or dictionary of `RecordingExtractor` objects, a list or dictionary of ground truth `SortingExtractor` objects (if available) and a list of sorter names as input. It returns a dictionary of `SortingExtractor` objects.

```
import spikeinterface.extractors as se
from spyica.unitsrecovery import units_recovery

recording1 = se.read_somerecording('path-to-file1')
...
recordingN = se.read_somerecording('path-to-fileN')

sorters_list = ['sorter1_name', ..., 'sorterN_name'] 

#if ground truth is not available
recovery_object = units_recovery(sorters_list, [recording1, ..., recordingN])

#if ground truth (gt) is available
recovery_object = units_recovery(sorters_list, [recording1, ..., recordingN], gt=gt_list_or_dict)
```

### References

If you use SpyICA, please cite:

```
@inproceedings{buccino2018independent,
  title={Independent Component Analysis for Fully Automated Multi-Electrode Array Spike Sorting},
  author={Buccino, Alessio P and Hagen, Espen and Einevoll, Gaute T and H{\"a}fliger, Philipp D and Cauwenbergh, Gert},
  booktitle={2018 40th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC)},
  pages={2627--2630},
  year={2018},
  organization={IEEE}
}
```

```
@inproceedings{buccino2018real,
  title={Real-Time Spike Sorting for Multi-Electrode Arrays with Online Independent Component Analysis},
  author={Buccino, Alessio Paolo and Hsu, Sheng-Hsiou and Cauwenberghs, Gert},
  booktitle={2018 IEEE Biomedical Circuits and Systems Conference (BioCAS)},
  pages={1--4},
  year={2018},
  organization={IEEE}
}
```

