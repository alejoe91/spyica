import os

import spikeinterface.comparison as sc
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import numpy as np
from spikeinterface import extract_waveforms, aggregate_units
from spikeinterface.core.base import load_extractor
from ..spyICAsorter import SpyICASorter
from ..preprocessing import subtract_templates
from ..tools import clean_correlated_sources
from pathlib import Path
from typing import Union
from shutil import rmtree


def _set_sorters_params(sorters_params_dict, we_params_dict):
    if 'sorters_params' not in sorters_params_dict.keys():
        sorters_params_dict['sorters_params'] = {}
    if 'engine' not in sorters_params_dict.keys():
        sorters_params_dict['engine'] = 'loop'
    if 'engine_kwargs' not in sorters_params_dict.keys():
        sorters_params_dict['engine_kwargs'] = {}
    if 'verbose' not in sorters_params_dict.keys():
        sorters_params_dict['verbose'] = False
    if 'with_output' not in sorters_params_dict.keys():
        sorters_params_dict['with_output'] = True
    if 'docker_images' not in sorters_params_dict.keys():
        sorters_params_dict['docker_images'] = {}

    if 'load_if_exists' not in we_params_dict.keys():
        we_params_dict['load_if_exists'] = False
    if 'precompute_template' not in we_params_dict.keys():
        we_params_dict['precompute_template'] = ('average',)
    if 'ms_before' not in we_params_dict.keys():
        we_params_dict['ms_before'] = 3.
    if 'ms_after' not in we_params_dict.keys():
        we_params_dict['ms_after'] = 4.
    if 'max_spikes_per_unit' not in we_params_dict.keys():
        we_params_dict['max_spikes_per_unit'] = 500
    if 'return_scaled' not in we_params_dict.keys():
        we_params_dict['return_scaled'] = True
    if 'dtype' not in we_params_dict.keys():
        we_params_dict['dtype'] = None
    return sorters_params_dict, we_params_dict


def _get_saved_recordings(output_folder):
    output_folder = Path(output_folder)
    subfolders = [sub for sub in output_folder.iterdir() if sub.is_dir()]
    recordings_backprojected = [load_extractor(sub) for sub in subfolders]
    return recordings_backprojected


def _compare_one_sorter(sorter_name, sortings_pre, agg_sortings, gts, comparisons):
    print(f'Performance comparisons for {sorter_name}')
    for key in sortings_pre.keys():
        if key[1] == sorter_name:
            print(f'Recording name: {key[0]}')
            comparison_post = sc.compare_sorter_to_ground_truth(tested_sorting=agg_sortings[key],
                                                                gt_sorting=gts[key[0]])
            print('Before recovery:')
            print(comparisons[key].print_performance())
            print('After recovery:')
            print(comparison_post.print_performance())


def _do_recovery_loop(task_args):

    key, well_detected_score, isi_thr, fr_thr, sample_window_ms, \
    percentage_spikes, balance_spikes, detect_threshold, method, skew_thr, n_jobs, we_params, compare, \
    output_folder, job_kwargs = task_args
    recording = load_extractor(output_folder / 'back_recording' / key[1] / key[0])
    if compare is True:
        gt = load_extractor(output_folder / 'back_recording' / key[1] / (key[0] + '_gt'))
    else:
        gt = None
    sorting = load_extractor(output_folder / 'back_recording' / key[0] / (key[1] + '_pre'))
    we = extract_waveforms(recording, sorting,
                           folder=output_folder / 'waveforms' / key[0] / key[1],
                           load_if_exists=we_params['load_if_exists'],
                           ms_before=we_params['ms_before'], ms_after=we_params['ms_after'],
                           max_spikes_per_unit=we_params['max_spikes_per_unit'],
                           return_scaled=we_params['return_scaled'], dtype=we_params['dtype'],
                           overwrite=True, **job_kwargs)
    if gt is not None:
        comparison = sc.compare_sorter_to_ground_truth(tested_sorting=sorting, gt_sorting=gt)
        selected_units = comparison.get_well_detected_units(well_detected_score)
        print(key[1][:-1])
        if key[1] == 'hdsort':
            selected_units = [unit - 1000 for unit in selected_units]
    else:
        isi_violation = st.compute_isi_violations(we)[0]
        good_isi = np.argwhere(np.array(list(isi_violation.values())) < isi_thr)[:, 0]

        firing_rate = st.compute_firing_rate(we)
        good_fr_idx_up = np.argwhere(np.array(list(firing_rate.values())) < fr_thr[1])[:, 0]
        good_fr_idx_down = np.argwhere(np.array(list(firing_rate.values())) > fr_thr[0])[:, 0]

        selected_units = [unit for unit in range(sorting.get_num_units())
                          if unit in good_fr_idx_up and unit in good_fr_idx_down and unit in good_isi]

    templates = we.get_all_templates()
    templates_dict = {str(unit): templates[unit - 1] for unit in selected_units}

    recording_subtracted = subtract_templates(recording, sorting,
                                              templates_dict, we.nbefore, selected_units)

    sorter = SpyICASorter(recording_subtracted)
    sorter.mask_traces(sample_window_ms=sample_window_ms, percent_spikes=percentage_spikes,
                       balance_spikes_on_channel=balance_spikes, detect_threshold=detect_threshold,
                       method=method, **job_kwargs)
    sorter.compute_ica(n_comp='all')
    cleaning_result = clean_correlated_sources(recording, sorter.W_ica, skew_thresh=skew_thr, n_jobs=n_jobs,
                                               chunk_size=recording.get_num_samples(0) // n_jobs,
                                               **job_kwargs)
    sorter.A_ica[cleaning_result[1]] = -sorter.A_ica[cleaning_result[1]]
    sorter.W_ica[cleaning_result[1]] = -sorter.W_ica[cleaning_result[1]]
    sorter.source_idx = cleaning_result[0]
    sorter.cleaned_A_ica = sorter.A_ica[cleaning_result[0]]
    sorter.cleaned_W_ica = sorter.W_ica[cleaning_result[0]]

    ica_recording = st.preprocessing.lin_map(recording_subtracted, sorter.cleaned_W_ica)
    recording_back = st.preprocessing.lin_map(ica_recording, sorter.cleaned_A_ica.T)
    recording_back.save_to_folder(folder=output_folder / 'back_recording' / key[0] / key[1])


class UnitsRecovery:

    def __init__(self, sorters: list, recordings: Union[list, dict], gt=None, sorters_params={}, output_folder=None,
                 overwrite=False, we_params={}, well_detected_score=.7, isi_thr=.3, fr_thr=None, sample_window_ms=2,
                 percentage_spikes=None, balance_spikes=False, detect_threshold=5, method='locally_exclusive',
                 skew_thr=0.1, n_jobs=4, parallel=False, **job_kwargs):
        """
        Apply spike sorting algorithm two times to increase its accuracy. After the first sorting, well detected units
        are removed from the recording. ICA is run on the "new recording" to increase its SNR and then ease the detection
        of small units. Finally, the spike sorting algorithm is run again on the ica-filtered recording.

        Multiple sorting algorithms can be run at the same time, each one on its own recording. The recovery can be run
        in parallel or in a loop. The former option is suggested if the number of recordings or sortings is high.
        Parameters
        ----------
        sorters: list
            list of sorters name to be run.
        recordings: list or dict
            list or dict of RecordingExtractors. If dict, the keys are sorter names.
        gt: list or dict
            list or dict of ground truth SortingExtractors.
        sorters_params: dict
            dict with keys the parameters of spikeinterface.sorters.run_sorters().
            If a parameter is not set, its default values is used.
        output_folder: str
            String with name or path of the output folder. If none it is named 'recovery_output'
        overwrite: bool
            If True and output_folder exists, it will be overwritten. If false and output_folder exists an exception is raised.
        we_params:
            dict with keys the parameters of spikeinterface.core.extract_waveforms().
            If a parameter is not set, its default values is used.
        well_detected_score: float
            agreement score to mark a unit as well detected. Used only if gt is provided.
        isi_thr: float
            If the ISI violation ratio of a unit is above the threshold, it will be discarded.
        fr_thr: list
            list with 2 values. If the firing rate of a unit is not in the provided interval,
            it will be discarded.
        sample_window_ms: list or int
            If list [ms_before, ms_after] of recording selected for each detected spike in subsampling for ICA.
        percentage_spikes: float
            percentage of detected spikes to be used in subsampling for ICA. If None, all spikes are used.
        balance_spikes: bool
            If true, same percentage of spikes is selected channel by channel. If None, spikes are picked randomly.
            Used only if percentage_spikes is not None
        detect_threshold: float
            MAD threshold for spike detection in subsampling for ICA.
        method: str
            How to detect peaks:
            * 'by_channel' : peak are detected in each channel independently. (default)
            * 'locally_exclusive' : locally given a radius the best peak only is taken but
              not neighboring channels.
        skew_thr: float
            Skewness threshold for ICA sources cleaning. If the skewness is lower than the threshold,
            it will be discarded.
        n_jobs: int
            Number of parallel processes
        parallel: bool
            If True, the recovery is run in parallel for each sorter. If False, the recovery is run in loop.
        job_kwargs: dict
            Parameters for parallel processing of RecordingExtractors.

        Returns
        --------
        unitsrecovery object
        """
        self._sorters = sorters
        if output_folder is None:
            output_folder = 'recovery_output'
        self._output_folder = Path(output_folder)
        if fr_thr is None:
            fr_thr = [3.5, 19.5]
        self._params_dict = {'wd_score': well_detected_score, 'isi_thr': isi_thr, 'fr_thr': fr_thr,
                             'parallel': parallel,
                             'sample_window_ms': sample_window_ms, 'percentage_spikes': percentage_spikes,
                             'balance_spikes': balance_spikes, 'detect_threshold': detect_threshold,
                             'method': method, 'skew_thr': skew_thr, 'n_jobs': n_jobs, 'job_kwargs': job_kwargs}

        self._sorters_params, self._we_params = _set_sorters_params(sorters_params, we_params)

        # assert len(sorters) == len(recordings), "The number of sorters must equal the number of recordings"
        if self._output_folder.is_dir() and not overwrite:
            raise Exception('Output folder already exists. Set overwrite=True to overwrite it')
        elif self._output_folder.is_dir() and overwrite:
            rmtree(self._output_folder )
        self._sortings_pre = ss.run_sorters(sorters, recordings, working_folder=self._output_folder / 'sorting_pre',
                                            sorter_params=self._sorters_params['sorters_params'],
                                            mode_if_folder_exists='overwrite', engine=self._sorters_params['engine'],
                                            engine_kwargs=self._sorters_params['engine_kwargs'],
                                            verbose=self._sorters_params['verbose'],
                                            with_output=self._sorters_params['with_output'])

        if not isinstance(recordings, dict):
            self._recordings = {key[0]: recordings[int(key[0][-1])] for key in self._sortings_pre.keys()}
        else:
            self._recordings = recordings
        if not isinstance(gt, dict) and gt is not None:
            assert len(recordings) == len(gt), "Recordings and gts must be of same length"
            self._gt = {key[0]: gt[int(key[0][-1])] for key in self._sortings_pre.keys()}
        else:
            if isinstance(gt, dict) and isinstance(recordings, dict):
                assert gt.keys() == recordings.keys(), "Recordings and gts dictionaries must have same keys"
            self._gt = gt
        if gt is not None:
            self._comparisons = {}
            self._compare = True
        else:
            self._compare = False
        self._recordings_backprojected = {}
        self._aggregated_sortings = {}
        self._sortings_post = {}

    def run(self):

        task_args_list = []
        for key in self._sortings_pre.keys():
            # recording_dict = self._recordings[key[0]].to_dict()
            # sorting_dict = self._sortings_pre[key].to_dict()
            # gt_dict = self._gt[key[0]].to_dict() if self._gt is not None else None
            # comparison = sc.compare_sorter_to_ground_truth(tested_sorting=self._sortings_pre[key], gt_sorting=self._gt[key[0]])
            # self._comparisons[key] = comparison
            # task_args_list.append((recording_dict, gt_dict, sorting_dict, key,
            #                        self._params_dict['wd_score'], self._params_dict['isi_thr'],
            #                        self._params_dict['fr_thr'], self._params_dict['sample_window_ms'],
            #                        self._params_dict['percentage_spikes'], self._params_dict['balance_spikes'],
            #                        self._params_dict['detect_threshold'], self._params_dict['method'],
            #                        self._params_dict['skew_thr'], self._params_dict['n_jobs'], self._we_params,
            #                        comparison, self._output_folder, self._params_dict['job_kwargs']))
            self._recordings[key[0]].save_to_folder(folder=self._output_folder / 'back_recording' / key[1] / key[0])
            self._sortings_pre[key].save_to_folder(folder=self._output_folder / 'back_recording' / key[0] / (key[1] + '_pre'))
            self._gt[key[0]].save_to_folder(folder=self._output_folder / 'back_recording' / key[1] / (key[0] + '_gt'))
            task_args_list.append((key, self._params_dict['wd_score'], self._params_dict['isi_thr'],
                                   self._params_dict['fr_thr'], self._params_dict['sample_window_ms'],
                                   self._params_dict['percentage_spikes'], self._params_dict['balance_spikes'],
                                   self._params_dict['detect_threshold'], self._params_dict['method'],
                                   self._params_dict['skew_thr'], self._params_dict['n_jobs'], self._we_params,
                                   self._compare, self._output_folder, self._params_dict['job_kwargs']))

        if self._params_dict['parallel']:
            # raise NotImplementedError()
            from joblib import Parallel, delayed
            Parallel(n_jobs=self._params_dict['n_jobs'], backend='loky')(
                delayed(_do_recovery_loop)(task_args) for task_args in task_args_list)
        else:
            for task_args in task_args_list:
                _do_recovery_loop(task_args)

        for key in self._sortings_pre.keys():
            if key[1] in self._recordings_backprojected.keys():
                self._recordings_backprojected[key[1]].append(
                    load_extractor(self._output_folder / 'back_recording' / key[0] / key[1]))
            else:
                self._recordings_backprojected[key[1]] = \
                    [load_extractor(self._output_folder / 'back_recording' / key[0] / key[1])]

        for sorter in self._recordings_backprojected.keys():
            self._sortings_post[sorter] = ss.run_sorters(sorter, self._recordings_backprojected[sorter],
                                                         working_folder=self._output_folder / 'sortings_post' / sorter,
                                                         sorter_params=self._sorters_params['sorters_params'],
                                                         mode_if_folder_exists='overwrite',
                                                         engine=self._sorters_params['engine'],
                                                         engine_kwargs=self._sorters_params['engine_kwargs'],
                                                         verbose=self._sorters_params['verbose'],
                                                         with_output=self._sorters_params['with_output'])
            for key in self._sortings_post[sorter].keys():
                self._aggregated_sortings[key] = aggregate_units(
                    [self._sortings_post[sorter][key], self._sortings_pre[key]])
                self._comparisons[key] = sc.compare_sorter_to_ground_truth(tested_sorting=self._sortings_pre[key],
                                                                           gt_sorting=self._gt[key[0]])

    def compare_performance(self, sorter_name=None):
        if sorter_name is not None:
            _compare_one_sorter(sorter_name, self._sortings_pre, self._aggregated_sortings,
                                self._gt, self._comparisons)
        else:
            for sorter in self._sortings_post.keys():
                _compare_one_sorter(sorter, self._sortings_pre, self._aggregated_sortings,
                                    self._gt, self._comparisons)

    @property
    def sortings_pre(self):
        return self._sortings_pre

    @property
    def sortings_post(self):
        return self._sortings_post

    @property
    def recordings_backprojected(self):
        return self._recordings_backprojected

    @property
    def comparisons(self):
        return self._comparisons

    @property
    def aggregated_sortings(self):
        return self._aggregated_sortings


def units_recovery(*args, **kwargs):
    rec = UnitsRecovery(*args, **kwargs)
    rec.run()
    return rec


units_recovery.__doc__ = UnitsRecovery.__doc__
