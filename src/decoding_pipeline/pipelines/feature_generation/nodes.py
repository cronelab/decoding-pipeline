"""
This is a boilerplate pipeline 'feature_generation'
generated using Kedro 0.18.3
"""
import copy
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt

def _apply_car_filter(ch_include, ch_car_list, car_include, signals):
    ch_car_valid = []
    for (idx, cur_car_set) in ch_car_list:
        ch_car_valid.append([x for x in cur_car_set if x in ch_include])
    
    n_samples = signals.shape[0]
    signals_car = copy.deepcopy(signals)

    if ch_car_valid:
        n_sublists = len(ch_car_valid)
        for sublist_idx in range(n_sublists):
            cur_valid_car_set = ch_car_valid[sublist_idx]
            n_ch_car_set = len(cur_valid_car_set)

            if n_ch_car_set > 1:
                ch_car_idx_list = [ch_include.index(ch) for ch in cur_valid_car_set]

                signals_car[:, np.array(ch_car_idx_list)] = signals[:, np.array(ch_car_idx_list)] - np.mean(signals[:, np.array(ch_car_idx_list)], axis=1)[:, np.newaxis]
                
    else:
        signals_car = signals - np.mean(signals, axis=1)[:, np.newaxis]

    return {
        'signals_car': signals_car if car_include else signals,
        'ch_car': ch_car_valid
    }

def car_filter(selected_channels, car_channel_params, patient_id, partitioned_data):
    ch_include = selected_channels['ch_include']
    car_include = car_channel_params[patient_id]['include']
    ch_car_list = car_channel_params[patient_id]['channels']

    save_dict = {}
    for partition_key, partition_load_func in partitioned_data.items():
        data_dict = partition_load_func()

        signals = data_dict['signals']

        car_signals_dict = _apply_car_filter(ch_include, ch_car_list, car_include, signals)
        
        save_dict[partition_key] = car_signals_dict

    return save_dict 

def _generate_spectrogram(signals, sampling_rate, spectrogram_params):
    window_length = spectrogram_params['window_length']
    shift = spectrogram_params['shift']
    logarithm = spectrogram_params['logarithm']

    factor = int(sampling_rate/1000)
    window_length = window_length * factor
    shift = shift * factor

    fs = 1e3 * factor

    noverlap = window_length - shift
    nperseg = window_length

    f, t, sxx = signal.spectrogram(signals, fs=fs, noverlap=noverlap, nperseg=nperseg, axis=0)

    sxx = np.moveaxis(sxx, -2, -1)

    return {
        'f': f,
        't': t,
        'sxx': sxx
    }

def generate_spectrogram(partitioned_data, partitioned_data_fs, spectrogram_params):

    sampling_rate = list(partitioned_data_fs.values())[0]()['sampling_rate']

    save_dict = {}
    for partition_key, partition_load_data in partitioned_data.items():
        signals = partition_load_data['signals_car']

        save_dict[partition_key] = lambda: _generate_spectrogram(signals, sampling_rate, spectrogram_params)

    return save_dict

def plot_spectrogram_transform_figure(partitioned_data):
    sxx = list(partitioned_data.values())[0]()['sxx']

    a, bins = np.histogram(np.log10(sxx.flatten()), bins=100)
    b, b_bins = np.histogram(sxx.flatten(), bins=100)

    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 5))

    ax2.hist(np.log10(sxx.flatten()), bins, density=False) 
    ax1.hist(sxx.flatten(), b_bins, density=False) 
    ax1.set_title("Spectrogram Raw Values")
    ax2.set_title("Spectrogram Log10 Values")
    ax1.set_ylabel("Count")
    ax2.set_ylabel("Count")

    fig.subplots_adjust(hspace=0.4)

    plt.close()

    return fig