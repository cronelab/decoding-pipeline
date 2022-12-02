"""
This is a boilerplate pipeline 'feature_generation'
generated using Kedro 0.18.3
"""
import copy
from turtle import update
import numpy as np
import scipy.signal as signal

import matplotlib.pyplot as plt
import matplotlib as mpl

from scripts.utility_scripts import create_closure, create_closure_func

def _apply_car_filter(ch_include, ch_car_list, car_include, data_dict):
    signals = data_dict['signals']
    stimuli = data_dict['stimuli']
    sampling_rate = data_dict['sampling_rate']
    t_samples = data_dict['t_samples']
    t_seconds = data_dict['t_seconds']

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
        'ch_car': ch_car_valid,
        'signals': signals_car if car_include else signals,
        'stimuli': stimuli,
        'sampling_rate': sampling_rate,
        't_samples': t_samples,
        't_seconds': t_seconds
    }  

def car_filter(selected_channels, car_channel_params, patient_id, partitioned_data):
    ch_include = selected_channels['ch_include']
    car_include = car_channel_params[patient_id]['include']
    ch_car_list = car_channel_params[patient_id]['channels']

    save_dict = {}
    for partition_key, partition_load_func in partitioned_data.items():
        print(partition_key)
        data_dict = partition_load_func()

        print(data_dict)

        signals = data_dict['signals']

        save_dict[partition_key] = create_closure_func(_apply_car_filter, ch_include, ch_car_list, car_include, data_dict)

        # car_signals_dict = _apply_car_filter(ch_include, ch_car_list, car_include, signals)
        
        # save_dict[partition_key] = create_closure(car_signals_dict)
        

        # print(car_signals_dict)

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
        'sxx': sxx,
        'sampling_rate': sampling_rate
    }

def generate_spectrogram(partitioned_data, partitioned_data_fs, spectrogram_params):

    sampling_rate = list(partitioned_data_fs.values())[0]()['sampling_rate']

    save_dict = {}
    for partition_key, partition_load_data in partitioned_data.items():
        signals = partition_load_data()['signals']

        save_dict[partition_key] = create_closure_func(_generate_spectrogram, signals, sampling_rate, spectrogram_params)

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

def _align_and_warp_spectrogram(t, sampling_rate, window_length, window_placement):
    t_shift = t
    if window_placement == 'centered':
        t_shift = t
    elif window_placement == 'causal':
        t_shift = t + window_length/sampling_rate*(1/2)
    elif window_placement == 'anticausal':
        t_shift = t - window_length/sampling_rate*(1/2)
        
    return t_shift

def _create_alignment_and_warping(data_func, spectrogram_params):
    data = data_func()

    window_length = spectrogram_params['window_length']
    window_placement = spectrogram_params['window_placement']
    
    logarithm = spectrogram_params['logarithm']

    sampling_rate = data['sampling_rate']
    t = data['t']
    f = data['f']
    sxx = data['sxx']

    if logarithm:
        sxx = np.log10(sxx)

        sxx[np.isneginf(sxx)] = 0
        sxx[np.isnan(sxx)] = 0
    
    t_shift = _align_and_warp_spectrogram(t, sampling_rate, window_length, window_placement)

    return {
            'f': f,
            't': t_shift,
            'sxx': sxx,
            'sampling_rate': sampling_rate
        }

def align_and_warp_spectrogram(partitioned_data, spectrogram_params):
    save_dict = {}
    for partition_key, partition_load_data in partitioned_data.items():
        save_dict[partition_key] = create_closure_func(_create_alignment_and_warping, partition_load_data, spectrogram_params)
    return save_dict

def _downsample_data_to_spectrogram(sxx_data_func, signal_data_func):
    sxx_data_dict = sxx_data_func()
    signal_data_dict = signal_data_func()

    signals = signal_data_dict['signals']
    stimuli = signal_data_dict['stimuli']
    sampling_rate = signal_data_dict['sampling_rate']
    t_samples = signal_data_dict['t_samples']
    t_seconds = signal_data_dict['t_seconds']

    t_sxx = sxx_data_dict['t']

    # Computing the number of time points in the spectrogram.
    N_t_sxx = len(t_sxx)
    
    # Computing the number of channels in the spectrogram.
    N_chs = signals.shape[1]
    N_stim_chs = stimuli.shape[1]
    
    # Initializing the resampled stimuli and time arrays.
    signals_raw_sxx = np.zeros((N_t_sxx, N_chs))
    stimuli_sxx     = np.zeros((N_t_sxx, N_stim_chs))
    t_samples_sxx   = np.zeros((N_t_sxx,))
    t_seconds_sxx   = np.zeros((N_t_sxx,))
    
    # Iterating through each time sample from the spectrogram.
    for n in range(N_t_sxx):
        
        # Finding the index of the time arrays that most closely matches the time sample from the spectrogram.
        diff_t_sxx   = np.abs(t_seconds - t_sxx[n])
        min_diff_idx = np.argmin(diff_t_sxx)
        
        # Recreating the resampled time and stimuli arrays based on the above-calculated index.
        t_samples_sxx[n]      = t_samples[min_diff_idx]
        t_seconds_sxx[n]      = t_seconds[min_diff_idx]
        stimuli_sxx[n]        = stimuli[min_diff_idx, :]
        signals_raw_sxx[n, :] = signals[min_diff_idx, :]

    return {
        "stimuli": stimuli_sxx,
        "signals": signals_raw_sxx,
        "t_samples": t_samples_sxx,
        "t_seconds": t_seconds_sxx,
        "sampling_rate": sampling_rate
    }

def downsample_data_to_spectrogram(partitioned_sxx_data, partitioned_signal_data):
    save_dict = {}
    for partition_sxx_key, sxx_data_func in partitioned_sxx_data.items():
        signal_data_func = partitioned_signal_data[partition_sxx_key]
        save_dict[partition_sxx_key] = create_closure_func(_downsample_data_to_spectrogram, sxx_data_func, signal_data_func)
    return save_dict

def plot_downsampled_signals(partitioned_sxx_data, partitioned_signal_data, partitioned_sxx_std_data):
    save_dict = {}
    for partition_sxx_key, sxx_data_func in partitioned_sxx_data.items():
        signal_data_func = partitioned_signal_data[partition_sxx_key]
        sxx_std_func = partitioned_sxx_std_data[partition_sxx_key]

        sxx_data_dict = sxx_data_func()
        sxx_std_data_dict = sxx_std_func()
        signal_data_dict = signal_data_func()

        stimuli = signal_data_dict['stimuli']
        signals = signal_data_dict['signals']
        t_seconds = signal_data_dict['t_seconds']

        f =  sxx_data_dict['f']
        t = sxx_data_dict['t']
        sxx = sxx_data_dict['sxx']

        sxx_std = sxx_std_data_dict['sxx']

        fig, (ax, ax1, ax2, ax3) = plt.subplots(4, figsize=(20,10))

        ax.plot(t_seconds, stimuli[:, 0], color='k', linewidth=1)
        ax.margins(x=0)
        ax.set_ylabel('State')

        ax1.plot(t_seconds, signals[:, 0], color='k', linewidth=0.5)
        ax1.margins(x=0)
        ax1.set_ylabel(r'Voltage ($\mu$V)')

        ax2.pcolormesh(
            t,
            f,
            sxx[:,:,0],
            # norm=mpl.colors.PowerNorm(gamma=1.0 / 5),
            cmap="seismic",
            vmin=-3, 
            vmax=3
        #     cmap="YlGnBu"
        )

        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_ylim([0, 140])

        ax3.pcolormesh(
            t,
            f,
            sxx_std[:,:,0],
            # norm=mpl.colors.PowerNorm(gamma=1.0 / 5),
            cmap="seismic",
            vmin=-3, 
            vmax=3
        #     cmap="YlGnBu"
        )

        ax3.set_ylabel('Frequency (Hz)')
        ax3.set_ylim([0, 140])


        ax3.set_xlabel('Time (s)')

        ax.set_title('Downsampled States, Signals, Raw Spectrogram and Standardized Spectrogram')

        save_dict[f"{partition_sxx_key}.png"] = fig

        plt.close()
    return save_dict

def extract_calibration_statistics(partitioned_calibration_sxx, partitioned_calibration_data, selected_sessions, patient_id):
    calibration_dict = selected_sessions[patient_id]['calibration']

    # Find all dates that have no session data. By default, all sessions will be included for calibration
    updated_dict = {}
    for partition_key in list(partitioned_calibration_sxx.keys()):
        # TODO: Switch the array based splitting to regex based splitting
        date = partition_key.split('_')[-2]
        session = partition_key.split('_')[-1]
        if date in list(calibration_dict.keys()):
            continue
        else:
            sessions_list = updated_dict.setdefault(date, [])
            sessions_list.append(session)
            updated_dict[date] = sessions_list
    
    calibration_dict.update(updated_dict)

    stimuli = None
    save_dict = {}
    for date_key, sessions_list in calibration_dict.items():
        intermed_list = []
        for partition_sxx_key, sxx_data_func in partitioned_calibration_sxx.items():
            continue_loop = True
            if date_key in partition_sxx_key and partition_sxx_key.split('_')[-1] in sessions_list:
                all_stimuli=partitioned_calibration_data[partition_sxx_key]()['stimuli']
                stimuli=all_stimuli[:, 0]

                # Check to make sure calibration stimuli is not all zeros
                assert 1 in stimuli, "Stimuli contains all zeros, think about possibly excluding this calibration session"

                continue_loop = False
            
            if continue_loop:
                continue

            data_dict = sxx_data_func()

            sxx = data_dict['sxx']

            sxx = sxx[:, stimuli == 1, :]

            mean = np.mean(sxx, axis=1)[:,np.newaxis,:]
            std = np.std(sxx, axis=1)[:,np.newaxis,:]
            sxx_len = sxx.shape[1]

            intermed_list.append({
                'mean': mean,
                'std': std,
                'count': sxx_len
            })
        
        total_len = np.sum([x['count'] for x in intermed_list])
        fractions_len = [x['count']/total_len for x in intermed_list]

        global_mean = sum([x['mean']*frac for x,frac in zip(intermed_list, fractions_len)])
        global_std = sum([x['std']*frac for x,frac in zip(intermed_list, fractions_len)])

        save_dict[f'Calibration_statistics_{date_key}'] = create_closure({
            'mean': global_mean,
            'std': global_std 
        })
    
    return save_dict

def _standardize_spectrogram(sxx_func, stats_func):
    sxx_dict = sxx_func()
    stats_dict = stats_func()

    sxx = sxx_dict['sxx']
    mean_sxx = stats_dict['mean']
    std_sxx = stats_dict['std']

    sxx_dict['sxx'] = (sxx - mean_sxx)/std_sxx

    return sxx_dict

def standardize_spectrograms(partitioned_sxx, partitioned_statistics):
    
    save_dict = {}
    for partition_sxx_key, sxx_data_func in partitioned_sxx.items():
        date = partition_sxx_key.split('_')[-2]
        statistics_key = list(filter(lambda x: x.split('_')[-1] == date, partitioned_statistics.keys()))[0]

        stats_func = partitioned_statistics[statistics_key]

        save_dict[partition_sxx_key] = create_closure_func(_standardize_spectrogram, sxx_data_func, stats_func)

    return save_dict

def _extract_state_information(stimuli):
    unique_states = np.unique(stimuli)

    state_information_dict = {}
    for state in unique_states:
        state_information_dict[int(state)] = {}
        idx_stimuli = np.where(stimuli == state)[0]

        temp_diff_arr = np.diff(idx_stimuli)
        temp_diff_arr = np.insert(temp_diff_arr, 0, 1)

        disjoint_idx = np.where(temp_diff_arr>1)[0]
        unique_val_idx = np.split(idx_stimuli, disjoint_idx)

        state_information_dict[int(state)]['unique_val_idx'] = unique_val_idx
        state_information_dict[int(state)]['start_end_idx'] = [(x[0], x[-1]) for x in unique_val_idx]
        state_information_dict[int(state)]['num_steps'] = [len(x) for x in unique_val_idx]
    return state_information_dict

def _extract_center_out_indices(stimuli, states_list, stimulus_state, reached_state, sampling_rate):
    stimulus_code_info = _extract_state_information(stimuli[:, np.where(np.array(states_list) == stimulus_state)[0]].flatten())
    result_code_info = _extract_state_information(stimuli[:, np.where(np.array(states_list) == reached_state)[0]].flatten())

    state_dict = {}
    for state in stimulus_code_info.keys():
        state_dict[state] = {}
        stimulus_start_end_list = stimulus_code_info[state]['start_end_idx']
        result_start_end_list = result_code_info[state]['start_end_idx']

        if state == 0:
            stimulus_start_end_list = stimulus_start_end_list[1:-1]
            result_start_end_list = result_start_end_list[1:-1]

        state_dict[state]['start_end_idx'] = [(x[0], y[0]) for x, y in zip(stimulus_start_end_list, result_start_end_list)]
        state_dict[state]['unique_val_idx'] = [np.arange(x[0], x[1] + 1) for x in state_dict[state]['start_end_idx']]
        state_dict[state]['num_steps'] = [len(x) for x in state_dict[state]['unique_val_idx']]
        state_dict[state]['sampling_rate'] = sampling_rate
        
    return state_dict

def extract_center_out_state_dict(partitioned_data, bci_states, state_information, patient_id):
    states_list = np.array(bci_states[patient_id]['center_out'])
    stimulus_state = state_information[patient_id]['center_out']['stimulus_state']
    reached_state = state_information[patient_id]['center_out']['reached_state']

    save_dict = {}
    for partition_key, partition_func in partitioned_data.items():
        data_dict = partition_func()

        stimuli = data_dict['stimuli']
        sampling_rate = data_dict['sampling_rate']

        save_dict[partition_key] = create_closure(_extract_center_out_indices(stimuli, states_list, stimulus_state, reached_state, sampling_rate))

    return save_dict