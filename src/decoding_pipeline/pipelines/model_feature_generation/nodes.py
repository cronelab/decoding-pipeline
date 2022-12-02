"""
This is a boilerplate pipeline 'model_feature_generation'
generated using Kedro 0.18.3
"""
from scripts.utility_scripts import create_closure, create_closure_func

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def _window_spectrogram(data_func, indices, shift, window_size, precision):
    sxx_data_dict = data_func()
    
    sxx = sxx_data_dict['sxx']
    
    return np.moveaxis(sliding_window_view(sxx[:, indices, :][:, ::shift, :], window_shape=window_size, axis=1), [0, -1], [1, -2]).astype(precision)
    

def generate_model_windowed_sxx_data(spectrogram_dict, curated_states_partition, sessions, model_data_params, current_experiment, patient_id):
    pre_stimulus_time = model_data_params['pre_stimulus_time']
    post_completion_time = model_data_params['post_completion_time']
    precision = model_data_params['precision']

    window_size = model_data_params['window_size']
    shift = model_data_params['shift']

    model_data_dict = {}
    model_data_filenames_dict = {}
    global_trial_idx = 0
    for session_type, session_data in sessions[patient_id][current_experiment].items():
        for sxx_partition_key, sxx_partition_func in spectrogram_dict.items():
            date = sxx_partition_key.split('_')[-2]
            session = sxx_partition_key.split('_')[-1]

            if date in session_data.keys() and session in session_data[date]:
                curated_states_data_func = curated_states_partition[sxx_partition_key]
                curated_states_dict = curated_states_data_func()

                model_data_metadata = {}
                for state, state_information in curated_states_dict.items():
                    cur_dict = {}

                    start_end_idx = state_information['start_end_idx']
                    
                    sampling_rate = state_information['sampling_rate']
                    
                    samples_pre = int(np.ceil((pre_stimulus_time * sampling_rate)/shift))
                    samples_post = int(np.ceil((post_completion_time * sampling_rate)/shift)) 
                    
                    cur_dict['start_end_idx'] = [(x[0]-samples_pre, x[1]+samples_post) for x in start_end_idx]
                    cur_dict['unique_val_idx'] = [np.arange(x[0], x[1]+1) for x in cur_dict['start_end_idx']]
                    cur_dict['num_steps'] = [len(x) for x in cur_dict['unique_val_idx']]

                    model_data_metadata[state] = cur_dict

                for state, metadata_dict in model_data_metadata.items():
                    if state == 0:
                        continue
                        
                    trial_idx_list = []
                    local_trial_idx_list = []
                    for trial_idx, indices in enumerate(metadata_dict['unique_val_idx']):
                        model_data_dict[f"{session_type}_{date}_{session}_T{trial_idx}_state{state}"] = create_closure_func(_window_spectrogram, sxx_partition_func, indices, shift, window_size, precision)
                        
                        trial_idx_list.append(global_trial_idx)
                        global_trial_idx += 1
                        
                        local_trial_idx_list.append(trial_idx)
                        
                    cur_partition_dict = model_data_filenames_dict.setdefault(sxx_partition_key, {})

                    cur_partition_dict[state] = {
                            'date': date,
                            'session_type': session_type,
                            'session': session,
                            'state': state,
                            'local_trials_idx_list': local_trial_idx_list,
                            'global_trials_idx_list': trial_idx_list
                    }
                        
    return model_data_dict, model_data_filenames_dict