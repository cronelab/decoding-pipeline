"""
This is a boilerplate pipeline 'model_feature_generation'
generated using Kedro 0.18.3
"""
from scripts.utility_scripts import create_closure, create_closure_func

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from custom_pytorch_datasets import CorticomExperimentDataset

def _generate_trial_sxx_data(data_func, indices):
    sxx_data_dict = data_func()
    
    sxx = sxx_data_dict['sxx']
    
    return sxx[:, indices, :].astype(np.float16)

# def _window_spectrogram(data_func, indices, shift, window_size):
#     sxx_data_dict = data_func()
    
#     sxx = sxx_data_dict['sxx']
    
#     return np.moveaxis(sliding_window_view(sxx[:, indices, :][:, ::shift, :], window_shape=window_size, axis=1), [0, -1], [1, -2])
    
def _window_spectrogram(sxx, shift, window_size):
    return np.moveaxis(sliding_window_view(sxx, window_shape=window_size, axis=1), [0, -1], [1, -2])[::shift, ...]

def generate_model_sxx_data(spectrogram_dict, curated_states_partition, sessions, model_data_params, current_experiment, patient_id):
    pre_stimulus_time = model_data_params['pre_stimulus_time']
    post_completion_time = model_data_params['post_completion_time']

    window_size = model_data_params['window_size']
    shift = model_data_params['shift']

    model_data_dict = {}
    model_data_filenames_dict = {}
    global_trial_idx = 0
    model_data_filenames = []
    for session_type, session_data in sessions[patient_id][current_experiment].items():
        for sxx_partition_key, sxx_partition_func in spectrogram_dict.items():
            date = sxx_partition_key.split('_')[-2]
            session = sxx_partition_key.split('_')[-1]

            if date in session_data.keys() and session in session_data[date]:
                curated_states_data_func = curated_states_partition[sxx_partition_key]
                curated_states_dict = curated_states_data_func()
                # sxx_data_dict = sxx_partition_func()

                # sampling_rate = sxx_data_dict['sampling_rate']
#                 sampling_rate = 1000

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

                # sxx = sxx_data_dict['sxx']

                intermed_model_dict = {}
                for state, metadata_dict in model_data_metadata.items():
                    if state == 0:
                        continue
                        
                    trial_idx_list = []
                    local_trial_idx_list = []
                    for trial_idx, indices in enumerate(metadata_dict['unique_val_idx']):
                        model_data_dict[f"{session_type}_{date}_{session}_T{trial_idx}_state{state}"] = create_closure_func(_generate_trial_sxx_data, sxx_partition_func, indices)
                        model_data_filenames.append(f"{session_type}_{date}_{session}_T{trial_idx}_state{state}")

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
                        
    return model_data_dict, model_data_filenames_dict, model_data_filenames

def _get_trial_filenames(model_data_filenames_dict):
    trial_filenames_list = []
    for partition_key, partition_data_func in model_data_filenames_dict.items():
        model_data_dict = partition_data_func #
        
        for state, trial_information_dict in model_data_dict.items():
            session_type = trial_information_dict['session_type']
            date = trial_information_dict['date']
            local_trials_idx_list = trial_information_dict['local_trials_idx_list']
            session = trial_information_dict['session']
            
            trial_filenames_list += [f"{session_type}_{date}_{session}_T{trial_idx}_state{state}" for trial_idx in local_trials_idx_list]
    
    return trial_filenames_list

def _generate_train_test_split_indices(model_data_filenames_dict, model_data_filenames, model_data_params, current_experiment):
    data_split_type = model_data_params[current_experiment]['sel_split_type']

    split_type_params = model_data_params[current_experiment]['split_types'][data_split_type]

    leave_out = split_type_params['leave_out']
    randomized = split_type_params['randomized']
    random_seed = split_type_params['random_seed']
    sel_session_type = model_data_params[current_experiment]['sel_session_type']

    np.random.seed(random_seed)

    if data_split_type == 'leave_day_out':
        dates_list = []
        for partition_key, partition_data_func in model_data_filenames_dict.items():
            date = partition_key.split('_')[-2]
            session = partition_key.split('_')[-1]
            
            data_dict = partition_data_func()
            
            for state, trial_information_dict in data_dict.items():
                if sel_session_type != trial_information_dict['session_type']:
                    break
                
                if date not in dates_list:
                    dates_list.append(date)
                    break
        
        dates_list = np.array(list(set(dates_list)))

        dates_list_permuted = dates_list
        if randomized:
            dates_list_permuted = np.random.permutation(dates_list)

        test_list_dates = dates_list_permuted[:leave_out]
        train_list_dates = dates_list_permuted[leave_out:]
        
        train_list = []
        test_list = []
        train_labels_list = []
        test_labels_list = []
        for partition_key, partition_data_func in model_data_filenames_dict.items():
            date = partition_key.split('_')[-2]
            session = partition_key.split('_')[-1]
            
            data_dict = partition_data_func()
            
            for state, trial_information_dict in data_dict.items():
                session_type = trial_information_dict['session_type']
                
                if sel_session_type != session_type:
                    continue
                
                if date in train_list_dates:
                    train_list += trial_information_dict['global_trials_idx_list']
                    train_labels_list += [state]*len(trial_information_dict['global_trials_idx_list'])
                    
                elif date in test_list_dates:
                    test_list += trial_information_dict['global_trials_idx_list']
                    test_labels_list += [state]*len(trial_information_dict['global_trials_idx_list'])

    elif data_split_type == 'leave_session_out':
        dates_and_sessions_list = []
        for partition_key, partition_data_func in model_data_filenames_dict.items():
            date = partition_key.split('_')[-2]
            session = partition_key.split('_')[-1]
            
            data_dict = partition_data_func()
            
            for state, trial_information_dict in data_dict.items():
                if sel_session_type != trial_information_dict['session_type']:
                    break
                
                if {'date': date, 'session': session} not in dates_and_sessions_list:
                    dates_and_sessions_list.append({'date': date, 'session': session})
                    break

        dates_and_sessions_list = np.array(dates_and_sessions_list)

        dates_and_sessions_list_permuted = dates_and_sessions_list
        if randomized:
            dates_and_sessions_list_permuted = np.random.permutation(dates_and_sessions_list)
        
        test_list_dates_and_sessions = dates_and_sessions_list_permuted[:leave_out]
        train_list_dates_and_sessions = dates_and_sessions_list_permuted[leave_out:]
        
        train_list = []
        test_list = []
        train_labels_list = []
        test_labels_list = []
        for partition_key, partition_data_func in model_data_filenames_dict.items():
            date = partition_key.split('_')[-2]
            session = partition_key.split('_')[-1]
            
            data_dict = partition_data_func ()
            for state, trial_information_dict in data_dict.items():
                if {'date': date, 'session': session} in train_list_dates_and_sessions:
                    train_list += trial_information_dict['global_trials_idx_list']
                    train_labels_list += [state]*len(trial_information_dict['global_trials_idx_list'])
                    
                elif {'date': date, 'session': session} in test_list_dates_and_sessions:
                    test_list += trial_information_dict['global_trials_idx_list']
                    test_labels_list += [state]*len(trial_information_dict['global_trials_idx_list'])

    elif data_split_type == 'leave_trial_out':
        trials_list = []
        labels_list= []
        for partition_key, partition_data_func in model_data_filenames_dict.items():
            date = partition_key.split('_')[-2]
            session = partition_key.split('_')[-1]
            
            data_dict = partition_data_func()
            
            for state, trial_information_dict in data_dict.items():
                if sel_session_type != trial_information_dict['session_type']:
                    break
                    
                trials_list += trial_information_dict['global_trials_idx_list']
                labels_list += [state]*len(trial_information_dict['global_trials_idx_list'])

        trials_list = np.array(trials_list)
        labels_list = np.array(labels_list)

        permutation_indices = np.arange(len(labels_list))
        if randomized:
            permutation_indices = np.random.permutation(permutation_indices)

        trials_list_permuted = trials_list[permutation_indices]
        trials_labels_list_permuted = labels_list[permutation_indices]
        
        test_list = trials_list_permuted[:leave_out]
        train_list = trials_list_permuted[leave_out:]

        test_labels_list = trials_labels_list_permuted[:leave_out]
        train_labels_list = trials_labels_list_permuted[leave_out:]

    
    return {
        'train_list_idx': np.array(train_list),
        'test_list_idx': np.array(test_list),
        'train_labels_list': np.array(train_labels_list),
        'test_labels_list': np.array(test_labels_list),
        'train_filenames': model_data_filenames[np.array(train_list)],
        'test_filenames': model_data_filenames[np.array(test_list)]
    }

def experiment_dataloader_collate(data):
        sxx_list = []
        labels_list = []
        
        for sxx, label in data:
            windowed_sxx = _window_spectrogram(sxx, shift, window_size)
            
            print(windowed_sxx.shape)
            
            sxx_list.append(windowed_sxx)
            labels_list.append(label)
        
        windowed_sxx = np.concatenate(sxx_list)
        
        return windowed_sxx, labels_list

def generate_datasets(trial_loading_dict, train_test_split_indices):
    train_labels_list = train_test_split_indices['train_labels_list'] 
    test_labels_list = train_test_split_indices['test_labels_list']

    train_filenames = train_test_split_indices['train_filenames']
    test_filenames = train_test_split_indices['test_filenames']

    training_dataset = CorticomExperimentDataset(train_filenames, train_labels_list, trial_loading_dict)
    testing_dataset = CorticomExperimentDataset(test_filenames, test_labels_list, trial_loading_dict)

    return {
        "training_dataset": training_dataset,
        "testing_dataset": testing_dataset
    }