"""
This is a boilerplate pipeline 'plot_generation'
generated using Kedro 0.18.3
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_center_out_trajectories(partitioned_states, partitioned_data, bci_states, patient_id):
    colors = ['#6082B6', '#A9A9A9', '#808080', '#36454F', '#7393B3', '#818589', '#E5E4E2', '#8A9A5B', '#71797E', '#848884', '#708090', '#C0C0C0']

    states_list = np.array(bci_states[patient_id]['center_out'])

    save_dict = {}
    for partition_key, partition_func in partitioned_data.items():
        state_dict = partitioned_states[partition_key]()
        
        data_dict = partition_func()

        stimuli = data_dict['stimuli']

        x_cursor_state = stimuli[:, np.where(np.array(states_list) == 'cursorX')[0]].flatten()
        y_cursor_state = stimuli[:, np.where(np.array(states_list) == 'cursorY')[0]].flatten()

        x_cursor_state = x_cursor_state - x_cursor_state[x_cursor_state > 0][0]
        y_cursor_state = -(y_cursor_state - y_cursor_state[y_cursor_state > 0][0])

        fig = plt.figure(figsize=(10, 10))

        for idx, state in enumerate(state_dict.keys()):
            if state != 0:
                for indices in state_dict[state]['unique_val_idx']:
                    x_data = x_cursor_state[indices]
                    y_data = y_cursor_state[indices]

                    plt.plot(x_data, y_data, '-', color=colors[idx])
                    
        plt.title('Center Out Session Trajectories')
                    
        plt.xticks([])
        plt.yticks([])

        save_dict[f"{partition_key}.png"] = fig

        plt.close()

    return save_dict

def generate_mean_sxx_dict(spectrogram_dict, curated_states_partition, sessions, patient_id, spectrogram_params, plot_sxx_params, current_experiment):

    shift = spectrogram_params['shift']
    pre_stimulus_time = plot_sxx_params['pre_stimulus_time']
    post_stimulus_time = plot_sxx_params['post_stimulus_time']

    mean_dict_total = {}
    for session_type, session_data in sessions[patient_id][current_experiment].items():
        total_length_dict = {}
        for curated_states_key, curated_states_data_func in curated_states_partition.items():
            date = curated_states_key.split('_')[-2]
            session = curated_states_key.split('_')[-1]

            if date in session_data.keys() and session in session_data[date]:
                curated_states_dict = curated_states_data_func()

                for state, state_information in curated_states_dict.items():
                    cur_count = total_length_dict.setdefault(state, 0)
                    cur_count += len(state_information['start_end_idx'])
                    total_length_dict[state] = cur_count

        mean_dicts = {}
        for sxx_partition_key, sxx_partition_func in spectrogram_dict.items():
            date = sxx_partition_key.split('_')[-2]
            session = sxx_partition_key.split('_')[-1]

            if date in session_data.keys() and session in session_data[date]:
                curated_states_data_func = curated_states_partition[sxx_partition_key]
                curated_states_dict = curated_states_data_func()
                sxx_data_dict = sxx_partition_func()

                sampling_rate = sxx_data_dict['sampling_rate']

                samples_pre = int(np.ceil((pre_stimulus_time * sampling_rate)/shift))
                samples_post = int(np.ceil((post_stimulus_time * sampling_rate)/shift))
                t_sample = np.arange(-(samples_pre * shift)/sampling_rate, ((samples_post + 1) * shift)/sampling_rate, shift/sampling_rate)
                f_sxx = sxx_data_dict['f']

                avg_sxx_states_dict = {}
                for state, state_information in curated_states_dict.items():
                    cur_dict = {}

                    start_end_idx = state_information['start_end_idx']
                    cur_dict['start_end_idx'] = [(x[0]-samples_pre, x[0]+samples_post) for x in start_end_idx]
                    cur_dict['unique_val_idx'] = [np.arange(x[0], x[1]+1) for x in cur_dict['start_end_idx']]
                    cur_dict['num_steps'] = [len(x) for x in cur_dict['unique_val_idx']]

                    avg_sxx_states_dict[state] = cur_dict
                    avg_sxx_states_dict[state]['length'] = len(state_information['num_steps'])


                sxx = sxx_data_dict['sxx']

                for state, recalculated_state_information in avg_sxx_states_dict.items():
                    cur_mean = mean_dicts.setdefault(state, {'mean_sxx': 0, 't': t_sample, 'f': f_sxx})

                    mean_sxx_direction = np.mean(np.array(([sxx[:, idx_array, :] for idx_array in avg_sxx_states_dict[state]['unique_val_idx']])), axis = 0)
                    mean_sxx_direction *= (avg_sxx_states_dict[state]['length']/total_length_dict[state])

                    cur_mean['mean_sxx'] += mean_sxx_direction

                    mean_dicts[state] = cur_mean

        mean_dict_total[session_type] = mean_dicts
    return mean_dict_total

def _plot_average_sxx(current_experiment, task_paradigm, state, sxx_data, ch_suffix_key, ch_suffix_order, plot_sxx_params, grid_layout):
    cur_mean_sxx = sxx_data['mean_sxx']
    t_samples = sxx_data['t']
    f_samples = sxx_data['f']
    
    fig_width = plot_sxx_params['fig_width']
    fig_height = plot_sxx_params['fig_height']
    sharex = plot_sxx_params['sharex']
    sharey = plot_sxx_params['sharey']
    vmin = plot_sxx_params['vmin']
    vmax = plot_sxx_params['vmax']
    
    sel_grid = None
    sel_suffix_key = None
    for cur_grid in grid_layout:
        if np.array(cur_grid).flatten()[0][0] == ch_suffix_key:
            sel_suffix_key = ch_suffix_key
            sel_grid = np.array(cur_grid)
            break

    sel_ch_suffix_order = ch_suffix_order[sel_suffix_key]

    grid_rows = sel_grid.shape[0]
    grid_cols = sel_grid.shape[1]
    fig, axs  = plt.subplots(grid_rows, grid_cols, figsize = (fig_width, fig_height), sharex=sharex, sharey=sharey, constrained_layout=True);

    ch_ind = 0
    for ch in sel_ch_suffix_order:
        row = np.where(sel_grid == ch)[0][0]
        col = np.where(sel_grid == ch)[1][0]

        if grid_rows == 1:
            im = axs[col].pcolormesh(t_samples, f_samples, cur_mean_sxx[:,:,ch_ind], cmap = 'seismic', vmin = vmin, vmax = vmax);
            axs[col].set_title(ch)
            axs[col].axvline(x=0, color = 'k')

        elif grid_cols == 1:
            im = axs[row].pcolormesh(t_samples, f_samples, cur_mean_sxx[:,:,ch_ind], cmap = 'seismic', vmin = vmin, vmax = vmax);
            axs[row].set_title(ch)
            axs[row].axvline(x=0, color = 'k')

        else:
            im = axs[row,col].pcolormesh(t_samples, f_samples, cur_mean_sxx[:,:,ch_ind], cmap = 'seismic', vmin= vmin, vmax= vmax);
            axs[row,col].set_title(ch)
            axs[row,col].axvline(x=0, color = 'k')

        # Updating the channel index.
        ch_ind += 1

    fig.suptitle(f"{task_paradigm} {current_experiment} state {state}", fontsize=16)
    
    return fig

def plot_average_sxx(prefixed_channels, mean_dict_total, current_experiment, plot_sxx_params, grid_layout):
    ch_suffix_order = prefixed_channels['ch_suffix_order']
    
    save_dict = {}
    for task_paradigm, state_data in mean_dict_total.items():
        for state, sxx_data in state_data.items():
            for ch_suffix_key in ch_suffix_order.keys():
                fig = _plot_average_sxx(current_experiment, task_paradigm, state, sxx_data, ch_suffix_key, ch_suffix_order, plot_sxx_params, grid_layout)
                save_dict[f"{current_experiment}_{task_paradigm}_{ch_suffix_key}_state_{state}.png"] = fig
                plt.close
                
    return save_dict