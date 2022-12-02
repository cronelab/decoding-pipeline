"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from typing import Dict

import numpy as np
import matplotlib.pyplot as plt

from scripts.utility_scripts import create_closure

def remove_prefix(word, prefix):
    if word.startswith(prefix):
        word_new = word[len(prefix):]
    return word_new

def _import_electrode_information(h5file):
    eeg = h5file.group.eeg()
    aux = h5file.group.aux()
    eeglabels = eeg.get_labels(); 
    auxlabels = aux.get_labels()
    
    return {
        'auxlabels': auxlabels,
        'eeglabels': eeglabels
    }

def import_single_electrode_information(data: Dict) -> Dict:
    h5file = list(data.values())[0]()
    labels = _import_electrode_information(h5file)
    return labels


def _extract_channel_labels(channels, eeglabels, channel_selection):
    ch_motor = channels['motor']
    ch_sensory = channels['sensory']

    overlapped_chs = set(ch_motor) & set(ch_sensory)
    assert not overlapped_chs, "Channels are overlapping"

    ch_motor   = [x for x in ch_motor if x in eeglabels]
    ch_sensory = [x for x in ch_sensory if x in eeglabels]

    ch_sensorimotor = ch_motor + ch_sensory
    ch_other = [x for x in eeglabels if x not in ch_sensorimotor]

    sel_channels = []
    if channel_selection == 'all':
        sel_channels = ch_sensorimotor + ch_other
    elif channel_selection == 'both':
        sel_channels = ch_sensorimotor
    elif channel_selection == 'motor':
        sel_channels = ch_motor
    elif channel_selection == 'sensory':
        sel_channels = ch_sensory

    return sel_channels

def extract_single_channel_labels(channels, patient_id, labels, channel_selection):
    patient_channels = channels[patient_id]
    eeglabels = labels['eeglabels']
    return _extract_channel_labels(patient_channels, eeglabels, channel_selection)

def _elim_channel_labels(channels, ch_bad, ch_elim):
    ch_exclude = ch_bad + ch_elim
    ch_include = [n for n in channels if n not in ch_exclude]

    return {
        'ch_include': ch_include,
        'ch_exclude': ch_exclude
    }

def elim_single_channel_labels(channels, patient_id, ch_bad_params, ch_elim_params):
    ch_bad = ch_bad_params[patient_id]['channels']
    ch_elim = ch_elim_params[patient_id]['channels']

    return _elim_channel_labels(channels, ch_bad, ch_elim)

def prefix_single_channel_info(channels, patient_id, grid_split):
    ch_include = channels['ch_include']

    if patient_id == 'CC01':
        split_index = grid_split[patient_id]['split_indices'][0]
        split_channels = ['A' + remove_prefix(ch, 'chan') if int(remove_prefix(ch, 'chan')) < split_index else 'B' + remove_prefix(ch, 'chan') for ch in ch_include]

        # TODO: In the future, change this to regex to match the numbers
        ch_suffixes_unique = np.unique([ch[0] for ch in split_channels])

        ch_suffix_order = {}
        for ch_suffix in ch_suffixes_unique:
            ch_suffix_order[ch_suffix] = [ch for ch in split_channels if ch_suffix in ch]
        
        return {
            'ch_suffixes_unique': list(ch_suffixes_unique),
            'ch_suffix_order': ch_suffix_order 
        }

def extract_bci_data(h5_data, selected_channels, electrode_labels, states, patient_id, gain, current_experiment):
    eeglabels = electrode_labels['eeglabels']
    auxlabels = electrode_labels['auxlabels']

    ch_include = selected_channels['ch_include']
    ch_exclude = selected_channels['ch_exclude']

    selected_states = states[patient_id][current_experiment]

    num_channels = len(eeglabels)

    neural_data = np.zeros((0, num_channels))
    stimuli = np.zeros((0,))

    save_dict = {}
    for partition_key, partition_load_func in h5_data.items():
        h5file = partition_load_func()
        eeg = h5file.group.eeg()
        eeg_data = eeg.dataset[:]

        aux = h5file.group.aux()

        print(selected_states)

        # Populating stimuli data array in the correct ordering
        stimuli_data = np.empty((len(aux.dataset[:]), len(selected_states)))
        for idx, state in enumerate(selected_states):
            print(f"{'state_printing':*<40}")
            print(state)

            stimuli_data[:, idx] = aux[:, [state]][:, 0]

        # stimuli_data = aux[:, selected_states]

        eeg_data = eeg_data*gain
        stimuli_data = stimuli_data.astype(int)

        n_channels_include = len(ch_include)

        signals_list = [None] * n_channels_include

        # Iterating across each name in the included channels list.
        for (ch_ind, ch_name) in enumerate(ch_include):
            eeg_ind = np.argwhere(eeglabels == ch_name)[0][0]
            signals_list[ch_ind] = eeg_data[:, eeg_ind]

        # Converting the signals_list to the proper array format.
        signals = np.squeeze(np.array(signals_list).transpose())
        
        # If the signals don't have enough dimensions (only one channel).
        if n_channels_include == 1:
            signals = np.expand_dims(signals,1)

        sampling_rate = int(eeg.get_rate())

        N_samples      = len(stimuli_data)
        t_samples      = np.arange(0,N_samples)
        t_seconds      = t_samples/eeg.get_rate()

        # print(signals)

        # TODO: Finish this to save appropriately
        save_dict[partition_key] = create_closure({
            'signals': signals,
            'stimuli': stimuli_data,
            'sampling_rate': sampling_rate,
            't_samples': t_samples,
            't_seconds': t_seconds
        })
    
    return save_dict

def plot_bci_states(partitioned_data, states, patient_id, current_experiment):
    state_names = states[patient_id][current_experiment]

    save_dict = {}
    for partition_key, partition_load_func in partitioned_data.items():
        data_dict = partition_load_func()

        stimuli = data_dict['stimuli']
        sampling_rate = data_dict['sampling_rate']

        num_stimuli = stimuli.shape[-1]

        N = stimuli.shape[0]
        n_samples = np.arange(0, N)
        t_samples = n_samples/sampling_rate

        fig, sub_axs = plt.subplots(num_stimuli, figsize=(20,10))

        for idx, ax in enumerate(sub_axs):
            subplot_data = stimuli[:, idx]
            ax.plot(t_samples, subplot_data)
            ax.set_ylabel('State')
            ax.set_title(state_names[idx])
            
        fig.subplots_adjust(hspace=0.4)

        save_dict[f"{partition_key}.png"] = fig

        plt.close()

    return save_dict







