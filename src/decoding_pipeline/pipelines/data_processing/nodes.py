"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from typing import Dict

import numpy as np


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

    match patient_id:
        case 'CC01':
            split_index = grid_split[patient_id]['split_indices'][0]
            split_channels = ['A' + ch.removeprefix('chan') if int(ch.removeprefix('chan')) < split_index else 'B' + ch.removeprefix('chan') for ch in ch_include]

            # TODO: In the future, change this to regex to match the numbers
            ch_suffixes_unique = np.unique([ch[0] for ch in split_channels])

            ch_suffix_order = {}
            for ch_suffix in ch_suffixes_unique:
                ch_suffix_order[ch_suffix] = [ch for ch in split_channels if ch_suffix in ch]
            
            return {
                'ch_suffixes_unique': list(ch_suffixes_unique),
                'ch_suffix_order': ch_suffix_order 
            }