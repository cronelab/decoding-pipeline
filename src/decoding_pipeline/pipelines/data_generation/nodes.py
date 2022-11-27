"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.18.3
"""

from scripts.convert_bci_to_hdf5 import convert_bcistream, convert_dat


def generate_center_out_hdf5_dataset(bcistreams, selected_sessions, patient_id):
    sessions_dict = selected_sessions[patient_id]['center_out']

    for partition_key, partition_load_func in bcistreams.items():
        
        continue_loop = True
        for paradigm_key, date_session_dict in sessions_dict.items():
            for date_key, sessions_list in date_session_dict.items():
                if date_key in partition_key and partition_key.split('_')[-1] in sessions_list:
                    continue_loop = False

        if continue_loop:
            continue
        
        partition_data = partition_load_func()

        filename = partition_data.filename.replace(".dat", ".hdf5")

        print(filename)

        convert_dat(partition_data, h5filename=filename, add_everything=True)

    return {}

def generate_calibration_hdf5_dataset(bcistreams, selected_sessions, patient_id):
    sessions_dict = selected_sessions[patient_id]['center_out']

    for partition_key, partition_load_func in bcistreams.items():
        continue_loop = True
        for paradigm_key, date_session_dict in sessions_dict.items():
            for date_key, sessions_list in date_session_dict.items():
                if date_key in partition_key:
                    continue_loop = False

        if continue_loop:
            continue

        partition_data = partition_load_func()

        filename = partition_data.filename.replace(".dat", ".hdf5")

        print(filename)

        convert_dat(partition_data, h5filename=filename, add_everything=True)

    return {}
