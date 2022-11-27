"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.18.3
"""

from scripts.convert_bci_to_hdf5 import convert_bcistream, convert_dat


def generate_hdf5_dataset(bcistreams, selected_sessions, patient_id, current_experiment, current_run_type):
    sessions_dict = selected_sessions[patient_id][current_experiment]
    calibration_dict = selected_sessions[patient_id]['calibration']

    for partition_key, partition_load_func in bcistreams.items():
        continue_loop = True
        for paradigm_key, date_session_dict in sessions_dict.items():
            for date_key, sessions_list in date_session_dict.items():
                if current_run_type == 'calibration':
                    if date_key in partition_key:
                        calibration_sessions_list = calibration_dict.get(date_key, [])
                        if len(calibration_sessions_list):
                            if partition_key.split('_')[-1] in calibration_sessions_list:
                                continue_loop = False
                        else:
                            continue_loop = False
                else:
                    if date_key in partition_key and partition_key.split('_')[-1] in sessions_list:
                        continue_loop = False

        if continue_loop:
            continue
        
        partition_data = partition_load_func()

        filename = partition_data.filename.replace(".dat", ".hdf5")

        print(filename)

        convert_dat(partition_data, h5filename=filename, add_everything=True)

    return {}
