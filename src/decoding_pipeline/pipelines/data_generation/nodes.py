"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.18.3
"""

from scripts.convert_bci_to_hdf5 import convert_bcistream, convert_dat


def generate_hdf5_dataset(bcistreams):
    for partition_key, partition_load_func in bcistreams.items():
        partition_data = partition_load_func()
        filename = partition_data.filename.replace(".dat", ".hdf5")

        print(filename)

        convert_dat(partition_data, h5filename=filename, add_everything=True)

    return {}

