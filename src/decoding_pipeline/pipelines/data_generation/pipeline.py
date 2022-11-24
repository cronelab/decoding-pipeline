"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.18.3
"""

from .nodes import generate_hdf5_dataset

from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_hdf5_dataset,
            inputs="center_out_dat",
            outputs="center_out_hdf5",
            name="convert_center_out_dat_to_hdf5_node",
        ),
    ])
