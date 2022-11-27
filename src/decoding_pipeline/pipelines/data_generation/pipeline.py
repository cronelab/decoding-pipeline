"""
This is a boilerplate pipeline 'data_generation'
generated using Kedro 0.18.3
"""

from .nodes import generate_center_out_hdf5_dataset, generate_calibration_hdf5_dataset

from kedro.pipeline import Pipeline, node

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=generate_center_out_hdf5_dataset,
            inputs=["center_out_dat", "params:sessions", "params:patient_id"],
            outputs="center_out_hdf5",
            name="convert_center_out_dat_to_hdf5_node",
        ),
        node(
            func=generate_calibration_hdf5_dataset,
            inputs=["calibration_dat", "params:sessions", "params:patient_id"],
            outputs="calibration_hdf5",
            name="generate_calibration_dat_to_hdf5_node"
        )
    ])
