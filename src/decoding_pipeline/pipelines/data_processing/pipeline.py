"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from .nodes import import_single_electrode_information, extract_single_channel_labels, elim_single_channel_labels, prefix_single_channel_info

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=import_single_electrode_information,
            inputs="center_out_hdf5",
            outputs="electrode_labels",
            name="import_electrode_information_node",
        ),
        node(
            func=extract_single_channel_labels,
            inputs=["params:channels", "params:patient_id", "electrode_labels", "params:channel_selection"],
            outputs="selected_channels_unfiltered",
            name="selected_channels_unfiltered_node"
        ),
        node(
            func=elim_single_channel_labels,
            inputs=["selected_channels_unfiltered", "params:patient_id", "params:bad_channels", "params:elim_channels"],
            outputs="selected_channels",
            name="selected_channels_node"
        ),
        node(
            func=prefix_single_channel_info,
            inputs=["selected_channels", "params:patient_id", "params:grid_split"],
            outputs="prefixed_channels",
            name="prefix_channels_node"
        )
    ],
    namespace="channel_labelling",
    inputs="center_out_hdf5",
    outputs="prefixed_channels",
    parameters={"params:patient_id": "params:patient_id"})