"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.3
"""

from .nodes import import_single_electrode_information, extract_single_channel_labels, elim_single_channel_labels, prefix_single_channel_info, extract_bci_data, plot_bci_states

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

def create_pipeline(**kwargs) -> Pipeline:
    channel_labelling_pipeline = pipeline([
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
    outputs={"prefixed_channels": "prefixed_channels", "selected_channels": "selected_channels"},
    parameters={"params:patient_id": "params:patient_id"})

    data_extraction_pipeline = pipeline([
        node(
            func=import_single_electrode_information,
            inputs="center_out_hdf5",
            outputs="electrode_labels",
            name="import_electrode_information_node",
        ),
        node(
            func=extract_bci_data,
            inputs=["center_out_hdf5", "selected_channels", "electrode_labels", "params:bci_states", "params:patient_id", "params:gain"],
            outputs="center_out_extracted_pkl",
            name="extract_bci_data_node"
        ),
        node(
            func=extract_bci_data,
            inputs=["calibration_hdf5", "selected_channels", "electrode_labels", "params:bci_states", "params:patient_id", "params:gain"],
            outputs="calibration_extracted_pkl",
            name="extract_calibration_data_node"
        ),
    ],
    namespace="data_extraction",
    inputs=set(["calibration_hdf5", "center_out_hdf5", "selected_channels"]),
    outputs=set(["center_out_extracted_pkl", "calibration_extracted_pkl"]),
    parameters={"params:patient_id": "params:patient_id", "params:gain": "params:gain", "params:bci_states": "params:bci_states"})

    dataset_metrics_pipeline = pipeline([
        node(
            func=plot_bci_states,
            inputs=["center_out_extracted_pkl", "params:bci_states", "params:patient_id"],
            outputs="state_plots",
            name="plot_bci_states_node"
            
        )
    ],
    namespace="dataset_metrics",
    inputs=set(["center_out_extracted_pkl"]),
    outputs="state_plots",
    parameters={"params:patient_id": "params:patient_id", "params:bci_states": "params:bci_states"})

    # return channel_labelling_pipeline + data_extraction_pipeline

    return pipeline(
        pipe=channel_labelling_pipeline + data_extraction_pipeline + dataset_metrics_pipeline,
        namespace="data_preprocessing",
        inputs=set(["calibration_hdf5", "center_out_hdf5"]),
        outputs={"prefixed_channels": "prefixed_channels", "center_out_extracted_pkl": "center_out_extracted_pkl", "calibration_extracted_pkl": "calibration_extracted_pkl", "state_plots": "state_plots", "selected_channels": "selected_channels"},
        parameters={"params:patient_id": "params:patient_id", "params:gain": "params:gain", "params:bci_states": "params:bci_states"}
    )