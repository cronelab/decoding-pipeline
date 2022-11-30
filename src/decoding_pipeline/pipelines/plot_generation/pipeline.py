"""
This is a boilerplate pipeline 'plot_generation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import plot_center_out_trajectories, generate_mean_sxx_dict, plot_average_sxx


def create_pipeline(**kwargs) -> Pipeline:
    plot_generation_pipeline = pipeline([
        node(
            func=plot_center_out_trajectories,
            inputs=["center_out_curated_states_pkl", "center_out_downsampled_pkl", "params:bci_states", "params:patient_id"],
            outputs="trajectory_plots",
            name="plot_center_out_trajectories_node"
        ),
        node(
            func=generate_mean_sxx_dict,
            inputs=["center_out_spectrogram_std_pkl", "center_out_curated_states_pkl", "params:sessions", "params:patient_id", "params:spectrogram_params", "params:plot_sxx_params", "params:current_experiment"],
            outputs="center_out_mean_spectrogram_dict_pkl",
            name="generate_mean_spectrogram_dict_node"
        ),
        node(
            func=plot_average_sxx,
            inputs=["prefixed_channels", "center_out_mean_spectrogram_dict_pkl", "params:current_experiment", "params:plot_sxx_params", "params:grid_layout"],
            outputs="average_spectrogram_plots",
            name="plot_average_spectrogram_node"
        )
    ])
    return plot_generation_pipeline
