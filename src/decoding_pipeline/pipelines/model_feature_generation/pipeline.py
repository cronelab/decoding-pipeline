"""
This is a boilerplate pipeline 'model_feature_generation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_model_windowed_sxx_data


def create_pipeline(**kwargs) -> Pipeline:
    model_feature_generation_pipeline = pipeline([
        node(
            func=generate_model_windowed_sxx_data,
            inputs=["center_out_spectrogram_std_pkl", "center_out_curated_states_pkl", "params:sessions", "params:model_data_params", "params:current_experiment", "params:patient_id"],
            outputs=["center_out_model_spectrogram_std_pkl", "center_out_model_spectrogram_indices"],
            name="generate_model_windowed_sxx_data_node"
        ),
    ])

    return model_feature_generation_pipeline
