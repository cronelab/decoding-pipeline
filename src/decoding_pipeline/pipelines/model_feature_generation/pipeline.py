"""
This is a boilerplate pipeline 'model_feature_generation'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline

from .nodes import generate_model_sxx_data, _generate_train_test_split_indices


def create_pipeline(**kwargs) -> Pipeline:
    model_feature_generation_pipeline = pipeline([
        node(
            func=generate_model_sxx_data,
            inputs=["center_out_spectrogram_std_pkl", "center_out_curated_states_pkl", "params:sessions", "params:model_data_params", "params:current_experiment", "params:patient_id"],
            outputs=["center_out_model_spectrogram_std_pkl", "center_out_model_spectrogram_indices", "center_out_model_filenames"],
            name="generate_model_windowed_sxx_data_node"
        ),
        node(
            func=_generate_train_test_split_indices,
            inputs=["center_out_model_spectrogram_indices", "center_out_model_filenames", "params:model_data_params", "params:current_experiment"],
            outputs="center_out_train_test_indices",
            name="generate_train_test_split_node"
        ),
        
    ])

    return model_feature_generation_pipeline
