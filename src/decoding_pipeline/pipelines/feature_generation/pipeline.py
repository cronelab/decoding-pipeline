"""
This is a boilerplate pipeline 'feature_generation'
generated using Kedro 0.18.3
"""

from .nodes import car_filter, generate_spectrogram, plot_spectrogram_transform_figure

from kedro.pipeline import Pipeline, node
from kedro.pipeline.modular_pipeline import pipeline


def create_pipeline(**kwargs) -> Pipeline:
    data_preprocessing_pipeline = pipeline([
        node(
            func=car_filter,
            inputs=["selected_channels", "params:car_filtering","params:patient_id", "center_out_extracted_pkl"],
            outputs='car_signals_dict',
            name='car_filter_node'
        ),
        node(
            func=generate_spectrogram,
            inputs=["car_signals_dict","center_out_extracted_pkl","params:spectrogram_params"],
            outputs="center_out_spectrogram_pkl",
            name="generate_spectrogram_node"
        ),
        node(
            func=plot_spectrogram_transform_figure,
            inputs="center_out_spectrogram_pkl",
            outputs="spectrogram_transform_plot",
            name="generate_spectrogram_transform_node"
        )
    ])

    return data_preprocessing_pipeline
