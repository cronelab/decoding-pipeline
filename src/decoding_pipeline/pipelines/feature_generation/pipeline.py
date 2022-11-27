"""
This is a boilerplate pipeline 'feature_generation'
generated using Kedro 0.18.3
"""

from .nodes import car_filter, generate_spectrogram, plot_spectrogram_transform_figure, align_and_warp_spectrogram, downsample_data_to_spectrogram, plot_downsampled_signals

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
            func=car_filter,
            inputs=["selected_channels", "params:car_filtering","params:patient_id", "calibration_extracted_pkl"],
            outputs='calibration_car_signals_dict',
            name='calibration_car_filter_node'
        ),
        node(
            func=generate_spectrogram,
            inputs=["car_signals_dict","center_out_extracted_pkl","params:spectrogram_params"],
            outputs="center_out_spectrogram_pkl",
            name="generate_spectrogram_node"
        ),
        node(
            func=generate_spectrogram,
            inputs=["calibration_car_signals_dict","calibration_extracted_pkl","params:spectrogram_params"],
            outputs="calibration_spectrogram_pkl",
            name="generate_calibration_spectrogram_node"
        ),
        node(
            func=plot_spectrogram_transform_figure,
            inputs="center_out_spectrogram_pkl",
            outputs="spectrogram_transform_plot",
            name="plot_spectrogram_transform_node"
        ),
        node(
            func=align_and_warp_spectrogram,
            inputs=["center_out_spectrogram_pkl", "params:spectrogram_params"],
            outputs="center_out_spectrogram_warped_pkl",
            name="align_and_warp_spectrogram_node"
        ),
        node(
            func=align_and_warp_spectrogram,
            inputs=["calibration_spectrogram_pkl", "params:spectrogram_params"],
            outputs="calibration_spectrogram_warped_pkl",
            name="align_and_warp_calibration_spectrogram_node"
        ),
        node(
            func=downsample_data_to_spectrogram,
            inputs=["center_out_spectrogram_warped_pkl", "car_signals_dict"],
            outputs="center_out_downsampled_pkl",
            name="downsample_data_to_spectrogram_node"
        ),
        node(
            func=plot_downsampled_signals,
            inputs=["center_out_spectrogram_warped_pkl", "center_out_downsampled_pkl"],
            outputs="downsampled_data_plots",
            name="plot_downsampled_signals_node"
        )
    ])

    return data_preprocessing_pipeline
