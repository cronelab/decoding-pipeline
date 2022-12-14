"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from decoding_pipeline.pipelines import data_generation as dg
from decoding_pipeline.pipelines import data_processing as dp
from decoding_pipeline.pipelines import feature_generation as fg
from decoding_pipeline.pipelines import plot_generation as pg
from decoding_pipeline.pipelines import model_feature_generation as mfg


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_generation_pipeline = dg.create_pipeline()
    data_processing_pipeline = dp.create_pipeline()
    feature_generation_pipeline = fg.create_pipeline()
    plot_generation_pipeline = pg.create_pipeline()
    model_feature_generation_pipeline = mfg.create_pipeline()

    return {
        "__default__": data_generation_pipeline + data_processing_pipeline + feature_generation_pipeline + plot_generation_pipeline + model_feature_generation_pipeline,
        "dg": data_generation_pipeline,
        "dp": data_processing_pipeline,
        "fg": feature_generation_pipeline,
        "pg": plot_generation_pipeline,
        "mfg": model_feature_generation_pipeline
    }
