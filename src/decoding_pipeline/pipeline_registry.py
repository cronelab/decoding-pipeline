"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from decoding_pipeline.pipelines import data_generation as dg
from decoding_pipeline.pipelines import data_processing as dp


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_generation_pipeline = dg.create_pipeline()
    data_processing_pipeline = dp.create_pipeline()

    return {
        "__default__": data_generation_pipeline + data_processing_pipeline,
        "dg": data_generation_pipeline,
        "dp": data_processing_pipeline,
    }
