from kedro.io.core import AbstractDataSet, DataSetError
import numpy as np

class NPDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _save(self, np_array):
        np.save(self._filepath, np_array)

    def _load(self):
        return np.load(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)