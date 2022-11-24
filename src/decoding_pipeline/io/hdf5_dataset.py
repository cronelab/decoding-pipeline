import sys
sys.path.insert(0, "../../scripts")

from scripts.h5eeg import H5EEGFile

from kedro.io.core import AbstractDataSet, DataSetError

class HDF5Dataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _save(self, _):
        pass

    def _load(self):
        return H5EEGFile(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)