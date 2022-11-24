# import sys
# import os
# sys.path.insert(0, os.getcwd() + "/../../scripts")

# # print(os.getcwd() + "/../../scripts")

import scripts.filereader
from scripts.filereader import bcistream

from kedro.io.core import AbstractDataSet, DataSetError

class DATDataset(AbstractDataSet):
    def __init__(self, filepath):
        self._filepath = filepath

    def _save(self, _):
        raise DataSetError("Read Only DataSet")

    def _load(self):
        return bcistream(self._filepath)

    def _describe(self):
        return dict(filepath=self._filepath)


