from pathlib import Path
from typing import Any
import os



class PathContainer:
    def __init__(self, path):
        self.path = path

    def reAssign(self, path):
        self.path = path
    
    def __getattribute__(self, name: str) -> Any:
        if name == "path":
            return super().__getattribute__('path')
        elif name == "reAssign":
            return super().__getattribute__('reAssign')
        else:
            return getattr(super().__getattribute__('path'), name)
    def __truediv__(self, other):
        return self.path / other
    
    def __fspath__(self):
        return str(self.path)


CRYOVIA_PATH = PathContainer(Path().home() / ".cryovia")
SEGMENTATION_MODEL_DIR = PathContainer(Path().home() / ".cryovia" / "SegmentationModels")
CLASSIFIER_PATH = PathContainer(Path().home() / ".cryovia" / "Classifiers")
SHAPE_CURVATURE_PATH = PathContainer(Path().home() / ".cryovia" / "Shape_curvatures")
cryovia_TEMP_DIR = PathContainer(Path().home() / ".cryovia" / "temp")
DATASET_PATH = PathContainer(Path().home() / ".cryovia" / "DATASETS")

if os.environ.get("CRYOVIA_MODE") is None:
    os.environ["CRYOVIA_MODE"] = "0"
