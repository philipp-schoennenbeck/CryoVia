import pickle





class CustomUnpickler(pickle.Unpickler):


    def find_class(self, module, name):
        
        if name == 'Analyser':
            from cryovia.cryovia_analysis.analyser import Analyser
            return Analyser
        if name == 'Membrane':
            from cryovia.cryovia_analysis.membrane import Membrane
            return Membrane
        if name == 'Point':
            from cryovia.cryovia_analysis.point import Point
            return Point
        if name == 'Dataset':
            from cryovia.gui.dataset import Dataset
            return Dataset
        if name == 'ShapeClassifier':
            from cryovia.gui.shape_classifier import ShapeClassifier
            return ShapeClassifier
        if name == 'segmentationModel':
            from cryovia.gui.segmentation_files.segmentation_model import segmentationModel, Config
            return segmentationModel
        if name == 'Config':
            from cryovia.gui.segmentation_files.segmentation_model import segmentationModel, Config
            return Config
         
        return super().find_class(module, name)
