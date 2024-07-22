import sys, shutil, os 
from pathlib import Path
import silence_tensorflow.auto

from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QMainWindow, QToolButton, QMessageBox
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5 import QtCore

from traceback import format_exception

import click
import multiprocessing as mp

from cryovia.gui.path_variables import CRYOVIA_PATH, SEGMENTATION_MODEL_DIR, CRYOVIA_PATH, CLASSIFIER_PATH, SHAPE_CURVATURE_PATH, cryovia_TEMP_DIR, DATASET_PATH

# CRYOVIA_PATH = Path().home() / ".cryovia"


def copyShapeCurvatures():
    """
    Finds the installation path of CryoVia and copies the default curvature files to the CryoVia directory.
    Parameters
    ----------


    Returns
    -------
    
    """
    global CRYOVIA_PATH
    cryovia_install_dir = Path(cryovia.__file__).parent
    default_path = cryovia_install_dir /"default_models"/ "Shape_curvatures"
    if not default_path.exists():
        raise FileNotFoundError(default_path)
    for file in os.listdir(default_path):
        filename = default_path / file
        if filename.suffix == ".npy":
            shutil.copy(filename, CRYOVIA_PATH / "Shape_curvatures" / file)
            print(f"Copy shape curvature {file}")
    

def checkDefaultClassifier():
    """
    Checks if the default shape classifier already exists. If not, copy them from the installation path.
    Parameters
    ----------


    Returns
    -------
    
    """
    global CRYOVIA_PATH
    cryovia_install_dir = Path(cryovia.__file__).parent
    for c in ["Default_NN", "Default_GBC"]:
        if not (CRYOVIA_PATH / "Classifiers" / f"{c}_classifier.pickle").exists():
            default_path = cryovia_install_dir / "default_models" / f"{c}_classifier.pickle"
            if not default_path.exists():
                raise FileNotFoundError(default_path)
            shutil.copy(default_path, CRYOVIA_PATH / "Classifiers" )
            weight_path = cryovia_install_dir / "default_models" / f"{c}_weights.h5"
            if weight_path.exists():
                shutil.copy(weight_path, CRYOVIA_PATH /"Classifiers")
            print(f"Copy default classifier")

def checkDefaultSegmentationModel():
    """
    Checks if the default segmentation model already exists. If not, copy them from the installation path.
    Parameters
    ----------


    Returns
    -------
    
    """
    global CRYOVIA_PATH
    cryovia_install_dir = Path(cryovia.__file__).parent

    for modelname in ["Default", "Default_thin"]:

        if not (CRYOVIA_PATH / "SegmentationModels" / modelname).exists():
            default_path = cryovia_install_dir / "default_models" / modelname
            if not default_path.exists():
                raise FileNotFoundError(default_path)
            weights_path = default_path / "best_weights.h5"
            pickle_path = default_path / "Segmentator.pickle"
            if not weights_path.exists():
                raise FileNotFoundError(weights_path)
            if not pickle_path.exists():
                raise FileNotFoundError(pickle_path)
            shutil.copytree(default_path, CRYOVIA_PATH / "SegmentationModels" / modelname)
            print(f"Copy {modelname} segmentation model")




def copyDataFiles(copy_shapes=False):
    """
    Checks for default classifier and segmentation model.
    Parameters
    ----------


    Returns
    -------
    
    """
    global CRYOVIA_PATH
    checkDefaultClassifier()
    checkDefaultSegmentationModel()
    if copy_shapes:
        copyShapeCurvatures()
    
    

   


def checkFirstTime():
    """
    Creates the cryovia directories if they do not exist and then copies some files.
    Parameters
    ----------


    Returns
    -------
    
    """
    global CRYOVIA_PATH
    dirs = ["DATASETS", "Classifiers", "SegmentationModels",]
    for d in dirs:
        d = CRYOVIA_PATH / d
        if not d.exists():
            d.mkdir(parents=True)
            print(f"Creating {d}")
    d = CRYOVIA_PATH /  "Shape_curvatures"
    copy_shapes = False
    if not d.exists():
        d.mkdir(parents=True)
        print(f"Creating {d}")
        copy_shapes = True
    copyDataFiles(copy_shapes)










def create_temp_dir():
    global CRYOVIA_TEMP_DIR
    if CRYOVIA_TEMP_DIR.exists():
        return
    CRYOVIA_TEMP_DIR.mkdir(parents=True)

class CentralWidget(QWidget):
    """
    The widget from which to start the other windows.
    """
    def __init__(self, parent):
        super().__init__(parent)
        cryovia_install_dir = Path(cryovia.__file__).parent
        self.setLayout(QHBoxLayout())
        self.current_window = None
        analyser_pixmap = QPixmap(str(cryovia_install_dir / "gui" / "icons" / "cell.png")).scaledToHeight(100)
        analyser_icon = QIcon(analyser_pixmap)
        self.analyser_button = QToolButton(icon=analyser_icon,text="Membrane analyser")
        self.analyser_button.setIconSize(QSize(100,100))
        self.analyser_button.clicked.connect(self.open_analyser)
        self.analyser_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)

        train_cnn_pixmap = QPixmap(str(cryovia_install_dir / "gui" / "icons" / "cnn.png")).scaledToHeight(100)
        train_cnn_icon = QIcon(train_cnn_pixmap)
        self.train_cnn_button = QToolButton(icon=train_cnn_icon, text="Neural networks")
        self.train_cnn_button.setIconSize(QSize(100,100))
        self.train_cnn_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.train_cnn_button.clicked.connect(self.open_segmentator)

        # shape_icon = QIcon("icons/shapes.png")
        shape_pixmap = QPixmap(str(cryovia_install_dir / "gui" / "icons" / "shapes.png")).scaledToHeight(100)
        shape_icon = QIcon(shape_pixmap)
        self.train_shape_classifier_button = QToolButton(icon=shape_icon, text="Train classifier")
        self.train_shape_classifier_button.setIconSize(QSize(100,100))
        self.train_shape_classifier_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.train_shape_classifier_button.clicked.connect(self.open_drawer)


        edge_detector_pixmap = QPixmap(str(cryovia_install_dir / "gui" / "icons" / "edgeDetector.png")).scaledToHeight(100)
        edge_detector_icon = QIcon(edge_detector_pixmap)
        self.edge_detector_button = QToolButton(icon=edge_detector_icon, text="Edge detection")
        self.edge_detector_button.setIconSize(QSize(100,100))
        self.edge_detector_button.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.edge_detector_button.clicked.connect(self.open_edge_detector)
        self.layout().addWidget(self.analyser_button)
        self.layout().addWidget(self.train_cnn_button)
        self.layout().addWidget(self.train_shape_classifier_button)
        self.layout().addWidget(self.edge_detector_button)


    def open_drawer(self):
        """
        Opens the shape classifier window.
        """
        self.current_window = CreateNewShapesWindow(custom_parent=self)
        # self.current_window.show()
        self.open_window()

    def open_edge_detector(self):
        """
        Opens the edge detector window.
        """
        self.current_window = MainWindow(custom_parent=self)
        self.open_window()


    def open_segmentator(self):
        """
        Opens the neural network training window.
        """
        self.current_window = SegmentationWindow(custom_parent=self)
        self.open_window()

    def open_analyser(self):
        """
        Opens the analyser window.
        """
        self.current_window = DatasetGui(custom_parent=self)
        self.open_window()

    def open_window(self):
        """
        Opens the current selected window.
        """


        if self.current_window is not None:
            # self.change_button_clickability(False)
            self.setEnabled(False)
            self.current_window.show()


    def child_closed(self):
        """
        Enables the opening of other windows when one window is closed.
        """
        if self.current_window is not None:
            self.current_window = None
        self.setEnabled(True)
            # self.change_button_clickability(True)


    # def change_button_clickability(self, clickable):
    #     self.analyser_button.setEnabled(clickable)
    #     self.train_cnn_button.setEnabled(clickable)
    #     self.train_shape_classifier_button.setEnabled(clickable)


class QStartingMenu(QMainWindow):
    def __init__(self):
        super().__init__()
        
        
        self.button_widget = CentralWidget(self)


        self.setCentralWidget(self.button_widget)
        self.setWindowTitle("CRYO-VIA")
        

class StringListParamType(click.ParamType):
    name = "string_list"


    def convert(self, value, param, ctx):
        if isinstance(value, (list, tuple)):
            if all([isinstance(i, int) for i in value]):
                return value
            else:
                return [self.convert(i, param, ctx) for i in value]
        elif isinstance(value, int):
            return [value]
        try:
            value:str
            values = value.split(",")
            return [int(v) for v in values]
            
        except ValueError:
            self.fail(f"{value!r} is not a list of integers seperated by commas", param, ctx)



@click.command()
@click.option("-n", "--njobs", help="Number of njobs to load in and save files. For analysing you can specify other values in the GUI. Default is number of cores/2.",
               type=click.IntRange(1,mp.cpu_count(),clamp=True), default=max(1, mp.cpu_count() // 2))
@click.option("-g", "--gpus", help="List of GPUs to be available to use (as integers in cuda), seperated by comma without spaces. Default is all available GPUs.", type=StringListParamType())
@click.option("--debug", is_flag=True, help="Enable debug mode", hidden=True, default=False)
def startGui(njobs, gpus, debug):
    """
    Starts the CryoVia GUI.
    Parameters
    ----------
    njobs   : number of parallel threads for some internal usage.
    gpus    : list of GPUs to use.

    Returns
    -------
    
    """
    if debug:
        global CRYOVIA_PATH, SEGMENTATION_MODEL_DIR, CRYOVIA_PATH, CLASSIFIER_PATH, SHAPE_CURVATURE_PATH, cryovia_TEMP_DIR, DATASET_PATH
        CRYOVIA_PATH.reAssign(CRYOVIA_PATH.parent / ".cryovia_debug")
        SEGMENTATION_MODEL_DIR .reAssign(CRYOVIA_PATH / "SegmentationModels")
        CLASSIFIER_PATH.reAssign(CRYOVIA_PATH / "Classifiers")
        SHAPE_CURVATURE_PATH.reAssign(CRYOVIA_PATH / "Shape_curvatures")
        cryovia_TEMP_DIR.reAssign(CRYOVIA_PATH / "temp")
        DATASET_PATH.reAssign(CRYOVIA_PATH / "DATASETS")


        CRYOVIA_PATH = Path().home() / ".cryovia_debug"

    if gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(i) for i in gpus])
    os.environ["CRYOVIA_NJOBS"] = str(njobs)
    GUI()


def show_error_popup(etype, evalue,tb):
    QMessageBox.information(None, str('error'),''.join(format_exception(etype, evalue, tb)))



def GUI():
    """
    Runs the GUI.
    ----------


    Returns
    -------
    
    """
    global cryovia, CreateNewShapesWindow, SegmentationWindow, segmentationModel, Config, DatasetGui, Dataset, MainWindow
    cryovia = __import__("cryovia", globals(), locals())
    CreateNewShapesWindow = __import__("cryovia.gui.shape_drawer", globals(), locals()).gui.shape_drawer.CreateNewShapesWindow
    SegmentationWindow = __import__("cryovia.gui.membrane_segmentation", globals(), locals()).gui.membrane_segmentation.SegmentationWindow
    segmentationModel = __import__("cryovia.gui.segmentation_files.segmentation_model", globals(), locals()).gui.segmentation_files.segmentation_model.segmentationModel
    Config = __import__("cryovia.gui.segmentation_files.segmentation_model", globals(), locals()).gui.segmentation_files.segmentation_model.Config
    DatasetGui = __import__("cryovia.gui.datasets_gui", globals(), locals()).gui.datasets_gui.DatasetGui
    Dataset = __import__("cryovia.gui.dataset", globals(), locals()).gui.dataset.Dataset
    MainWindow = __import__("grid_edge_detector.image_gui", globals(), locals()).image_gui.MainWindow
    
    

    # import cryovia
    # from cryovia.gui.shape_drawer import CreateNewShapesWindow
    # from cryovia.gui.membrane_segmentation import SegmentationWindow 
    # from cryovia.gui.segmentation_files.segmentation_model import segmentationModel, Config 
    # from cryovia.gui.datasets_gui import DatasetGui
    # from cryovia.gui.dataset import Dataset
    # from grid_edge_detector.image_gui import MainWindow



    checkFirstTime()

    sys.excepthook = show_error_popup
    app = QApplication(sys.argv)

    view = QStartingMenu()
    
    
    
    # view.resize(800, 600)
    view.show()
    sys.exit(app.exec_())


if __name__ == '__main__':

    startGui()
