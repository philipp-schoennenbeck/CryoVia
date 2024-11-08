import sys

from scipy.ndimage import label
# from PyQt5.QtCore    import *
from cryovia.cryovia_analysis.analyser import curvatureAnalyser
from PyQt5.QtCore import Qt, QPoint, QSize, QByteArray, QModelIndex,pyqtSignal, QObject,QThread, QCoreApplication
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QAction, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QListWidget, QGridLayout, QListWidgetItem, QStyle
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QAbstractItemView, QItemDelegate, QTextEdit, QTableWidget, QTableWidgetItem, QCheckBox, QFrame, QDialog
# from PyQt5.QtGui     import *
from PyQt5.QtGui import QIcon, QPainter, QPen,QPixmap, QColor, QImage,QValidator, QTextCursor
import numpy as np
# from matplotlib import pyplot as plt
from pathlib import Path
import typing
import os
import shutil
from cryovia.cryovia_analysis.shape_classifier import ShapeClassifier, ShapeClassifierFactory, get_all_classifier_paths, get_all_classifier_names, PROTECTED_SHAPES, SHAPE_CURVATURE_PATH, get_all_shapes, get_all_real_data_curvature_paths
from cryovia.cryovia_analysis.dataset import Dataset, get_all_dataset_names
from cryovia.gui.path_variables import DATASET_PATH
import qimage2ndarray as q2n
from PyQt5 import QtCore, QtGui
import pickle
from datetime import datetime
import pandas as pd
from cryovia.cryovia_analysis.analyser import AnalyserWrapper
import qimage2ndarray as q2np
from PIL import Image, ImageOps

from cryovia.gui.path_variables import cryovia_TEMP_DIR



# cryovia_TEMP_DIR = Path().home() / ".cryovia" / "temp"

class labelSessionData:
    def __init__(self, name, dataset:Dataset, data ):
        self.name = name
        self.dataset = dataset.name
        self.data = data
        self.last_run = dataset.times["Last run"]

    def save(self):
        global SHAPE_CURVATURE_PATH

        file = SHAPE_CURVATURE_PATH / f"{self.name}.pickle" 
        with open(file, "wb") as f:
            pickle.dump(self, f)

class thumbnailWidget(QFrame):
    shape = 120


    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Panel)
        self.setFrameShadow(QFrame.Raised)
        self.setLineWidth(1)
        self.imageLabel = QLabel()
        self.maskLabel = QLabel()
        self.image_viewer = None

        # self.imageLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum))
        # self.maskLabel.setSizePolicy(QSizePolicy(QSizePolicy.Minimum,QSizePolicy.Minimum))

        self.imagePixmap = QPixmap(self.shape,self.shape)
        self.maskPixmap = QPixmap(self.shape,self.shape)
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)
        self.imageLabel.adjustSize()
        self.maskLabel.adjustSize()
        self.setLayout(QGridLayout())



        
        
        self.layout().addWidget(self.imageLabel,0,0,alignment=Qt.AlignCenter)
        
        
        self.layout().addWidget(self.maskLabel,0,2,alignment=Qt.AlignCenter)



        
        # self.layout().setColumnStretch(1,1)
        # self.layout().setRowStretch(1,1)

        self.imageLabel.setToolTip("Original image")
        self.maskLabel.setToolTip("Segmentation")
        self.setMinimumSize(200, 230)


        
        return 

    def loadImages(self, wrapper:AnalyserWrapper, idx:int, ):
        self.clearAll()
        if wrapper is None:
            self.imageLabel.setText("File not found")
            self.maskLabel.setText("File not found")
            self.imagePixmap = None
            self.maskPixmap = None
            return
        self.setToolTip(str(wrapper.directory) + f"\nIndex: {str(idx)}")
        tn, seg = wrapper.get_thumbnails([idx])
        tn = tn[idx]
        seg = seg[idx]

        tn = np.array(tn)
        seg = np.array(seg)
        image = q2n.gray2qimage(tn)
       
        shape = self.shape

        if min(self.width(), self.height()) / 2 - 40 > self.shape:
            shape = min(self.width(), self.height()) / 2 - 40

        scaled = image.scaled(
            shape,
            shape,
            aspectRatioMode=Qt.KeepAspectRatio, 
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        self.imagePixmap = QPixmap.fromImage(scaled)
        self.imageLabel.setPixmap(self.imagePixmap)

        
        seg = q2n.gray2qimage(seg)

        scaled = seg.scaled(
            shape,
            shape,
            aspectRatioMode=Qt.KeepAspectRatio,
        )
        self.maskPixmap = QPixmap.fromImage(scaled)
        self.maskLabel.setPixmap(self.maskPixmap)




    
    def clearAll(self):
        if self.imagePixmap is not None:
            self.imagePixmap.fill(QColor("white"))
            self.maskPixmap.fill(QColor("white"))
            self.imageLabel.setPixmap(self.imagePixmap)
            self.maskLabel.setPixmap(self.maskPixmap)


    def resizeEvent(self, a0: QtGui.QResizeEvent) -> None:
        if self.imagePixmap is not None:
            shape = self.shape

            if min(self.width(), self.height()) / 2 - 40 > self.shape:
                shape = min(self.width(), self.height()) / 2 - 40

            scaled = self.imagePixmap.scaled(
                shape,
                shape,
                aspectRatioMode=Qt.KeepAspectRatio, 
                transformMode=Qt.TransformationMode.SmoothTransformation
            )
            self.imagePixmap = scaled
            self.imageLabel.setPixmap(self.imagePixmap)

            
            

            scaled = self.maskPixmap.scaled(
                shape,
                shape,
                aspectRatioMode=Qt.KeepAspectRatio,
            )
            self.maskPixmap = scaled
            self.maskLabel.setPixmap(self.maskPixmap)


        return super().resizeEvent(a0)

def get_analyser(row, dataset):
    if isinstance(row, str):
        micrograph = row
    else:
        micrograph = row["Micrograph"]
    if not isinstance(micrograph, str):
        micrograph = micrograph.iloc[0]
    wrapper = AnalyserWrapper(dataset.dataset_path / micrograph)
    return wrapper


class Dummy:
    def __init__(self):
        pass

class MembraneShapeSuggestorWidget(QWidget):
    def __init__(self,dataset, csv, parent=None, previousSession=None) -> None:
        super().__init__()
        self.customParent = parent
        self.dataset = dataset
        self.currentIndex = 0
        self.previousSession = previousSession
        self.csv = csv
        self.wrapper = None
        self.idx = None
        self.assignedShapes = {}
        self.curvatures = {}

        self.setLayout(QVBoxLayout())
        self.windowWidget = thumbnailWidget(self)
        self.layout().addWidget(self.windowWidget)
        

        self.nextButtonLayout = QHBoxLayout()

        self.nextButton = QPushButton(">")
        self.previousButton = QPushButton("<")

        self.previousButton.clicked.connect(lambda x: self.NextMembrane(-1))
        self.nextButton.clicked.connect(lambda x: self.NextMembrane(1))

        self.nextButtonLayout.addWidget(self.previousButton)
        self.nextButtonLayout.addWidget(self.nextButton)
        
        self.descriptionLabel = QLabel("")

        self.shapeButtonsLayout = QGridLayout()

        self.availableShapes = get_all_shapes()
        self.buttons = {}
        for i, shape in enumerate(self.availableShapes):
            button = QPushButton(shape)
            button.clicked.connect(lambda x, shape=shape: self.assignShape(shape))
            self.buttons[shape] = button
            self.shapeButtonsLayout.addWidget(button, i// 3, i%3)
        
        self.optionsLayout = QHBoxLayout()
        self.doneButton = QPushButton("Done")
        self.doneButton.clicked.connect(self.done)
        self.addNewShapeButton = QPushButton("New shape")
        self.addNewShapeButton.clicked.connect(self.addNewShape)

        self.optionsLayout.addWidget(self.addNewShapeButton)
        self.optionsLayout.addWidget(self.doneButton)


        self.layout().addLayout(self.nextButtonLayout)
        self.layout().addWidget(self.descriptionLabel)
        self.layout().addLayout(self.shapeButtonsLayout)
        self.layout().addLayout(self.optionsLayout)

        if self.previousSession is not None:
            self.extractShapes()
    
        self.NextMembrane(0)


    def addNewShape(self):
        # button = QPushButton("Editable Button", self)
        if Dummy in self.buttons:
            return
        # Create the text field over the button
        text_field = QLineEdit(self)
        text_field.setPlaceholderText("NewShape")
        # text_field.setText(button.text())
        # text_field.setGeometry(button.geometry())
        # text_field.setAlignment(button.alignment())
        
        # Connect the editingFinished signal to a validation method
        text_field.editingFinished.connect(self.validate_text)
        
        # Add widgets to the layout
        i = len(self.buttons.keys())
        # self.shapeButtonsLayout.addWidget(button, i// 3, i%3)
        self.shapeButtonsLayout.addWidget(text_field, i// 3, i%3)
        self.buttons[Dummy] = text_field
        
        

        # Ensure the text field is on top of the button
        text_field.raise_()
        text_field.setFocus()
        self.newShapeTextField = text_field

    def validate_text(self):
        pass
        
        text = self.newShapeTextField.text()
        shapes = get_all_shapes()
        if text in shapes:
            QMessageBox.warning(self, "Invalid Input", f"The shape '{text}' already exists.")
            self.newShapeTextField.setFocus()  # Keep focus on the text field
            self.newShapeTextField.selectAll()  # Optionally, select all text for easy replacement
        elif " " in text:
            QMessageBox.warning(self, "Invalid Input", f"No spaces in shape names.")
            self.newShapeTextField.setFocus()  # Keep focus on the text field
            self.newShapeTextField.selectAll()
        else:
            # Update the button text with the validated text
            # self.button.setText(text)
            

            
            self.layout().removeWidget(self.newShapeTextField)
            self.newShapeTextField.deleteLater()
            self.newShapeTextField = None
            del self.buttons[Dummy]
            if len(text) > 0:
                global SHAPE_CURVATURE_PATH
                new_path = SHAPE_CURVATURE_PATH / f"{text}.npy"
                curvatures = np.empty((0, 200), dtype=np.float64)
                # curvatures = np.array([], dtype=np.float64)
                np.save(new_path, curvatures)


                new_button = QPushButton(text)
                self.buttons[text] = new_button
                i = len(self.buttons.keys()) - 1
                self.shapeButtonsLayout.addWidget(new_button, i// 3, i%3)
                new_button.clicked.connect(lambda x, shape=text: self.assignShape(shape))
                if self.customParent is not None:
                    self.customParent.shapesListWidget.classifier_selection_changed()


    def NextMembrane(self, direction=1):
        self.currentIndex += direction
        if self.currentIndex < 0:
            self.currentIndex = 0
            
            return
        if self.currentIndex >= len(self.csv):
            self.currentIndex = len(self.csv) - 1
            return
        row = self.csv.iloc[self.currentIndex]
        self.wrapper = get_analyser(row, self.dataset)
        self.idx = row["Index"]
        self.windowWidget.loadImages(self.wrapper, self.idx)
        if self.currentIndex == 0:
            self.previousButton.setEnabled(False)
        else:
            self.previousButton.setEnabled(True)

        if self.currentIndex == len(self.csv) - 1:
            self.nextButton.setEnabled(False)
        else:
            self.nextButton.setEnabled(True)
        shape = row["Shape"]
        proba = row["Shape probability"]
        for button in self.buttons.values():
            button.setStyleSheet("")
        if shape in self.buttons:
            self.buttons[shape].setStyleSheet("background-color: lightsalmon;")

        if self.currentIndex in self.assignedShapes:
            self.descriptionLabel.setText(f"{shape}: {proba} --> {self.assignedShapes[self.currentIndex]}")
            if self.assignedShapes[self.currentIndex] in self.buttons:
                self.buttons[self.assignedShapes[self.currentIndex]].setStyleSheet("background-color: lightgreen;")
        else:

            self.descriptionLabel.setText(f"{shape}: {proba}")
    

    def extractShapes(self):
        if self.previousSession is not None:
            self.previousSession:labelSessionData
            for shape, curvatures in self.previousSession.data.items():
                for curvature in curvatures:
                    self.assignedShapes[curvature["index"]] = shape
                    self.curvatures[curvature["index"]] = curvature

    def assignShape(self, shape):
        self.assignedShapes[self.currentIndex] = shape
        row = self.csv.iloc[self.currentIndex]
        previous_shape = row["Shape"]
        proba = row["Shape probability"]
        wrapper = get_analyser(row, self.dataset)
        membrane = wrapper.analyser[row["Index"]]
        self.curvatures[self.currentIndex]= {"curvatures":membrane.resize_curvature(200,100), "index":self.currentIndex}

        self.descriptionLabel.setText(f"{previous_shape}: {proba} --> {self.assignedShapes[self.currentIndex]}")
        self.NextMembrane(1)

    
    def done(self):
        global SHAPE_CURVATURE_PATH
        qm = QMessageBox()
        qm.setText(f"Add curvatures to shape files (Cannot be converted without resetting the files).\nAdd this as a separate option.")
        qm.setIcon(qm.Icon.Warning)
        qm.addButton(QPushButton("Add to shape files"),  QMessageBox.ButtonRole.YesRole)

        qm.addButton(QPushButton("Separate option"), QMessageBox.ButtonRole.AcceptRole)
        # qm.addButton(QPushButton("Remove for future analysis"), QMessageBox.ButtonRole.ActionRole)
        qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
        # qm.setCheckBox()
        
        ret = qm.exec_()
        
        
        if ret == 2:
            return
        if ret == 0:
            shapes_values = {}
            for key, value in self.assignedShapes.items():
                if value not in shapes_values:
                    shapes_values[value] = []
                shapes_values[value].append(self.curvatures[key]["curvatures"])
            

            for key, curvatures in shapes_values.items():
                curvatures = np.array(curvatures)
       
                file = SHAPE_CURVATURE_PATH / f"{key}.npy"
                old_curvatures = np.load(file)
                curvatures = np.concatenate((old_curvatures, curvatures))
                np.save(file, curvatures)
        if ret == 1:
            shapes_values = {}

            name = datetime.now().strftime(f"%Y%m%d-%H:%M:%S_{self.dataset.name}.pickle")
            manual_file = SHAPE_CURVATURE_PATH / name
            counter = 0
            

            while manual_file.exists():
                manual_file:Path = SHAPE_CURVATURE_PATH / name.replace(".pickle", f"_{counter}.pickle")
                counter += 1
            for key, value in self.assignedShapes.items():
                if value not in shapes_values:
                    shapes_values[value] = []
                shapes_values[value].append(self.curvatures[key])
            # for key in shapes_values.keys():
            #     shapes_values[key] = np.array(shapes_values[key])

            if self.previousSession is not None:
                newSession = self.previousSession
                newSession.data = shapes_values
            else:
                newSession = labelSessionData(manual_file.stem, self.dataset, shapes_values)
            newSession.save()
            
        self.close()


    def closeEvent(self, closeevent):
        if self.customParent is not None:
            self.customParent.shapesListWidget.classifier_selection_changed()

        super().closeEvent(closeevent)


class chooseDatasetWidget(QWidget):
    def __init__(self, parent) -> None:
        global DATASET_PATH
        super().__init__()
        self.setLayout(QVBoxLayout())
        self.customParent = parent

        self.explainLabel = QLabel("Choose a dataset. You can label the shapes of real vesicles and later add it to a training run of a classifier.\nThe shown shapes are sorted by least to highest shape probability.")
        self.explainSessionLabel = QLabel("Or choose a previous labeling session to continue.\nSessions from which the dataset has been deleted or changed cannot be continued.")
        datasets = [Dataset.load(i) for i in get_all_dataset_names()]
        self.datasets = {dataset.name:dataset for dataset in datasets}
        self.listWidget = QListWidget()
        for dataset_name, dataset in self.datasets.items():
            self.listWidget.addItem(dataset_name)
        self.buttonLayout = QHBoxLayout()

        self.previousSessionListWidget = QListWidget()
        previousSessions = get_all_real_data_curvature_paths()
        self.sessions = {}
        
        for session in previousSessions:
            with open(session, "rb") as f:
                session_object:labelSessionData = pickle.load(f)
            newItem = QListWidgetItem(session_object.name)
            self.previousSessionListWidget.addItem(newItem)
            self.sessions[session_object.name] = session_object
            if session_object.dataset in self.datasets:
                try:
                    dataset = Dataset.load(DATASET_PATH / session_object.dataset)
                    if dataset.times["Last run"] != session_object.last_run:
                        newItem.setFlags(newItem.flags() & ~Qt.ItemIsSelectable)
                        newItem.setToolTip("Cannot be selected because the dataset has changed since creating this session.")
                except FileNotFoundError:
                    newItem.setFlags(newItem.flags() & ~Qt.ItemIsSelectable)
                    newItem.setToolTip("Cannot be selected because the dataset does not exist anymore.")

        
        self.listWidget.itemSelectionChanged.connect(self.selectionChangeListWidget)
        self.previousSessionListWidget.itemSelectionChanged.connect(self.selectionChangePreviousListWidget)

        self.chooseButton = QPushButton("Okay")
        self.cancelButton = QPushButton("Cancel")

        self.buttonLayout.addWidget(self.chooseButton)
        self.buttonLayout.addWidget(self.cancelButton)

        self.layout().addWidget(self.explainLabel)
        self.layout().addWidget(self.listWidget)
        self.layout().addWidget(self.explainSessionLabel)
        self.layout().addWidget(self.previousSessionListWidget)
        self.layout().addLayout(self.buttonLayout)

        self.chooseButton.clicked.connect(self.customParent.openNextWindow)
        self.cancelButton.clicked.connect(self.close)
        self.chooseButton.setEnabled(False)
        self.dummyWindow = None

    def selectionChangeListWidget(self):
        if len(self.listWidget.selectedItems()) > 0:
            self.previousSessionListWidget.clearSelection()
        self.enableButton()

    def selectionChangePreviousListWidget(self):
        if len(self.previousSessionListWidget.selectedItems()) > 0:
            self.listWidget.clearSelection()
        self.enableButton()
    
    def enableButton(self):
        self.chooseButton.setEnabled((len(self.listWidget.selectedItems()) + len(self.previousSessionListWidget.selectedItems())) > 0)


class ShapeDrawingWindow(QLabel):
    """
    A window to draw shapes in for a shape classifier to train on.
    """
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.myPenColorOn = Qt.red   
        self.myPenColorOff = Qt.black
        self.myPenWidth = 10        
        self.currentlyDraw = 0
        
        self.lastPoint = QPoint()
        
        self.setPixmap(QPixmap(500,500))
        self.pixmap().fill(QColor("black"))


    def mousePressEvent(self, event):
        if event.button() == 4:
            self.pixmap().fill(QColor("black"))
            self.update()
        if (event.button() == Qt.LeftButton) and self.currentlyDraw == 0:
            self.lastPoint = event.pos()
            self.currentlyDraw = 1
        elif (event.button() == Qt.RightButton) and self.currentlyDraw == 0:
            self.lastPoint = event.pos()
            self.currentlyDraw = -1


    def mouseMoveEvent(self, event):

        pencolor = self.myPenColorOff
        if self.currentlyDraw == 1:
            pencolor = self.myPenColorOn
        if ((event.buttons() & Qt.LeftButton) or (event.buttons() & Qt.RightButton)) and self.currentlyDraw != 0:
            
            painter = QPainter(self.pixmap())
            painter.setPen(QPen(pencolor, self.myPenWidth, 
                                Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            # print(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()
        
    def sizeHint(self) -> QSize:
        return self.pixmap().size()


    def mouseReleaseEvent(self, a0) -> None:
        if self.currentlyDraw == 1 and a0.button() == Qt.LeftButton:
            self.currentlyDraw = 0
        elif self.currentlyDraw == -1 and a0.button() == Qt.RightButton:
            self.currentlyDraw = 0
        return super().mouseReleaseEvent(a0)




def QPixmapToArray(pixmap:QPixmap):
    """
    Converts QPixmap to a numpy array.
    Parameters
    ----------
    pixmap  : A Qpixmap to convert.

    Returns
    -------
    img     : numpy array
    """
    ## Get the size of the current pixmap

    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits()
    byte_str.setsize(h*w*4)

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w,h,4))

    return img


# class QShapeValidator(QValidator):
#     def __init__(self, parent, known_shapes) -> None:
#         super().__init__(parent)
#         self.known_shapes = known_shapes
#     def validate(self, a0: str, a1: int) -> typing.Tuple['QValidator.State', str, int]:
#         if a0 in self.known_shapes or len(a0) == 0:
#             return (QValidator.State.Intermediate, a0, a1)
#         else:
#             return (QValidator.State.Acceptable, a0, a1)

#     def fixup(self, a0: str) -> str:
#         counter = 0
#         if len(a0) == 0:
#             a0 = "New shape"
#         split_name = a0.split("_")
#         if len(split_name) > 1 and split_name[-1].isnumeric():
#             a0 = "_".join(split_name[:-1])
#         while True:
#             new_name = a0 + "_" + str(counter)
#             if new_name not in self.known_shapes:
#                 return new_name
#             counter += 1




class CalcCurvatureWorker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, image, shape,idx, flip, vary_size, rotate, adaptive):
        super().__init__()
        self.image = Image.fromarray(image)
        self.shape = shape
        self.flip = flip
        self.vary_size = vary_size
        self.rotate = rotate
        self.adaptive = adaptive
        self.idx = idx
        # self.threshold = threshold


    def create_augmented_images(self ):
        """
        Creates augmentation of the given images. Can augment sizes, flips and rotations.
        Parameters
        ----------


        Returns
        -------
        new_images : numpy arrays
        """
        
        new_images = [self.image]

        if self.vary_size:
            varied_images = []
            for multi in [0.5, 0.75, 1.25, 1.5]:
                for img in new_images:
                    varied_images.append(img.resize([int(s * multi) for s in img.size]))
            new_images.extend(varied_images)
        if self.flip:
            flipped_images = []
            for img in new_images:
                flipped_images.append(ImageOps.flip(img))
            new_images.extend(flipped_images)
        
        if self.rotate:
            rotated_images = []
            for img in new_images:
                for d in [90,180,270]:
                    rotated_images.append(img.rotate(d))
            new_images.extend(rotated_images)
        new_images = [np.array(img) for img in new_images]
        return new_images


    def calc_curvature(self, new_images):
        """
        Calculates the curvature of shapes in new_images
        Parameters
        ----------
        new_images : numpy arrays with binary shapes

        Returns
        -------
        curvatures : curvatures of each image
        """
        curvatures = []
        
        for img in new_images:
            analyser = curvatureAnalyser(img)
            if self.adaptive:
                analyser.estimateCurvatureAdaptive(400, 5, step=20)
            else:
                analyser.estimateCurvature(max_neighbour_dist=80)
            curvs = analyser.membranes[0].resize_curvature(200,100)
            
            # raise NotImplementedError
            # g = graph(image=img, max_nodes=5, segmentation_thickness=None, min_size=0, max_hole_size=0)
            # g.estimate_curvature(max_neighbour_dist=80)
            # curvs = g.get_all_curvatures_values(for_shape_predicting=True, )
            # if curvs is None:
            #     print("no curvs")
            #     continue
            curvatures.append(curvs)


        return curvatures


    def run(self):
        new_images = self.create_augmented_images()
        curvatures = self.calc_curvature(new_images)
        self.progress.emit((curvatures, self.shape, self.idx))
        self.finished.emit()
    




class ConfusionTableWidget(QTableWidget):
    def __init__(self, parent, classifier:ShapeClassifier):
        super().__init__()
        self.customParent = parent

        self.classifier = classifier
        if self.classifier.confusion_matrix is None:
            self.setWindowTitle("Not trained yet.")
        else:
            self.setWindowTitle(f"Confusion matrix of {self.classifier.name} classifier")
            confusion_matrix, classes = self.classifier.confusion_matrix
            self.setRowCount(len(confusion_matrix) )
            self.setColumnCount(len(confusion_matrix) )
            classes = [str(i) for i in classes]
            self.setVerticalHeaderLabels(classes)
            self.setHorizontalHeaderLabels(classes)
            # for counter, cls in enumerate(classes):
            #     self.setItem(counter + 1, 0, QTableWidgetItem(cls))
            #     self.setItem(0, counter + 1, QTableWidgetItem(cls))
            for row, row_items in enumerate(confusion_matrix):
                for column, column_item in enumerate(row_items):
                    self.setItem(row, column, QTableWidgetItem(str(column_item)))
                
            self.resizeColumnsToContents()
            self.resizeRowsToContents()


class SideWindow(QWidget):
    def __init__(self, parent, draw_window:ShapeDrawingWindow , shape_widget):
        super().__init__(parent=parent)
        self.drawWindow = draw_window
        self.workerCounter = 0
        self.workers = []
        self.threads = []

        self.shapeWidget:ShapesListWidget = shape_widget

        self.setLayout(QVBoxLayout())

        self.addImageToShapeButton = QPushButton("+")
        self.addImageToShapeButton.font().setBold(True)
        self.addImageToShapeButton.clicked.connect(self.addImageToShape)
        self.addImageToShapeButton.setToolTip("Add the curvature values of the current image to the selected shape in the bottom left corner.")

        self.clearButton = QPushButton()
        self.clearButton.setIcon(self.clearButton.style().standardIcon(QStyle.SP_BrowserReload))
        self.clearButton.clicked.connect(self.clearDrawer)
        self.clearButton.setToolTip("Clear the drawing board.")


        self.varySizesCheckbox = QCheckBox(text="Vary size")
        self.varySizesCheckbox.setChecked(True)
        self.varySizesCheckbox.setToolTip("Vary sizes for curvature estimation.")
        self.rotateCheckbox = QCheckBox(text="Rotate")
        self.rotateCheckbox.setChecked(True)
        self.rotateCheckbox.setToolTip("Calculate curvature for rotations as well.")
        self.flipCheckbox = QCheckBox(text="Flip")
        self.flipCheckbox.setChecked(True)
        self.flipCheckbox.setToolTip("Calculate curvature for flipped image as well.")
        self.adaptiveCheckbox = QCheckBox(text="Adaptive curvature")
        self.adaptiveCheckbox.setChecked(False)
        self.adaptiveCheckbox.setToolTip("Whether to use adaptive curvature estimation.")

        self.buttonLayout = QHBoxLayout()

        self.buttonLayout.addWidget(self.addImageToShapeButton)
        self.buttonLayout.addWidget(self.clearButton)
        self.buttonLayout.addWidget(self.rotateCheckbox)
        self.buttonLayout.addWidget(self.flipCheckbox)
        self.buttonLayout.addWidget(self.varySizesCheckbox)
        self.buttonLayout.addWidget(self.adaptiveCheckbox)

        self.messageBoard = QTextEdit(self)
        self.messageBoard.setReadOnly(True)
        self.messageBoard.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        
        self.messageBoard.font().setFamily("Courier")
        self.messageBoard.font().setPointSize(10)

        self.layout().addLayout(self.buttonLayout)
        self.layout().addWidget(self.messageBoard)


    def clearDrawer(self):
        
        self.drawWindow.pixmap().fill(QColor("black"))
        self.drawWindow.update()

    def addImageToShape(self):
        def checkImage(image_to_check:np.array):
            padded_image = np.pad(image_to_check,1)
            inverse_image = (padded_image == 0) * 1
            labs, features = label(inverse_image, np.ones((3,3)))
            return features in [1,2]
                
        image = self.drawWindow.pixmap().toImage() 
        if not image.allGray():
            # gray_image = image.convertToFormat(QImage.Format.Format_Grayscale8)

            np_image = q2np.rgb_view(image)
            np_image = np.mean(np_image, -1)
            np_image = np_image > 0
            if checkImage(np_image):
                result = self.calcCurvature(np_image)
                if not result:
                    self.newMessage("No shape selected.\n")
                    return
            else:
                self.newMessage("Shape is invalid.\n")
            self.clearDrawer()



    def calcCurvature(self, image):

        shape = self.shapeWidget.usedListWidget.selectedItems()
        if len(shape) == 0:
            shape = self.shapeWidget.unusedListWidget.selectedItems()
            if len(shape) == 0:
                return False
        shape = shape[0].text()
        shape = shape.replace(" (!)", "")

        self.newMessage(f"Starting to calculate curvature for image {self.workerCounter}.\n")
        thread = QThread()
        worker = CalcCurvatureWorker(image, shape, self.workerCounter, self.flipCheckbox.isChecked(), self.varySizesCheckbox.isChecked(), self.rotateCheckbox.isChecked(), self.adaptiveCheckbox.isChecked() )

        self.workerCounter += 1
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.finishedRunning)
        worker.progress.connect(self.progressEmited)
        
        thread.start()   
        self.threads.append(thread)
        self.workers.append(worker)
        
        return True

    def finishedRunning(self):

        pass
    
    def progressEmited(self, emit):
        global SHAPE_CURVATURE_PATH
        curvatures, shape, idx = emit
        path = SHAPE_CURVATURE_PATH / f"{shape}.npy"
        old_curvatures = np.load(path)
        if len(old_curvatures) > 0:
            new_curvatures = np.concatenate([old_curvatures, curvatures])
        else:
            new_curvatures = curvatures
        np.save(path, new_curvatures)
        self.newMessage(f"Finished calculating curvature for image {idx}.\n")

        


    def newMessage(self, message:str):
        cursor = self.messageBoard.textCursor()
        cursor.insertText(message)
        self.messageBoard.moveCursor(QTextCursor.End)
        QCoreApplication.processEvents()







class ClassifierListWidgetItem(QListWidgetItem):
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent)
        self.classifier:ShapeClassifier = ShapeClassifierFactory(**kwargs)
        self.setText(str(self.classifier))
        self.classifier.changed_hooks.append(self.reset_name)
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
        
    def __repr__(self) -> str:
        return self.classifier.name

    def reset_name(self):
        self.setText(str(self.classifier))

class NotAllowedValidator(QValidator):
    def __init__(self, parent, not_allowed_func, corresponding_button:QPushButton, spaces_forbidden=False):
        super().__init__(parent)
        self.not_allowed_func = not_allowed_func
        self.button = corresponding_button
        self.spaces_forbidden = spaces_forbidden

    
    def validate(self, a0: str, a1: int) -> typing.Tuple['QValidator.State', str, int]:
        used_names = self.not_allowed_func()
        if self.spaces_forbidden:
            a0 = a0.replace(" ", "")
        if a0 in used_names:
            if self.button is not None:
                self.button.setEnabled(False)
            return (QValidator.State.Intermediate, a0, a1)
        if len(a0) == 0:
            if self.button is not None:
                self.button.setEnabled(False)
            return (QValidator.State.Intermediate, a0, a1)
        if self.button is not None:
            self.button.setEnabled(True)
        return (QValidator.State.Acceptable, a0, a1)


class NewNameDialog(QWidget):
    def __init__(self, parent, not_available_names_func, labelstring, title, clicked_func, spaces_forbidden=False):
        super().__init__()
        self.customParent:ClassifierListWidget = parent
        self.setWindowTitle(title)

        self.setLayout(QVBoxLayout())

        self.label = QLabel(text=labelstring)

        self.lineEdit = QLineEdit()
        self.lineEdit.returnPressed.connect(self.addClicked)
        
        self.buttonLayout = QHBoxLayout()
        self.okButton = QPushButton("Add")
        self.okButton.setEnabled(False)
        self.okButton.clicked.connect(self.addClicked)
        self.cancelButton = QPushButton("Cancel")
        self.buttonLayout.addWidget(self.okButton)
        self.buttonLayout.addWidget(self.cancelButton)
        self.lineEdit.setValidator(NotAllowedValidator(self, not_available_names_func, self.okButton, spaces_forbidden))
        self.cancelButton.clicked.connect(self.close)
        self.clickedFunc = clicked_func

        self.layout().addWidget(self.label)
        self.layout().addWidget(self.lineEdit)
        self.layout().addLayout(self.buttonLayout)
    
    def addClicked(self):
        new_name = self.lineEdit.text()
        if self.lineEdit.validator().validate(new_name, 0)[0] == QValidator.State.Acceptable:
            self.clickedFunc(new_name)
            self.close()
    



class CustomItemDelegate(QItemDelegate):
    def createEditor(self, parent, option: 'QStyleOptionViewItem', index) -> QWidget:
        def editingFinishedFunc():
            self.parent().selectedItems()[0].classifier.rename(editor.text())
        editor = QLineEdit(parent)
        parent:ClassifierListWidget
        editor.setValidator(NotAllowedValidator(editor, get_all_classifier_names, None))
        editor.editingFinished.connect(editingFinishedFunc)
        return editor 
    

class ClassifierListWidget(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        
        self.setLayout(QVBoxLayout())
        
        self.ListWidget = QListWidget(self)
        # self.ListWidget.doubleClicked.connect(self.test)
        self.ListWidget.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.ListWidget.setItemDelegate(CustomItemDelegate(self.ListWidget))
        self.ListWidget.itemSelectionChanged.connect(self.loadNewClassifier)

        self.classifierLabel = QLabel("Available classifiers")
        self.realDataWindow = None

        self.attributesLayout = QGridLayout()

        self.attributesLabel =QLabel("Attributes")
        self.typeLabel = QLabel("Type")
        self.typeCombobox = QComboBox()
        self.typeCombobox.addItems(["NeuralNetwork", "GradientBoostingClassifier"])
        self.typeCombobox.currentTextChanged.connect(self.changeType)

        self.only_closedLabel = QLabel("Only closed")
        self.only_closedCheckbox = QCheckBox()
        self.only_closedCheckbox.clicked.connect(self.changedClosed)


        self.attributesLayout.addWidget(self.attributesLabel)
        self.attributesLayout.addWidget(self.typeLabel, 1,0)
        self.attributesLayout.addWidget(self.typeCombobox, 1,1)
        self.attributesLayout.addWidget(self.only_closedLabel,2,0)
        self.attributesLayout.addWidget(self.only_closedCheckbox,2,1)


        self.addButton = QPushButton("+")
        self.addButton.font().setBold(True)
        self.addButton.clicked.connect(self.openClassifierDialog)
        self.addButton.setToolTip("Create a new classifier.")

        self.removeButton = QPushButton()
        self.removeButton.setIcon(self.removeButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.removeButton.clicked.connect(self.removeClassifier)
        self.removeButton.setToolTip("Remove this classifier.")

        self.copyButton = QPushButton()
        self.copyButton.setIcon(self.copyButton.style().standardIcon(QStyle.SP_DialogResetButton))
        self.copyButton.clicked.connect(self.copyClassifier)
        self.copyButton.setToolTip("Create a copy of this classifier.")

        self.trainButton = QPushButton()
        self.trainButton.setIcon(self.trainButton.style().standardIcon(QStyle.SP_ComputerIcon))
        self.trainButton.clicked.connect(self.trainClassifier)
        self.trainButton.setToolTip("Train this classifier with all added curvatures.")

        self.showConfusionMatrixButton = QPushButton("#")
        self.showConfusionMatrixButton.font().setBold(True)
        self.showConfusionMatrixButton.clicked.connect(self.showConfusionMatrix)
        self.showConfusionMatrixButton.setToolTip("Show confusion matrix of last training.")

        # self.renameButton = QPushButton()
        # self.renameButton.setIcon(self.renameButton.style().standardIcon(QStlye.))
    
        self.getRealDataButton = QPushButton("Add training data from datasets")
        self.getRealDataButton.clicked.connect(self.getRealData)

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.removeButton)
        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addWidget(self.copyButton)
        self.buttonLayout.addWidget(self.trainButton)
        self.buttonLayout.addWidget(self.showConfusionMatrixButton)

        self.layout().addLayout(self.attributesLayout)

        self.layout().addWidget(self.classifierLabel)
        self.layout().addWidget(self.ListWidget)
        self.layout().addLayout(self.buttonLayout)
        self.layout().addWidget(self.getRealDataButton)



        # self.removeButton.font().setBold(True)
        self.createItems()

    def getRealData(self):
        self.realDataWindow = chooseDatasetWidget(self)
        self.realDataWindow.show()
        

    def openNextWindow(self):
        dataset = None
        previousSession = None
        dataset_list = self.realDataWindow.listWidget.selectedItems()
        if len(dataset_list) == 1:

            dataset = self.realDataWindow.datasets[dataset_list[0].text()]
        elif len(dataset_list) == 0:
            dataset_list = self.realDataWindow.previousSessionListWidget.selectedItems()
            dataset = Dataset.load(self.realDataWindow.sessions[dataset_list[0].text()].dataset)
            previousSession = self.realDataWindow.sessions[dataset_list[0].text()]
        if dataset is not None:

            
            self.realDataWindow.close()
            # dataset = Dataset.load(dataset)
            csv:pd.DataFrame = dataset.csv
            sorted_csv = csv.sort_values("Shape probability")


            self.realDataWindow  = MembraneShapeSuggestorWidget(dataset, sorted_csv, self.parent(), previousSession=previousSession)
            self.realDataWindow.show()
            
            


    def changeType(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            classifier.type_ = self.typeCombobox.currentText()
            self.loadNewClassifier()
            
        
    def changedClosed(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            classifier.only_closed = self.only_closedCheckbox.isChecked() 
            self.loadNewClassifier()
           

    def loadNewClassifier(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            self.only_closedCheckbox.setChecked(classifier.only_closed)
            self.typeCombobox.setCurrentText(classifier.type)

    def showConfusionMatrix(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            self.confusionMatrixWidget = ConfusionTableWidget(self, classifier)
            self.confusionMatrixWidget.show()

    def createItems(self):
        self.ListWidget.clear()
        # for item in self.ListWidget.items():
        #     self.ListWidget.removeItemWidget(item)
        
        paths = sorted(get_all_classifier_paths())
        for path in paths:
            self.ListWidget.addItem(ClassifierListWidgetItem(self.ListWidget, filepath=path))

    def openClassifierDialog(self):
        self.dialog = NewNameDialog(self, get_all_classifier_names, "New classifier name", "New classifier", self.createNewClassifier)
        self.dialog.show()

    def createNewClassifier(self, name):
        self.ListWidget.addItem(ClassifierListWidgetItem(self.ListWidget, name=name))

    def copyClassifier(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            copied_classifier = classifier.create_copy()
            self.ListWidget.addItem(ClassifierListWidgetItem(self.ListWidget, classifier=copied_classifier))

    def trainClassifier(self):
        
        items = self.ListWidget.selectedItems()
        if len(items) == 1:

            real_data = get_all_real_data_curvature_paths()
            real_data = {rd.stem:rd for rd in real_data}
            extra_data = []
            if len(real_data) > 0:
               dialog = chooseSessionDialog(real_data)
               ret = dialog.exec_()
               if ret == 0:
                    return
               else:
                    extra_data = [real_data[i.text()] for i in dialog.listwidget.selectedItems()]

            classifier:ShapeClassifier = items[0].classifier
            self.parent().sideWindow.newMessage(f"Training classifier: {classifier.name}. Please wait.\n")

            classifier.train(extra_data=extra_data)
            self.parent().sideWindow.newMessage(f"Training finished for classifier: {classifier.name}.\n")

    def removeClassifier(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            if classifier.writable:
                self.createWarningMessage(f"Remove {classifier.name} classifier? You cannot undo this.", classifier.remove)
            else:
                self.cannotDelete()
                # self.createWarningMessage("You cannot remove the Default classifier.", None)

    def createWarningMessage(self, message, func_when_okay):
        messageBox = QMessageBox()
        title = "Are you sure?"    
        reply = messageBox.question(self, title, message, messageBox.Yes | messageBox.No, messageBox.No)
        if reply == messageBox.Yes:
            if func_when_okay is not None:
                func_when_okay()
                self.createItems()


        # elif reply == messageBox.No:
            
        #     event.accept()
        # else:
        #     event.ignore()
        
    def cannotDelete(self):
        messageBox = QMessageBox()
        title = "Not possible"
        message = "You cannot remove the Default classifier."
        reply = messageBox.question(self, title, message, messageBox.Cancel, messageBox.Cancel)





class chooseSessionDialog(QDialog):
    def __init__(self, real_data, parent=None):
        def changeNextButton():
            number_of_datasets = len(self.listwidget.selectedItems())
            if number_of_datasets == 0:
                self.ok_button.setText("Train")
            else:
                self.ok_button.setText(f"Train with {number_of_datasets} sessions")

        super().__init__(parent)
        
        # self.setWindowTitle(title)
        
        # Create layout
        layout = QVBoxLayout(self)

        # Add a message label
        label = QLabel(f"You have training data from real datasets.\nDo you want to add any of these to the training?\nSelect the training sessions you want to add.")
        layout.addWidget(label)

        # Add the QListWidget
        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for i in real_data.keys():
            self.listwidget.addItem(i)
        self.listwidget.itemSelectionChanged.connect(changeNextButton)
        layout.addWidget(self.listwidget)

        # Add buttons
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("Next")
        
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)

        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)


class ShapesListWidget(QWidget):
    def __init__(self,parent, list_widget:QListWidget):
        super().__init__(parent)
        self.list_widget = list_widget

        self.setLayout(QGridLayout())
        self.usedListWidget = QListWidget(self)
        
        # print(self.usedListWidget.sizePolicy().)
        # self.usedListWidget.setFixedSize(120,100)

        self.unusedListWidget = QListWidget(self)
        # self.unusedListWidget.setFixedSize(120,100)


        self.usedListWidget.itemSelectionChanged.connect(self.clear_unused_selection)
        self.unusedListWidget.itemSelectionChanged.connect(self.clear_used_selection)
        self.buttonLayout = QVBoxLayout()
        
        self.usedToUnsedButton = QPushButton(text=">")
        # self.usedToUnsedButton.setIcon(self.usedToUnsedButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.usedToUnsedButton.font().setBold(True)
        self.usedToUnsedButton.clicked.connect(self.used_to_unused)
        self.usedToUnsedButton.setFixedSize(30,23)
        self.usedToUnsedButton.setToolTip("Remove this shape from the classifier.")

        self.unusedToUsedButton = QPushButton(text="<")
        self.unusedToUsedButton.font().setBold(True)
        self.unusedToUsedButton.clicked.connect(self.unused_to_used)
        self.unusedToUsedButton.setFixedSize(30,23)
        self.unusedToUsedButton.setToolTip("Add this shape to the classifier")

        self.removeShapeButton = QPushButton()
        self.removeShapeButton.setIcon(self.removeShapeButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.removeShapeButton.setFixedSize(30,23)
        self.removeShapeButton.clicked.connect(self.removeShape)
        self.removeShapeButton.setToolTip("Remove this shape. (You cannot use this shape for future training. Already trained classifiers keep this shape but cannot retrain with it)")
        
        self.addNewShapeButton = QPushButton("+")
        self.addNewShapeButton.font().setBold(True)
        self.addNewShapeButton.clicked.connect(self.openNewShapeDialog)
        self.addNewShapeButton.setFixedSize(30,23)
        self.addNewShapeButton.setToolTip("Create a new shape.")

        self.buttonLayout.addWidget(self.usedToUnsedButton, alignment=Qt.AlignmentFlag.AlignBottom)
        self.buttonLayout.addWidget(self.unusedToUsedButton, alignment=Qt.AlignmentFlag.AlignBottom)
        self.buttonLayout.addWidget(self.addNewShapeButton, alignment=Qt.AlignmentFlag.AlignTop)
        self.buttonLayout.addWidget(self.removeShapeButton, alignment=Qt.AlignmentFlag.AlignTop)
        
        
        self.usedLabel = QLabel(text="Used shapes")
        self.unusedLabel = QLabel(text="Available shapes")

        self.layout().addWidget(self.usedLabel, 0,0,alignment=Qt.AlignmentFlag.AlignBottom)
        self.layout().addWidget(self.unusedLabel, 0,2,alignment=Qt.AlignmentFlag.AlignBottom)
 
        self.layout().addWidget(self.usedListWidget,1,0)
        self.layout().addLayout(self.buttonLayout,1,1)
        self.layout().addWidget(self.unusedListWidget,1,2)

        self.list_widget.itemSelectionChanged.connect(self.classifier_selection_changed)
    
    def openNewShapeDialog(self):
        self.dialog = NewNameDialog(self, get_all_shapes, "New shape name", "New shape", self.createNewShape,True)
        self.dialog.show()

    def createNewShape(self, name):
        global SHAPE_CURVATURE_PATH
        new_path = SHAPE_CURVATURE_PATH / f"{name}.npy"
        curvatures = np.empty((0, 200), dtype=np.float64)
        # curvatures = np.array([], dtype=np.float64)
        np.save(new_path, curvatures)
        self.classifier_selection_changed()

    def removeShape(self):
        name = self.usedListWidget.selectedItems()
        if len(name) == 0:
            name = self.unusedListWidget.selectedItems()
            if len(name) == 0:
                return
        name = name[0].text()
        
        if name in PROTECTED_SHAPES:
            messageBox = QMessageBox()
            title = "Not possible."    
            message = f"You cannot remove \"{name}\" because it is part of the default classifier."
            
            reply = messageBox.question(self, title, message, messageBox.Cancel, messageBox.Cancel)
            return
        
        counter = 0
        if " (!)" in name:
            name = name.replace(" (!)", "")
        
        else:
            
        
            for idx in range(self.list_widget.count()):
                classifier:ShapeClassifier = self.list_widget.item(idx).classifier
                used, unused, removed = classifier.classes
                if name in used:
                    counter += 1
        
        
        messageBox = QMessageBox()
        title = "Are you sure?"    
        message = f"Are you sure you want to delete the shape \"{name}\" permanently?"
        if counter > 0:
            message += f"\nFound \"{name}\" in {counter} classifiers."
        reply = messageBox.question(self, title, message, messageBox.Yes | messageBox.No, messageBox.No)
        if reply == messageBox.Yes:
            if (SHAPE_CURVATURE_PATH / f"{name}.npy").exists():
                os.remove(SHAPE_CURVATURE_PATH / f"{name}.npy")
            items = self.list_widget.selectedItems()
            if len(items) == 1:
                classifier:ShapeClassifier = items[0].classifier
                classifier.remove_class(name)
            self.classifier_selection_changed()


    def used_to_unused(self):
        idx = self.usedListWidget.selectedIndexes()
        if len(idx) == 1:
            idx = idx[0].row()
            item = self.usedListWidget.takeItem(idx)
            self.unusedListWidget.addItem(item)
            cls = item.text()
            if " (!)" in cls:
                cls = cls.replace(" (!)", "")
            self.list_widget.selectedItems()[0].classifier.remove_class(cls)
            self.classifier_selection_changed()
            

    def unused_to_used(self):
        
        idx = self.unusedListWidget.selectedIndexes()
        if len(idx) == 1:
            idx = idx[0].row()
            item = self.unusedListWidget.takeItem(idx)
            self.usedListWidget.addItem(item)
            self.list_widget.selectedItems()[0].classifier.add_class(item.text())
            self.classifier_selection_changed()
    
    def clear_used_selection(self):
        
        idx = self.unusedListWidget.selectedIndexes()
        self.usedListWidget.clearSelection()
        if len(idx) == 1:
            self.unusedListWidget.setCurrentRow(idx[0].row())

    def clear_unused_selection(self):
        idx = self.usedListWidget.selectedIndexes()
        self.unusedListWidget.clearSelection()
        if len(idx) == 1:
            self.usedListWidget.setCurrentRow(idx[0].row())
    
    def classifier_selection_changed(self):
        items = self.list_widget.selectedItems()
        self.usedListWidget.clear()
        self.unusedListWidget.clear()
        if len(items) == 1:
            classifier:ShapeClassifier = items[0].classifier
            used, unused, removed = classifier.classes
            
            self.usedListWidget.addItems(used)
            self.unusedListWidget.addItems(unused)
            self.usedListWidget.addItems([r + " (!)" for r in removed])

class CreateNewShapesWindow(QWidget):
    def __init__(self, parent=None, custom_parent=None) -> None:
        super().__init__(parent=parent)
        self.customParent = custom_parent
        self.setLayout(QGridLayout())
        # self.layout().setContentsMargins(0,0,0,0)
        self.drawWindow = ShapeDrawingWindow(self)
        self.listWidget = ClassifierListWidget(self)
        self.shapesListWidget = ShapesListWidget(self, self.listWidget.ListWidget)
        self.sideWindow = SideWindow(self, self.drawWindow, self.shapesListWidget)

        self.layout().addWidget(self.drawWindow,0,1)
        self.layout().addWidget(self.sideWindow,1,1)
        self.layout().addWidget(self.listWidget,0,0, Qt.AlignmentFlag.AlignBottom)
        self.layout().addWidget(self.shapesListWidget,1,0)


        


    def closeEvent(self, a0) -> None:
        if self.customParent is not None:
            self.customParent.child_closed()
        return super().closeEvent(a0)



if __name__ == '__main__':
    pass
