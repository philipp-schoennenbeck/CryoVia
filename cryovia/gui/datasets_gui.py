
import sys
from typing import IO
from PyQt5.QtWidgets import QApplication, QStyleOptionViewItem, QTableView, QTabWidget,QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QGridLayout, QLabel, QLineEdit, QTextEdit, QAbstractItemView,QStyle, QComboBox, QFileDialog, QScrollArea, QAbstractScrollArea, QSplitter
from PyQt5.QtWidgets import QListView, QTreeView, QItemDelegate, QMainWindow, QToolButton, QDialog, QFrame, QListWidgetItem, QListWidget, QMessageBox, QCheckBox, QGroupBox, QSizePolicy, QShortcut, QStyledItemDelegate, QTreeWidget, QTreeWidgetItem
from PyQt5.QtWidgets import QMenu, QAction, QDesktopWidget, QToolTip
from PyQt5.QtCore import QAbstractTableModel, Qt, QSize, QItemSelectionModel, QItemSelection, QTimer, QEvent,QAbstractItemModel
from PyQt5.QtGui import QPixmap, QColor, QIntValidator, QDoubleValidator, QIcon, QValidator,QKeySequence, QWheelEvent, QMouseEvent, QFont
from PyQt5 import QtCore, QtGui
import math
import qimage2ndarray as q2n
import sparse
import matplotlib
from queue import Queue

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from collections import OrderedDict
import typing
from cryovia.gui.shape_drawer import *
from cryovia.cryovia_analysis.custom_utils import resizeMicrograph
from cv2 import circle 
# from cryovia.gui.membrane_segmentation import * 
import pandas as pd
from scipy.ndimage import label

from cryovia.gui.dataset import Dataset, get_all_dataset_names, dataset_factory, get_all_dataset_paths, DEFAULT_CONFIGS
from cryovia.gui.segmentation_files.prep_training_data import load_file
import seaborn as sns
from io import TextIOWrapper, BytesIO
from scipy.ndimage import binary_fill_holes
from grid_edge_detector.image_gui import CorrectDoubleValidator
from cryovia.cryovia_analysis.analyser import Analyser, AnalyserWrapper
import traceback
import copy
import multiprocessing as mp
import datetime
from PIL import Image
try:
    os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
except:
    pass

SHOWN_ATTRIBUTES = ["Circumference", "Diameter","Area", "Shape", "Shape probability", "Closed", "Thickness", "Mean thickness", "Min thickness", "Max thickness", "Min curvature", "Max curvature", "Is probably ice","Is enclosed","Enclosed distance", "Index", "Micrograph"]
NUMBER_OF_COLUMNS = 10
CELL_PADDING = 5
NUMERICAL_VALUES = set(["Circumference", "Diameter","Area","Shape probability", "Thickness", "Min thickness", "Max thickness", "Min curvature", "Max curvature","Is probably ice", "Enclosed distance", "Mean thickness"])
STRING_VALUES = set(["Shape"])
BOOL_VALUE = set(["Closed", "Is enclosed"])



CURRENTLY_RUNNING = set()





def writeMessage(msg):
    print(msg)
MESSAGE = writeMessage


def running(dataset):
    global CURRENTLY_RUNNING
    return dataset in CURRENTLY_RUNNING

class CustomTextIOWrapper(TextIOWrapper):
    def __init__(self, buffer: IO[bytes], worker) -> None:
        super().__init__(buffer)
        self.worker = worker

    def write(self, __s: str) -> int:
        self.worker.progress.emit((__s, 3))
        # return super().write(__s)


class customGroupBox(QGroupBox):
    def __init__(self, title, parent):
        super().__init__(title, parent)

        self.setStyleSheet(
            "QGroupBox::title {subcontrol-origin: margin;left: 7px;padding: 0px 5px 0px 5px; font:bold;}"
            "QGroupBox {"
                # "font: bold;"
                "border: 1px solid silver;"
                "border-radius: 6px;"
                "margin-top: 6px;"
            "}")


class AnalyserWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, in_queue, stopEvent):
        super().__init__()
        # self.datasets = datasets
        # self.configs = configs
        # self.njobs = njobs_threads_dict
        self.in_queue:Queue = in_queue
        self.stopEvent = stopEvent
        self.buffer = CustomTextIOWrapper(BytesIO(),self)




    def run(self):
        try:
            while True:
                try:
                    inp = self.in_queue.get(timeout=5)
                except:
                    if self.stopEvent.is_set():
                        try:
                            self.finished.emit()
                        except:
                            pass
                        return
                    continue
                if isinstance(inp, str) and inp=="STOP":
                    self.finished.emit()
                    return
                dataset, config, njobs = inp
            # for dataset, config in zip(self.datasets, self.configs):
                try:
                    self.progress.emit((dataset.name,0))
                    dataset.run(**njobs, run_kwargs=config, tqdm_file=self.buffer, stopEvent=self.stopEvent)
                    self.progress.emit((dataset.name,1))
                except Exception as e:
                    try:
                        self.progress.emit((traceback.format_exc(),2))
                        self.progress.emit((dataset.name,1))
                    except:
                        pass
        except Exception as e:
            # print(e)
            try:
                self.progress.emit((traceback.format_exc(),2))
            except:
                pass

        try:
            self.finished.emit()
        except:
            pass



class CheckFilesWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, in_queue, stopEvent):
        super().__init__()

        self.in_queue:Queue = in_queue
        self.stopEvent = stopEvent

    def run(self):
        try:
            while True:
                try:
                    inp = self.in_queue.get(timeout=5)
                except:
                    if self.stopEvent.is_set():
                        try:
                            self.finished.emit()
                        except:
                            pass
                        return
                    continue
                if isinstance(inp, str) and inp=="STOP":
                    self.finished.emit()
                    return
                dataset = inp
                try:
                    # self.progress.emit((dataset.name,0))
                    h, m = dataset.isHealthy(self.stopEvent)
                    self.progress.emit(((dataset.name, h, m), 0))
                except Exception as e:
                    try:
                        self.progress.emit((traceback.format_exc(),1))
                    except:
                        pass
        except Exception as e:
            # print(e)
            try:
                self.progress.emit((traceback.format_exc(),1))
            except:
                pass

        try:
            self.finished.emit()
        except:
            pass







def get_analyser(row, dataset):
    if isinstance(row, str):
        micrograph = row
    else:
        micrograph = row["Micrograph"]
    if not isinstance(micrograph, str):
        micrograph = micrograph.iloc[0]
    wrapper = AnalyserWrapper(dataset.dataset_path / micrograph)
    return wrapper



class QBoolValidator(QValidator):
    true_strings = ["True", "true", "Yes", "yes", "1"]
    false_strings = ["False", "false", "No", "no", "0"]

    def validate(self, a0: str, a1: int) -> typing.Tuple['QValidator.State', str, int]:
        l = len(a0)
        
        for bool_value, strings in zip(["True", "False"],[self.true_strings, self.false_strings]):
            for ts in strings:
                if len(ts) < l:
                    continue
                if len(ts) == l and ts == a0:
                    return (QValidator.State.Acceptable, bool_value, len(bool_value))
                if len(ts) > l:
                    if ts[:l] == a0:
                        return (QValidator.State.Intermediate, a0, a1)
        return (QValidator.State.Invalid, "Blub", 0)
        return super().validate(a0, a1)

    def fixup(self, a0: str) -> str:
        # state, s, _ = self.validate(a0, 0)
        l = len(a0)
        for bool_value, strings in zip(["True", "False"],[self.true_strings, self.false_strings]):
            for ts in strings:
                if ts[:l] == a0:
                    return bool_value
        return "False"




class FilterWidget(QWidget):
    def __init__(self, parent, id_) -> None:
        global SHOWN_ATTRIBUTES
        super().__init__()
        self.setParent(parent)
        
        self.id = id_
        self.setLayout(QHBoxLayout())
        
        
        self.attributeDropList = QComboBox()
        self.attributeDropList.addItem("")
        self.attributeDropList.addItems(SHOWN_ATTRIBUTES)

        self.attributeDropList.currentTextChanged.connect(self.attributeChanged)

        self.comparerDropList = QComboBox()

        self.valueLineEdit = QLineEdit()

        self.addFilterButton = QPushButton("+")
        self.addFilterButton.clicked.connect(self.addFilter)
        self.addFilterButton.setToolTip("Add another filter")

        self.removeFilterButton = QPushButton("-")
        self.removeFilterButton.clicked.connect(self.removeMyself)
        self.removeFilterButton.setToolTip("Remove this filter")

        self.layout().addWidget(self.attributeDropList)
        self.layout().addWidget(self.comparerDropList)
        self.layout().addWidget(self.valueLineEdit)
        self.layout().addWidget(self.addFilterButton)
        self.layout().addWidget(self.removeFilterButton)
    

    def attributeChanged(self, attribute):
        global NUMERICAL_VALUES, STRING_VALUES, BOOL_VALUE 
        self.clearAll()
        if attribute == "":
            
            return
        if attribute in NUMERICAL_VALUES:
            self.comparerDropList.addItems([">", "<", "=", "!="])
            self.valueLineEdit.setValidator(QDoubleValidator())
        elif attribute in STRING_VALUES:
            self.comparerDropList.addItems(["=", "!="])
            
        elif attribute in BOOL_VALUE:
            self.comparerDropList.addItems(["is", "is not"])
            self.valueLineEdit.setValidator(QBoolValidator())

    def clearAll(self):
        self.comparerDropList.clear()
        self.valueLineEdit.setValidator(None)
        self.valueLineEdit.setText("")
    
    def mainWindow(self):
        return self.parent().parent().parent().parent().parent().parent().parent().parent()

    def removeMyself(self):
        self.parent().parent().parent().parent().parent().removeFilter(self.id)
    
    def addFilter(self):

        self.parent().parent().parent().parent().parent().addFilter()

    def getValues(self):
        global NUMERICAL_VALUES, STRING_VALUES, BOOL_VALUE 

        returnDict = {}
        returnDict["Attribute"] = self.attributeDropList.currentText()
        returnDict["Comparer"] = self.comparerDropList.currentText()
        if self.valueLineEdit.text() == "":
            value = None
        else:
            if returnDict["Attribute"] in NUMERICAL_VALUES:
                value = float(self.valueLineEdit.text())
            elif returnDict["Attribute"] in STRING_VALUES:
                value = self.valueLineEdit.text()
            else:
                value = self.valueLineEdit.text() == "True"
        returnDict["Value"] = value
        return returnDict


class FilterListWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.filters = {}
        self.number_of_filters = 0
        self.setLayout(QHBoxLayout())
        self.groupBox = customGroupBox("Filter", self)
        self.layout().addWidget(self.groupBox)
        # self.groupBox.setLayout(QHBoxLayout())
        self.initWidget()

        
    
    def initWidget(self):
        listBox = QVBoxLayout()
        listBox.setAlignment(Qt.AlignTop)
        listBox.setSpacing(0)
        
        self.groupBox.setLayout(listBox)

        self.scroll = QScrollArea(self)
        listBox.addWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        self.scrollContent = QWidget(self.scroll)

        scrollLayout = QVBoxLayout(self.scrollContent)
        scrollLayout.setAlignment(Qt.AlignTop)
        scrollLayout.setSpacing(0)
        self.scrollContent.setLayout(scrollLayout)
        self.addFilter()
        # filterWidget = FilterWidget(self, self.number_of_filters)
        # scrollLayout.addWidget(filterWidget)
        # self.filters[self.number_of_filters] = filterWidget
        # self.number_of_filters += 1
        self.scroll.setWidget(self.scrollContent)
    
    def removeFilter(self, id_):
        if id_ in self.filters:
            self.scrollContent.layout().removeWidget(self.filters[id_])
            del self.filters[id_]
        if len(self.filters.keys()) == 0:
            self.addFilter()

    def addFilter(self):
        
        filterWidget = FilterWidget(self, self.number_of_filters)
        self.scrollContent.layout().addWidget(filterWidget)
        self.filters[self.number_of_filters] = filterWidget
        self.number_of_filters += 1
    
    def mainWindow(self):
        return self.parent().parent().parent()

    def applyFilters(self):
        filters = []
        for filterWidget in self.filters.values():
            values = filterWidget.getValues()
            if values["Attribute"] == "" or values["Value"] is None:
                continue
            filters.append(values) 
        self.mainWindow().datasetTabWidget.applyFilters(filters)

    def applyFiltersToAll(self):
        filters = []
        for filterWidget in self.filters.values():
            values = filterWidget.getValues()
            if values["Attribute"] == "" or values["Value"] is None:
                continue
            filters.append(values) 
        self.mainWindow().datasetTabWidget.applyFiltersToAll(filters)



    def clearAll(self):
        ids = list(self.filters.keys())
        for id_ in ids:
            self.scrollContent.layout().removeWidget(self.filters[id_])
            del self.filters[id_]
        self.addFilter()



class PandasModel(QtCore.QAbstractTableModel):
    """
    Class to populate a table view with a pandas dataframe
    """
    def __init__(self, data, dataset, parent=None):
        global SHOWN_ATTRIBUTES
        QtCore.QAbstractTableModel.__init__(self, parent)
        
        data:pd.DataFrame = data.round(6)
        data.set_index(pd.Index([i for i in range(data.shape[0])]))
        self._data:pd.DataFrame = data
        currently_shown_attributes = [attr for attr in SHOWN_ATTRIBUTES if attr in data.columns]
        self.shown_attributes = currently_shown_attributes
        self.shown_data = self._data[self.shown_attributes]
        self.dataset = dataset
    def rowCount(self, parent=None):
        return self.shown_data.shape[0]

    def columnCount(self, parent=None):
        return self.shown_data.shape[1]

    def data(self, index, role=QtCore.Qt.DisplayRole):
        
        if index.isValid():
            if role == QtCore.Qt.DisplayRole:
                shown_data = self.shown_data
                return str(shown_data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == QtCore.Qt.Horizontal and role == QtCore.Qt.DisplayRole:
            return self.shown_data.columns[col]
        return None

    def applyFilters(self, filters):
        to_show = self._data[self.shown_attributes]
        
        for filter in filters:

            if type(filter["Value"]) is float:
                if filter["Comparer"] == ">":
                    to_show = to_show[to_show[filter["Attribute"]] > filter["Value"]]
                elif filter["Comparer"] == "<":
                    to_show = to_show[to_show[filter["Attribute"]] < filter["Value"]]
                elif filter["Comparer"] == "=":
                    to_show = to_show[to_show[filter["Attribute"]] == filter["Value"]]
                elif filter["Comparer"] == "!=":
                    to_show = to_show[to_show[filter["Attribute"]] != filter["Value"]]

            elif type(filter["Value"]) is str:
                if filter["Comparer"] == "=":
                    to_show = to_show[to_show[filter["Attribute"]] == filter["Value"]]
                elif filter["Comparer"] == "!=":
                    to_show = to_show[to_show[filter["Attribute"]] != filter["Value"]]

            elif type(filter["Value"]) is bool:
                if filter["Comparer"] == "is":
                   
                    to_show = to_show[to_show[filter["Attribute"]] == filter["Value"]]
                elif filter["Comparer"] == "is not":
                   
                    to_show = to_show[to_show[filter["Attribute"]] != filter["Value"]]

        self.shown_data = to_show
        self.layoutChanged.emit()
    
    def sort(self, columnIdx, ascending):

        self.shown_data = self.shown_data.sort_values(self.shown_data.columns[columnIdx], ascending=ascending)
        self.layoutChanged.emit()

class ListFilterButtons(QWidget):
    def __init__(self, parent, filterwidget):
        super().__init__(parent)
        self.filterwidget = filterwidget
        self.setLayout(QHBoxLayout())

        self.applyButton = QPushButton("Apply")
        self.applyButton.clicked.connect(self.filterwidget.applyFilters)
        self.applyButton.setToolTip("Apply all filters to the current tab.")

        self.applyToAllButton = QPushButton("Apply to all tabs")
        self.applyToAllButton.clicked.connect(self.filterwidget.applyFiltersToAll)
        self.applyToAllButton.setToolTip("Apply all filters to all tabs.")

        self.clearAllButton = QPushButton("Clear all filters")
        self.clearAllButton.clicked.connect(self.filterwidget.clearAll)
        self.clearAllButton.setToolTip("Remove all filters.")

        self.layout().addWidget(self.applyButton)
        self.layout().addWidget(self.applyToAllButton)

        self.layout().addWidget(self.clearAllButton)


class CustomQTableView(QTableView):
    def __init__(self, parent, name):
        self.name = name
        super().__init__(parent)
        self.horizontalHeader().sectionClicked.connect(self.onHeaderClicked)
        self.headerClicked = None
        self.headerClickedCount = 0

    def mainWindow(self):
        return self.parent().parent().parent().parent()

    def closeEvent(self, a0) -> None:
        self.mainWindow().graphWidget.membraneGraphs.redrawData()
        return super().closeEvent(a0)
    
    def onHeaderClicked(self, logicalIndex):
        if self.headerClicked is None or self.headerClicked != logicalIndex:
            self.headerClicked = logicalIndex
            self.headerClickedCount = 0
        else:
            self.headerClickedCount = (self.headerClickedCount + 1) % 2
        
        self.model().sort(self.headerClicked, self.headerClickedCount==0)




    def deleteRow(self):

        idxs = self.selectedIndexes()
        if len(idxs) > 0:
            ret = self.parent().parent().deleteResult
            if ret is None:
                qm = QMessageBox()
                qm.setText(f"Remove data.")
                qm.setIcon(qm.Icon.Warning)
                qm.addButton(QPushButton("Remove in current table"),  QMessageBox.ButtonRole.YesRole)

                qm.addButton(QPushButton("Remove from csv file"), QMessageBox.ButtonRole.AcceptRole)
                # qm.addButton(QPushButton("Remove for future analysis"), QMessageBox.ButtonRole.ActionRole)
                qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
                # qm.setCheckBox()
                cb = QCheckBox("Do not ask again this session.")
                qm.setCheckBox(cb)
                
                ret = qm.exec_()
                
                if cb.isChecked() and ret != 2:
                    self.parent().parent().deleteResult = ret
                if ret == 2:
                    return

            dataset = self.parent().parent().datasetTabs[self.name][2]
        
            data:pd.DataFrame = self.model().shown_data
            rows = [idx.row() for idx in idxs]
            
            idx = data.iloc[rows].index

            self.removeIdxs(idx, dataset, ret)
            # all_data: pd.DataFrame = self.model()._data
            # self.model()._data = all_data.drop(index=idx)
            # self.model().shown_data = data.drop(index=idx)
            

            # self.model().layoutChanged.emit()

            # self.parent().parent().tableClicked()

            # if ret == 1:
            #     dataset.to_csv(csv=self.model()._data)
    def removeIdxs(self, idx, dataset, ret=0, ):
        if running(dataset.name):
            global MESSAGE
            MESSAGE(f"Cannot remove from {dataset.name} because analysis currently running.\n")
        data:pd.DataFrame = self.model().shown_data

        all_data: pd.DataFrame = self.model()._data
        self.model()._data = all_data.drop(index=idx)
        self.model().shown_data = data.drop(index=idx)
        

        self.model().layoutChanged.emit()

        self.parent().parent().tableClicked()

        if ret == 1:
            dataset.to_csv(csv=self.model()._data)


class DatasetTabsWidget(QTabWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setTabsClosable(True)
        self.setMovable(True)
        self.tabBar().tabMoved.connect(self.parent().graphWidget.membraneGraphs.redrawData)
        self.datasetTabs = {}

        self.tabCloseRequested.connect(lambda index: self.removeTab(index))
        self.deleteResult = None
    


    def printStats(self):
        tableview:CustomQTableView = self.currentWidget()
        
        if tableview is None:
            return
        index = tableview.selectedIndexes()
        if len(index) == 0:
            MESSAGE("Select a column to get stats.")
            return
        index = index[-1]
        row = index.row()
        column_idx = index.column()
        name = self.tabText(self.currentIndex())
        for name, (tableview,_, dataset) in self.datasetTabs.items():
            
        
            data:pd.DataFrame = tableview.model().shown_data
            column = data.columns[column_idx]
            std = data[column].std()
            mean = data[column].mean()
            min_ = data[column].min()
            max_ = data[column].max()
            median = np.median(data[column])
            median_absolute_deviation = np.median(np.abs(data[column] - np.median(data[column])))
            entries = len(data[column])
            text = f"{name}: {column}\nNumber of membranes: {entries}\nMinimum: {min_}\nMaximum: {max_}\nMean: {mean}\nStd: {std}\nMedian: {median}\nMedian absolute deviation: {median_absolute_deviation}\n"
            MESSAGE(text)


    def addDatasetTab(self, dataset):
        
        if dataset.name in self.datasetTabs:
            return
        df:pd.DataFrame = dataset.csv
        
        
        tableview = CustomQTableView(self, dataset.name) 
        
        # tableview.clicked.connect(self.tableClicked)

        sc = QShortcut(QKeySequence(QKeySequence.Delete), tableview)
        sc.activated.connect(tableview.deleteRow) 

        model = PandasModel(df, dataset)
        tableview.setModel(model)
        self.addTab(tableview, dataset.name)
        self.datasetTabs[dataset.name]  = (tableview, model, dataset)

        # tableview.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tableview.selectionModel().selectionChanged.connect(self.tableClicked)

    def applyFilters(self, filters):
        tableview = self.currentWidget()
        if tableview is None:
            return
        tableview.model().applyFilters(filters)
        
    def tableClicked(self):
        global MESSAGE
        

        tableview:CustomQTableView = self.currentWidget()
        
        if tableview is None:
            return
        index = tableview.selectedIndexes()
        if len(index) == 0:
            return
        index = index[-1]
        row = index.row()
        column = index.column()
        name = self.tabText(self.currentIndex())

        dataset = self.datasetTabs[name][2]
        if dataset.isZipped:
            MESSAGE(f"Cannot show information about this membrane because dataset {dataset.name} is still zipped.")
            return
        
        data:pd.DataFrame = tableview.model().shown_data
        try:
            idx = data.iloc[[row]].index
        except:
            return
        # data:pd.Series = data.iloc[row]
        dataset_row = tableview.model()._data.filter(idx, axis=0)
        wrapper = get_analyser(dataset_row, dataset)

        
        
        min_thickness = tableview.model()._data["Min thickness"].min(skipna=True)
        max_thickness = tableview.model()._data["Max thickness"].max(skipna=True)
        min_curvature = tableview.model()._data["Min curvature"].min(skipna=True)
        max_curvature = tableview.model()._data["Max curvature"].max(skipna=True)

        self.mainWindow().membraneWidget.loadImages(wrapper, dataset_row["Index"].iloc[0], min_thickness, max_thickness, min_curvature, max_curvature)
        column = data.columns[column]

        get_all = self.mainWindow().graphWidget.membraneGraphs.showAllCheckbox.isChecked()
        data = self.getCurrentData(all=get_all, column=column)
     
        plot_type = "violin"
        if data.dtypes[column] == "O" or  data.dtypes[column] == "bool":
            plot_type = "bar"
        
        self.mainWindow().graphWidget.membraneGraphs.showData(data, column, plot_type)

    def mainWindow(self):
        return self.parent().parent().parent().parent().parent()
    
    def applyFiltersToAll(self, filters):
        for idx in range(self.count()):
            tableview = self.widget(idx)
            tableview.model().applyFilters(filters)

    def getAllTabs(self):
        return [self.widget(idx) for idx in range(self.count())]


    def removeTab(self, index: int) -> None:
        name = self.tabText(index)
        del self.datasetTabs[name]
        return super().removeTab(index)

    def getCurrentData(self, all=False, column=None):
        if not all:
            tableview = self.currentWidget()
            
            if tableview is None:
                return
            data:pd.DataFrame = tableview.model().shown_data
            name = self.tabText(self.currentIndex())
            if column is not None:
                data = data[column]
                data = data[data.notna()]
                data = pd.DataFrame(data)
            data["dataset"] = name

            return data
        else:
            data = None
            for idx in range(self.count()):
                tableview = self.widget(idx)
                name = self.tabText(idx)
                current_data = tableview.model().shown_data
                if column is not None:
                    current_data = current_data[column]
                    current_data = current_data[current_data.notna()]
                    current_data = pd.DataFrame(current_data)
                current_data["dataset"] = name
                if data is None:
                    data =current_data
                else:
                    data = pd.concat((data, current_data))
            return data

    def exportData(self, all=False):
        if not all:
            tableview = self.currentWidget()
            
            if tableview is None:
                return
            tableviews = [tableview]
        else:
            tableviews = [self.widget(idx) for idx in range(self.count())]
            if len(tableviews) == 0:
                return
        

        filename,filt = QFileDialog.getSaveFileName(self, "Save file", ".", "CSV (*.csv)" )
        if filename is None or len(Path(filename).stem) == 0:
            return
       

        
        data = None
        for tableview in tableviews:
            curreunt_data = copy.deepcopy(tableview.model().shown_data)
            curreunt_data["Dataset"] = tableview.model().dataset.name
            if data is None:
                data = curreunt_data
            else:
                data = pd.concat((data, curreunt_data))
        
        data.to_csv(filename)
    
    def exportAllData(self):
        self.exportData(all=True)


class DatasetLabelDelegate(QItemDelegate):
    def createEditor(self, parent, option: 'QStyleOptionViewItem', index) -> QWidget:
        label = self.parent().item(index.row())
        def editingFinishedFunc():
            if editor.isModified():
                return
            if len(editor.text()) == 0:
                label.setText(label.dataset.name)
                return
            try:
                if running(label.dataset.name):
                    raise ValueError()
                label.dataset.rename(editor.text())
                editor.setModified(True)
            except FileExistsError as e:
                label.setText(label.dataset.name)
                pass
            except ValueError as e:
                label.setText(label.dataset.name)
        editor = QLineEdit(parent)
        parent:DatasetListWidget
        editor.editingFinished.connect(editingFinishedFunc)
        return editor 



class DoubleClickLineEdit(QLineEdit):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setReadOnly(True)
        self.setFrame(False)
        self.setStyleSheet("QLineEdit { border: none; }")  # Remove the border
        self.editingFinished.connect(self.editing_Finished)
        
        # self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def mousePressEvent(self, event):
        self.parent().parent().parent().selectedThisDataset(self.parent().dataset.name)

    def mouseDoubleClickEvent(self, event):
        # self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setReadOnly(False)
        self.setFocus()
        self.selectAll()


    def editing_Finished(self , **kwargs):
        
        if not self.isModified():
            # self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            return
        if len(self.text()) == 0:
            self.setText(self.parent().dataset.name)
            # self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            return
        try:
            if running(self.parent().dataset.name):
                raise ValueError()
            self.parent().dataset.rename(self.text())
            self.parent().parent().parent().loadDatasets()
            self.setModified(True)
        except FileExistsError as e:
            self.setText(self.parent().dataset.name)
            pass
        except ValueError as e:
            self.setText(self.parent().dataset.name)

        self.setReadOnly(True)
        # self.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        # self.super().editingFinished(**kwargs)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.setReadOnly(True)
    



    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("background-color: lightblue;")
        else:
            self.setStyleSheet("")


# class EditableLabel(QLabel):
#     def __init__(self, text, parent=None):
#         super().__init__(text, parent)
#         self.line_edit = QLineEdit()
#         self.line_edit.setText(text)
#         self.line_edit.setVisible(False)
#         self.line_edit.setFrame(False)
#         self.line_edit.editingFinished.connect(self.finish_editing)
#         self.line_edit.setFont(QFont(self.font().family(), self.font().pointSize()))

#     def resizeEvent(self, event):
#         super().resizeEvent(event)
#         self.line_edit.setGeometry(self.rect())

#     def mouseDoubleClickEvent(self, event):
#         self.start_editing()

#     def start_editing(self):
#         self.line_edit.setText(self.text())
#         # self.setVisible(False)
#         self.line_edit.setVisible(True)
#         self.line_edit.setFocus()
#         self.line_edit.selectAll()
#         print(self.isVisible(), self.line_edit.isVisible())

#     def finish_editing(self):
#         print(self.isVisible(), self.line_edit.isVisible())
#         self.setText(self.line_edit.text())
#         self.setVisible(True)
#         self.line_edit.setVisible(False)


# class EditableLabel(QLabel):
#     def __init__(self, text, parent=None):
#         super().__init__(text, parent)
#         self.line_edit = QLineEdit(self)
#         self.line_edit.setText(text)
#         self.line_edit.setVisible(False)
#         self.line_edit.setFrame(False)
#         self.line_edit.editingFinished.connect(self.finish_editing)
#         self.line_edit.setFont(QFont(self.font().family(), self.font().pointSize()))

#     def mouseDoubleClickEvent(self, event):
#         self.start_editing()

#     def start_editing(self):
#         self.line_edit.setText(self.text())
#         self.setVisible(False)
#         self.line_edit.setVisible(True)
#         self.line_edit.setFocus()
#         self.line_edit.selectAll()
#         print(self.isVisible(), self.line_edit.isVisible())

#     def finish_editing(self):
#         print(self.isVisible(), self.line_edit.isVisible())
#         self.setText(self.line_edit.text())
#         self.setVisible(True)
#         self.line_edit.setVisible(False)






class DatasetLabelWithIcon(QWidget):
    def __init__(self, dataset, isHealthy, missingFiles={}):
        super().__init__()
        self.dataset = dataset
        
        tooltip = self.setColorAndGetTooltip(isHealthy, missingFiles)
        # Create a horizontal layout    
        layout = QHBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(2)
        # Create a label for the text and add it to the layout
        self.text_label = DoubleClickLineEdit(str(self.dataset.name))
        layout.addWidget(self.text_label)

        # Create a label for the pixmap and add it to the layout
        self.pixmap_label = QLabel()
        
        pixmap = QPixmap(12, 12)  # Set the size of the pixmap
        pixmap.fill(QColor('transparent'))  # Fill it with transparent color
        
        # Create a QPainter to draw on the QPixmap
        painter = QPainter(pixmap)
        painter.setBrush(QColor(self.color))
        painter.setPen(QColor(self.color))
        painter.drawEllipse(0, 0, 12, 12)  # Draw the colored circle
        painter.end()

        self.pixmap_label.setPixmap(pixmap)
        self.pixmap_label.setToolTip(tooltip)
        layout.addWidget(self.pixmap_label, alignment=Qt.AlignRight)

        # Set the layout to the custom widget
        self.setLayout(layout)


    
    def setColorAndGetTooltip(self, isHealthy, missingFiles):

        "micrographs", "segmentations", "analysers"
        if isHealthy is None:
            self.color = "yellow"
            tooltip = "Checking if all files exist."
        else:
            if isHealthy:
                self.color = "green"
                tooltip = "All files still exist. You can run analysis on this dataset."
            else:
                self.color = "red"
                
                tooltip = "Files are missing."
                if missingFiles["micrographs"] > 0:
                    mf = missingFiles["micrographs"]
                    tooltip += f"\n{mf} micrographs are missing. You can not run this dataset anymore. "
                if missingFiles["segmentations"] > 0:
                    mf = missingFiles["segmentations"]
                    tooltip += f"\n{mf} segmentations are missing."
                    if missingFiles["micrographs"] > 0:
                        pass
                    else:
                        tooltip += "You can run the analysis if you rerun segmentation."
                if missingFiles["analysers"] > 0:
                    mf = missingFiles["analysers"]
                    tooltip += f"\n{mf} internal files are missing. These will be recreated if you run the analysis."
                if missingFiles["csv"] == 0:
                    tooltip += "\nYou can still look at the result output."
            
        return tooltip
    

    def redraw(self, isHealthy, missingFiles):
        tooltip = self.setColorAndGetTooltip(isHealthy, missingFiles)

        pixmap = QPixmap(12, 12)  # Set the size of the pixmap
        pixmap.fill(QColor('transparent'))  # Fill it with transparent color
        
        # Create a QPainter to draw on the QPixmap
        painter = QPainter(pixmap)
        painter.setBrush(QColor(self.color))
        painter.setPen(QColor(self.color))
        painter.drawEllipse(0, 0, 12, 12)  # Draw the colored circle
        painter.end()

        self.pixmap_label.setPixmap(pixmap)
        self.pixmap_label.setToolTip(tooltip)

    # def mouseDoubleClickEvent(self, event):
    #     self.text_label.startEditing()

    def set_selected(self, selected):
        if selected:
            self.setStyleSheet("background-color: lightblue;")
        else:
            self.setStyleSheet("")

    def __repr__(self):
        return self.dataset.name


class DatasetLabel(QListWidgetItem):
    def __init__(self,dataset, **kwargs) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        
        self.setText(str(self.dataset.name))
        
        # self.classifier.changed_hooks.append(self.reset_name)
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
        
        

    def __repr__(self) -> str:
        return self.dataset.name



# class DatasetLabel(QListWidgetItem):
#     def __init__(self, dataset, *args):
#         super().__init__(*args)
#         self.dataset = dataset
#         self.setReadOnly(True)
        

# class CircleIconWidget(QLabel):
#     def __init__(self, color="green"):
#         super().__init__()
#         self.color = color
#         self.initUI()

#     def initUI(self):
#         # Create a QPixmap to draw the green circle
#         pixmap = QPixmap(12, 12)  # Set the size of the pixmap
#         pixmap.fill(QColor('transparent'))  # Fill it with transparent color
        
#         # Create a QPainter to draw on the QPixmap
#         painter = QPainter(pixmap)
#         painter.setBrush(QColor(self.color))
#         painter.setPen(QColor(self.color))
#         painter.drawEllipse(0, 0, 12, 12)  # Draw the colored circle
#         painter.end()

#         self.setPixmap(pixmap)

#         self.setToolTip('Green Circle Icon')
        
class DatasetListWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setItemDelegate(DatasetLabelDelegate(self))
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.healthCheck = {}
        self.datasets = {}
        self.datasetsWidgets = {}
        self.startWorker()

        self.loadDatasets()
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.itemSelectionChanged.connect(self.showInfo)
        
        # self.itemDoubleClicked.connect(self.lookUpItem)



    def selectedThisDataset(self, dataset):
        self.setCurrentItem(self.datasetsWidgets[dataset])
    
    def lookUpItem(self, item):
        self.mainWindow().datasetTabWidget.addDatasetTab(self.itemWidget(item).dataset)


    def loadDatasets(self):
        self.clear()
        self.datasets = {}
        self.datasetsWidgets = {}
        datasets = get_all_dataset_paths()
        datasets = [Dataset.load(dataset) for dataset in datasets]
        for dataset in datasets:
            dummy = QListWidgetItem(self)
            health = None
            missingFiles = {}
            if dataset.name in self.healthCheck:
                health, missingFiles = self.healthCheck[dataset.name]
            label = DatasetLabelWithIcon(dataset, health, missingFiles)
            dummy.setSizeHint(label.sizeHint())
            self.setItemWidget(dummy, label)
            self.datasets[dataset.name] = label
            self.datasetsWidgets[dataset.name] = dummy
            self.queue.put(dataset)
        datasets_to_del = []
        for key, value in self.healthCheck.items():
            if key not in self.datasets:
                datasets_to_del.append(key)
        for key in datasets_to_del:
            del self.healthCheck[key]

    def startWorker(self):
        thread = QThread()
        queue = Queue()
        stopEvent = mp.get_context("spawn").Event()
        worker = CheckFilesWorker(queue, stopEvent)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.finishedRunning)
        worker.progress.connect(self.progressEmited)
        
        thread.start()   
        self.currentThread = thread
        self.worker = worker
        self.queue = queue
        self.stopEvent = stopEvent

    def performHealthCheck(self, dataset):
        if dataset in self.datasets:
            return
        self.queue.put(dataset)
    
    def stopThread(self, keep_running=False):
        global MESSAGE
        if self.currentThread is not None:
            if keep_running:
                self.queue.put("STOP")
            else:

                self.stopEvent.set()
                MESSAGE("Waiting until analysis is done or thread is closed.")
            
            # while self.currentThread.isRunning():
            #     time.sleep(0.5)


    def finishedRunning(self):
        pass

    def progressEmited(self, emit):
        global CURRENTLY_RUNNING, MESSAGE
        result, state = emit
        if state == 0:
            dataset, isHealthy, missingFiles = result
            if dataset in self.datasets:
                self.datasets[dataset].redraw(isHealthy, missingFiles)
            

        elif state == 1:
            
            MESSAGE(result)
        elif state == 3:
            MESSAGE(result)


    def on_selection_changed(self):
        for i in range(self.count()):
            item = self.item(i)
            custom_widget = self.itemWidget(item)
            if custom_widget:
                custom_widget.text_label.set_selected(item.isSelected())
                custom_widget.set_selected(item.isSelected())


    def mainWindow(self):
        return self.parent().parent().parent().parent().parent().parent()

    def showInfo(self, ):
        selection = self.selectedItems()
        infowidget = self.mainWindow().datasetInfoWidget
        if len(selection) != 1:
            infowidget.resetInfos()
            return
        for item in selection:
            item = self.itemWidget(item)
            
            dataset = item.dataset
            infowidget.loadDatasetInfo(dataset)

class DatasetInfoWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.setLayout(QGridLayout())

        self.initUI()
    
    def initUI(self):


        self.numberOfMicrographsLabelLabelLayout = QVBoxLayout()
        self.numberOfMicrographsLabel = QLabel("# Micrographs\n ")
        self.numberOfMicrographsLabelLabelLayout.addWidget(self.numberOfMicrographsLabel)


        self.numberOfMicrographsLabelAddButton = QPushButton("Add micrographs")
        self.numberOfMicrographsLabelAddButton.setToolTip("Add micrographs to the dataset")
        self.numberOfMicrographsLabelAddButton.clicked.connect(self.addMicrographs)

        self.numberOfMicrographsLabelLoadCsvButton = QPushButton("Load in CSV")
        self.numberOfMicrographsLabelLoadCsvButton.setToolTip("Load in a csv file with micrograph paths in first column and optional segmentation paths in second. It should have no header.")
        self.numberOfMicrographsLabelLoadCsvButton.clicked.connect(self.loadCsv)

        self.numberOfMicrographsLabelRemoveButton = QPushButton("Remove all micrographs")
        self.numberOfMicrographsLabelRemoveButton.setToolTip("Remove all micrographs from the dataset")
        # self.numberOfMicrographsLabelRemoveButton.setIcon(self.numberOfMicrographsLabelRemoveButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.numberOfMicrographsLabelRemoveButton.clicked.connect(self.removeMicrographs)
        # self.numberOfMicrographsLabelLabel = QLabel("")
        # self.numberOfMicrographsLabelLabelLayout.addWidget(self.numberOfMicrographsLabelLabel, Qt.AlignCenter)

        # self.numberOfMicrographsOnlyLabelLayout = QHboxLayout()
        self.numberOfMicrographsLabelLayout = QHBoxLayout()
        self.numberOfMicrographsChangeLayout= QVBoxLayout()
        # self.numberOfMicrographsLabelLayout.addWidget(self.numberOfMicrographsLabelLabel)
        self.numberOfMicrographsLabelLayout.addLayout(self.numberOfMicrographsChangeLayout)
        self.numberOfMicrographsChangeLayout.addWidget(self.numberOfMicrographsLabelAddButton)
        self.numberOfMicrographsChangeLayout.addWidget(self.numberOfMicrographsLabelLoadCsvButton)
        self.numberOfMicrographsChangeLayout.addWidget(self.numberOfMicrographsLabelRemoveButton)


        

        self.currentlyRunning = QLabel("")
        self.placeholder = QWidget()

        self.isZippedLabel = QLabel("")
        self.compressButton = QPushButton("Zip")
        self.compressButton.clicked.connect(self.compressDataset)

        self.zipLayout = QHBoxLayout()
        self.zipLayout.addWidget(self.isZippedLabel)
        self.zipLayout.addWidget(self.compressButton)


        self.layout().addWidget(self.placeholder,0,0)#
        self.layout().addLayout(self.numberOfMicrographsLabelLabelLayout, 1,0)
        # self.layout().addWidget(self.numberOfMicrographsLabelLabel, 2,0)

        self.layout().addLayout(self.numberOfMicrographsLabelLayout,1 ,1 )
        self.timesLabels = {}
        for counter, key in enumerate(["Created", "Last changed", "Last run"]):
            self.timesLabels[key] = QLabel(f"{key}: ")
            self.layout().addWidget(self.timesLabels[key], 2+counter, 0, 1,2)
        self.layout().addLayout(self.zipLayout, 5,0,1,2)
        self.layout().addWidget(self.currentlyRunning, 6,0,1,2)
        
        self.layout().setRowStretch(0,1)


    def compressDataset(self):
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        if len(selection) != 1:
            return
        
        dataset = self.mainWindow().datasetListWidget.listWidget.itemWidget(selection[0]).dataset

        if dataset.isZipped:
            
            dataset.unzip()
        else:
            qm = QMessageBox()
            qm.setText(f"You are about to zip the dataset \"{dataset.name}\".\nThis can help you save space as well as reduce the number of files.\nBut it probably takes a few minutes depending on the\n size of the dataset.\n"
                       "To access most of the functions for this dataset again, you have to unzip it.")
            qm.setIcon(qm.Icon.Warning)
            qm.addButton(QPushButton("Zip dataset"), QMessageBox.ButtonRole.YesRole)

            qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
            ret = qm.exec_()

            
            if ret == 1:
                return
            elif ret == 0:
                
                dataset.zip()
        self.loadDatasetInfo(dataset)


    def loadDatasetInfo(self, dataset):

        self.numberOfMicrographsLabel.setText(f"# Micrographs\n    {len(dataset)}")
        # self.numberOfMicrographsLabelLabel.setText(str(len(dataset)))
        for key, value in dataset.times.items():
            self.timesLabels[key].setText(f"{key}: {value}")
        if running(dataset.name):
            self.currentlyRunning.setText("Currently running analysis")
        else:
            self.currentlyRunning.setText("Not running")
        if dataset.isZipped:
            self.isZippedLabel.setText("Compressed")
            self.compressButton.setText("Unzip")
        else:
            self.isZippedLabel.setText("Uncompressed")
            self.compressButton.setText("Zip")

    
    def resetInfos(self):
        self.numberOfMicrographsLabel.setText("# Micrographs\n ")
        # self.numberOfMicrographsLabelLabel.setText("")
        self.currentlyRunning.setText("")
        self.isZippedLabel.setText("")
        for key, value in self.timesLabels.items():
            value.setText(f"{key}: ")


    def getApplyInfoCombobox(self, combobox):
        def applyInfoCombobox():
            if combobox.hasFocus():
                self.applyInfo()
        return applyInfoCombobox

    def mainWindow(self):
        return self.parent().parent().parent().parent()

    def loadCsv(self):
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        if len(selection) != 1:
            return
        dataset = self.mainWindow().datasetListWidget.listWidget.itemWidget(selection[0]).dataset
        dlg = QFileDialog()

        file_suffixes = " *.csv"
        file_suffixes = f"CSV-File (*{file_suffixes})"
        dlg.setFileMode(QFileDialog.ExistingFile)
        file, filt = dlg.getOpenFileName(self, "Choose csv file", ".",filter=file_suffixes)
        if len(file) == 0 or file is None:
            return
        dataset.loadMicrographCsv(file)

        nonMrcFiles = [file for file in dataset.micrograph_paths if Path(file).suffix != ".mrc" and file not in dataset.pixelSizes]
        if len(nonMrcFiles) > 0:
            pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
            result = pixelSizeWidget.exec()
            if result == 0:
                ps = 1
            ps = pixelSizeWidget.getPixelSize()
            
        for file in nonMrcFiles:
            dataset.pixelSizes[file] = ps 

        dataset.save()
        self.loadDatasetInfo(dataset)
        
        # nonMrcFiles = [file for file in files if Path(file).suffix != ".mrc" and file not in dataset.pixelSizes]
        # if len(nonMrcFiles) > 0:
        #     pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
        #     result = pixelSizeWidget.exec()
        #     if result == 0:
        #         return
        #     ps = pixelSizeWidget.getPixelSize()
            
        # dataset.addMicrographPaths(files)
        # for file in nonMrcFiles:
        #     dataset.pixelSizes[file] = ps 
        # dataset.save()
        # self.loadDatasetInfo(dataset)

    def removeMicrographs(self):
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        if len(selection) != 1:
            return
        dataset = self.mainWindow().datasetListWidget.listWidget.itemWidget(selection[0]).dataset
        dataset.removeMicrographPaths(all=True)
        self.loadDatasetInfo(dataset)


    def applyInfo(self):
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        if len(selection) != 1:
            return
        dataset = self.mainWindow().datasetListWidget.listWidget.itemWidget(selection[0]).dataset
        self.loadDatasetInfo(dataset)


    def addMicrographs(self):
        
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        if len(selection) != 1:
            return
        dataset = self.mainWindow().datasetListWidget.listWidget.itemWidget(selection[0]).dataset
        dlg = QFileDialog()

        file_suffixes = " *".join([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
        file_suffixes = f"Micrographs (*{file_suffixes})"
        dlg.setFileMode(QFileDialog.ExistingFiles)
        files, filt = dlg.getOpenFileNames(self, "Choose micrographs", ".",filter=file_suffixes)

        if len(files) == 0:
            return
        nonMrcFiles = [file for file in files if Path(file).suffix != ".mrc" and file not in dataset.pixelSizes]
        if len(nonMrcFiles) > 0:
            pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
            result = pixelSizeWidget.exec()
            if result == 0:
                return
            ps = pixelSizeWidget.getPixelSize()
            
        dataset.addMicrographPaths(files)
        for file in nonMrcFiles:
            dataset.pixelSizes[file] = ps 
        dataset.save()
        self.loadDatasetInfo(dataset)


class DatasetButtonWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):

        
        # Create buttons
        createNewDatasetButton = QPushButton('New', self)
        createNewDatasetButton.setToolTip("Create a new dataset.")
        
        createNewDatasetButton.clicked.connect(self.createNewDatset)

        copyDatasetButton = QPushButton("Copy", self)
        copyDatasetButton.setToolTip("Copy the currently selected dataset. This may take some time while data is being copied.")
        copyDatasetButton.clicked.connect(self.copyDataset)
        
        removeDatasetButton = QPushButton("Remove", self)
        removeDatasetButton.setToolTip("Remove this dataset.")
        # removeDatasetButton.setIcon(removeDatasetButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        removeDatasetButton.clicked.connect(self.removeDataset)
        
        lookupButton = QPushButton("Data >>")
        lookupButton.setToolTip("Look at the result from this dataset in the table.")
        lookupButton.clicked.connect(self.lookup)

        inspectDatasetWindowButton = MicrographInspectionButton("Inspect")
        # inspectDatasetWindowButton.setToolTip("")
        # inspectDatasetWindowButton.clicked.connect(self.inspect)

        # Create layout
        layout = QVBoxLayout()
        self.upperLayout = QHBoxLayout()
        self.lowerLayout = QHBoxLayout()
        self.upperLayout.addWidget(createNewDatasetButton)
        self.upperLayout.addWidget(copyDatasetButton)
        self.upperLayout.addWidget(removeDatasetButton)
        self.lowerLayout.addWidget(inspectDatasetWindowButton)
        self.lowerLayout.addWidget(lookupButton)

        layout.addLayout(self.upperLayout)
        layout.addLayout(self.lowerLayout)
        # Set layout
        self.setLayout(layout)


    def copyDataset(self):
        global MESSAGE
        listwidget : DatasetListWidget= self.parent().listWidget
        items = listwidget.selectedItems()
        if len(items) == 1:
            
            dataset =  listwidget.itemWidget(items[0]).dataset
            save_dir = dataset.dataset_path.parent
            if dataset.isZipped:
                MESSAGE(f"Cannot copy dataset {dataset.name} because it is still zipped.")
                return
            if running(dataset.name):
                MESSAGE("Cannot copy dataset because analysis is currently running\n")
                return
            dataset.copy(save_dir)
            self.parent().listWidget.loadDatasets()

    def createNewDatset(self):
        dataset_counter = 0
        datasets = get_all_dataset_names()
        while True:
            new_dataset_name = f"New_dataset_{dataset_counter}"
            if new_dataset_name not in datasets:
                break
            dataset_counter += 1
        dlg = QFileDialog()

        dlg.setFileMode(QFileDialog.ExistingFiles)
        folder = dlg.getExistingDirectory(self, "Choose dataset directory", ".")
        if folder is None or len(folder) == 0:
            return
        folder = Path(folder)
        new_dataset = Dataset(new_dataset_name, folder)
        self.parent().listWidget.loadDatasets()
        
    def removeDataset(self):
        listwidget : DatasetListWidget= self.parent().listWidget
        items = listwidget.selectedItems()
        if len(items) == 1:
            dataset = listwidget.itemWidget(items[0]).dataset
            if running(dataset.name):
                global MESSAGE

                MESSAGE("Cannot remove dataset because analysis is currently running\n")
                return

            qm = QMessageBox()
            qm.setText(f"You are about to remove the dataset \"{dataset.name}\".")
            qm.setIcon(qm.Icon.Warning)
            qm.addButton(QPushButton("Delete dataset"), QMessageBox.ButtonRole.YesRole)
            qm.addButton(QPushButton("Delete dataset and all data"), QMessageBox.ButtonRole.AcceptRole)
            qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
            ret = qm.exec_()

            if ret == 2:
                return
            elif ret == 0:
                dataset.remove(remove_data=False)
            elif ret == 1:
                dataset.remove(remove_data=True)

            # ret = qm.question(self,'', f"You are about to remove the dataset \"{dataset.name}\".",)
        elif len(items) > 1:
            do_for_all = False
            do_for_all_ret = None
            for counter, item in enumerate(items):
                dataset = listwidget.itemWidget(item).dataset
                if do_for_all:
                    ret = do_for_all_ret
                else:
                    qm = QMessageBox()
                    checkbox = QCheckBox("Apply to all?")
                    qm.setCheckBox(checkbox)
                    qm.setText(f"You are about to remove the dataset \"{dataset.name}\" ({counter+1}/{len(items)}).")
                    qm.setIcon(qm.Icon.Warning)
                    qm.addButton(QPushButton("Delete dataset"), QMessageBox.ButtonRole.YesRole)
                    qm.addButton(QPushButton("Delete dataset and all data"), QMessageBox.ButtonRole.AcceptRole)
                    qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
                    ret = qm.exec_()
                    if checkbox.isChecked():
                        do_for_all = True
                        do_for_all_ret = ret

                if ret == 2:
                    return
                elif ret == 0:
                    dataset.remove(remove_data=False)
                elif ret == 1:
                    dataset.remove(remove_data=True)


        self.parent().listWidget.loadDatasets()

    def lookup(self):
        listwidget : DatasetListWidget= self.parent().listWidget
        items = listwidget.selectedItems()
        for item in items:
            dataset = listwidget.itemWidget(item).dataset
            self.mainWindow().datasetTabWidget.addDatasetTab(dataset)
        self.mainWindow().graphWidget.membraneGraphs.redrawData()

    def mainWindow(self):
        return self.parent().parent().parent().parent().parent().parent()


class DatasetsListPlusButtons(QWidget):
    def __init__(self, parent: QWidget ):
        super().__init__(parent)

        self.setLayout(QVBoxLayout())
        self.listWidget = DatasetListWidget(self)

        self.buttonWidget = DatasetButtonWidget(self)

        self.layout().addWidget(self.listWidget)
        self.layout().addWidget(self.buttonWidget)







class DefaultFloatValidator(QDoubleValidator):
    def __init__(self, default):
        super().__init__()
        self.default = default
    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        return super().fixup(a0)

class DefaultIntValidator(QIntValidator):
    def __init__(self, default):
        super().__init__()
        self.default = default
    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        return super().fixup(a0)
    

class DefaultBoolValidator(QBoolValidator):
    def __init__(self, default):
        super().__init__()
        self.default = default
    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        return super().fixup(a0)

class QLabelWithValidator(QLineEdit):
    def __init__(self, text, type_, default):
        super().__init__(text)
        self.type_ = type_
        self.default = default
        if type_ is float:
            self.setValidator(DefaultFloatValidator(default))
        elif type_ is int:
            self.setValidator(DefaultIntValidator(default))
        elif type_ is bool:
            self.setValidator(DefaultBoolValidator(default))
        else:
            raise NotImplementedError(f"Not implemented for type {type_}")
    
    def value(self):
        if self.type_ is bool:
            return self.text() == "True"
        else:
            return self.type_(self.text())



# class CustomQGridLayout(QGridLayout):
#     def __init__(self, max_columns=2):
#         super().__init__()
#         # self.addWidget(QLabel(name), 0,0,1,2)
#         self.current_row = 0
#         self.current_column = 0
#         self.max_columns = max_columns
    
#     def next(self):
#         to_return = (self.current_row, self.current_column)
#         self.current_column += 1
#         if self.current_column >= self.max_columns:
#             self.current_column = 0
#             self.current_row += 1
#         return to_return




class SectionExpandButton(QWidget):
    """a QPushbutton that can expand or collapse its section
    """
    def __init__(self, item, text = "", parent = None, autoCheck=False,layoutWidget=None,check=True):
        
        super().__init__(parent)
        self.section = item
        self.setLayout(QHBoxLayout())
        self.checkbox = QCheckBox()
        self.checkbox.setChecked(check)
        if autoCheck:
            self.checkbox.setEnabled(False)
        self.layoutWidget = layoutWidget
        self.checkbox.clicked.connect(self.checkboxClicked)
        self.button = QPushButton(text)
        self.button.clicked.connect(self.on_clicked)
        self.layout().addWidget(self.checkbox,0)
        self.layout().addWidget(self.button,1)
        
    def checkboxClicked(self):
        
        self.layoutWidget.setEnabled(self.checkbox.isChecked())
        

    def on_clicked(self):
        """toggle expand/collapse of section by clicking
        """
        if self.section.isExpanded():
            self.section.setExpanded(False)
        else:
            self.section.setExpanded(True)




class NonMrcFilesPixelSizeWidget(QDialog):
    def __init__(self, parent=None, count=None) -> None:
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        if count is None:
            text = "There are files without given pixel size.\nPlease confirm the pixel size for these files."
        else:
            text = f"There are {count} files without given pixel size.\nPlease confirm the pixel size for these files."
        label = QLabel(text)
        pixelsizelabel = QLabel("Pixelsize [Å]")
        self.pixelsizelineedit = QLineEdit("1.0")
        self.pixelsizelineedit.setValidator(CorrectDoubleValidator(0.01, None, 1.0))
        lowerLayout = QHBoxLayout()
        lowerLayout.addWidget(pixelsizelabel)
        lowerLayout.addWidget(self.pixelsizelineedit)
        self.layout().addWidget(label)
        self.layout().addLayout(lowerLayout)

        self.buttonLayout = QHBoxLayout()
        self.confirmButton = QPushButton("Confirm")
        self.confirmButton.clicked.connect(self.confirm)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.cancel)
        self.buttonLayout.addWidget(self.confirmButton)
        self.buttonLayout.addWidget(self.cancelButton)
        
        self.layout().addLayout(self.buttonLayout)
        
    
    def cancel(self):
        self.reject()

    def confirm(self):
        self.accept()

    def getPixelSize(self):
        return float(self.pixelsizelineedit.text())


class ConfigWidget(QDialog):
    """a dialog to which collapsible sections can be added;
    subclass and reimplement define_sections() to define sections and
        add them as (title, widget) tuples to self.sections
    """
    def __init__(self, parent:QWidget=None, multiple=False, name="", configs={} ):
        global DEFAULT_CONFIGS
        super().__init__(parent)

        x,y,x_end, y_end = parent.parent().geometry().getCoords()
        new_x = ((x_end + x) // 2) - 150
        new_y = ((y_end + y) // 2) - 200
        self.setWindowTitle(name)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        layout = QVBoxLayout()
        layout.addWidget(self.tree)
        self.setLayout(layout)
        self.tree.setIndentation(0)
        self.buttons = {}
        self.gridlayouts = {}

        self.previous_configs = {"run":{}}
        for func, params in DEFAULT_CONFIGS.items():
            if func not in self.previous_configs:
                self.previous_configs[func] = {}
            for key, value in params.items():
                if func not in configs or key not in configs[func]:
                    self.previous_configs[func][key] = value
                else:
                    self.previous_configs[func][key] = configs[func][key]

        for key in DEFAULT_CONFIGS.keys():
            if "run" in configs:
                if key in configs["run"]:
                    self.previous_configs["run"][key] = configs["run"][key]

        self.configs = {key:{} for key in DEFAULT_CONFIGS.keys()}
        # self.sections = []
        self.createAllButtons()
        self.createAllParameters()
        self.setGeometry(new_x,new_y,500,500)


        self.checkBox = QCheckBox("Apply configs to all")
        if multiple:
            self.layout().addWidget(self.checkBox)
        
        self.buttonLayout = QHBoxLayout()
        self.confirmButton = QPushButton("Confirm")
        self.confirmButton.clicked.connect(self.confirm)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.cancel)
        self.buttonLayout.addWidget(self.confirmButton)
        self.buttonLayout.addWidget(self.cancelButton)
        
        self.layout().addLayout(self.buttonLayout)

    def cancel(self):
        self.reject()

    def confirm(self):
        self.accept()
    
    def getConfigs(self):
        config = {"run":{}}
        for func, value_dict in self.configs.items():
            config[func] = {}
            config["run"][func] = self.buttons[func].checkbox.isChecked()
            for key, lineedit in value_dict.items():
                if isinstance(lineedit, QComboBox):
                    config[func][key] = lineedit.currentText()
                elif isinstance(lineedit, QCheckBox):
                    config[func][key] = lineedit.isChecked()
                else:
                    config[func][key] = lineedit.value()
        return config


    def createAllButtons(self):
        global DEFAULT_CONFIGS
        for key in DEFAULT_CONFIGS.keys():
            button = self.add_button(key)
            self.buttons[key] = button

    def createAllParameters(self):
        global DEFAULT_CONFIGS
        def addParameter(shown_name, name, func):
            if hasattr(DEFAULT_CONFIGS[func][name], "__call__"):
                lineedit = QComboBox()
                items = DEFAULT_CONFIGS[func][name]()
                lineedit.addItems(items)
                if not hasattr(self.previous_configs[func][name], "__call__") and self.previous_configs[func][name] in items:
                    lineedit.setCurrentText(str(self.previous_configs[func][name]))
                elif "Default" in items:
                    lineedit.setCurrentText("Default")
                elif "Default_NN" in items:
                    lineedit.setCurrentText("Default_NN")
            elif isinstance(DEFAULT_CONFIGS[func][name], bool):
                lineedit = QCheckBox()
                # b = DEFAULT_CONFIGS[func][name]
                lineedit.setChecked(self.previous_configs[func][name])
            else:
                lineedit = QLabelWithValidator(str(self.previous_configs[func][name]), type(DEFAULT_CONFIGS[func][name]), self.previous_configs[func][name])
            label = QLabel(shown_name)

            self.add_label_lineedit(label, lineedit, func)

            self.configs[func][name] = lineedit

        addParameter("Max neighbour distance [Å]", "max_neighbour_dist", "estimateThickness")
        addParameter("Min Thickness [Å]", "min_thickness", "estimateThickness")
        addParameter("Max Thickness [Å]", "max_thickness", "estimateThickness")
        addParameter("Smoothing sigma", "sigma", "estimateThickness")
        addParameter("Max neighbour distance [Å]", "max_neighbour_dist", "estimateCurvature")
        # addParameter("STD Threshold","std_threshold", "identifyIceContamination")
        addParameter("Only used closed", "use_only_closed", "general")
        addParameter("Rerun", "rerun", "general")
        addParameter("Step size [Å]", "step_size", "general")
        addParameter("Min size", "min_size", "general")
        addParameter("Rerun segmentation", "rerun_segmentation", "segmentation")
        addParameter("Run only segmentation","only_segmentation", "segmentation")
        addParameter("Combine snippets (experimental)", "combine_snippets", "segmentation")
        addParameter("Identify instances", "identify_instances", "segmentation")
        addParameter("Max nodes", "max_nodes", "segmentation")
        # addParameter("Remove metal grid", "remove_metal_grid", "segmentation")
        addParameter("Segmentation model", "segmentation_model", "segmentation")
        addParameter("Shape classifier", "shape_classifier", "shapePrediction")
        addParameter("Only run for new data", "only_run_for_new_data", "general")
        addParameter("Membranes are dark", "dark_mode", "general")
        addParameter("Estimate middle plane", "estimate_middle_plane", "general")
        addParameter("Resize to [Å]","to_size","maskGrid")
        addParameter("Grid diameter [Å]","diameter","maskGrid")
        addParameter("Threshold","threshold","maskGrid")
        addParameter("Hole coverage","coverage_percentage","maskGrid")
        addParameter("Grid coverage","outside_coverage_percentage","maskGrid")
        addParameter("Detect ring","detect_ring","maskGrid")
        addParameter("Ring width [Å]","ring_width","maskGrid")
        addParameter("Wobble","wobble","maskGrid")
        addParameter("High pass sigma [Å]","high_pass","maskGrid")
        addParameter("Remove only edge","return_ring_width", "maskGrid")
        addParameter("Use manual mask","use_existing_mask", "maskGrid")
        addParameter("Distance","distance", "maskGrid")
        addParameter("Cropping [Å]", "crop", "maskGrid")



    def add_button(self, title):
        """creates a QTreeWidgetItem containing a button 
        to expand or collapse its section
        """
        item = QTreeWidgetItem()
        self.tree.addTopLevelItem(item)
        layoutWidget = QWidget()
        layoutWidget.setLayout(QGridLayout())
        to_check = True
        if title in self.previous_configs["run"]:
            to_check = self.previous_configs["run"][title]
        elif title == "maskGrid":
            to_check = False
        button = SectionExpandButton(item, text = title,autoCheck=title=="general" or title=="segmentation", layoutWidget=layoutWidget, check=to_check)
        self.tree.setItemWidget(item, 0,button)
        
        self.add_widget(item, layoutWidget)
        self.gridlayouts[title] = {"layout":layoutWidget, "counter":0}
        return button

    def add_widget(self, button, widget):
        """creates a QWidgetItem containing the widget,
        as child of the button-QWidgetItem
        """
        section = QTreeWidgetItem(button)
        section.setDisabled(True)
        self.tree.setItemWidget(section, 0, widget)
        return section

    def add_label_lineedit(self, label, lineedit, func):
        layout:QGridLayout = self.gridlayouts[func]["layout"].layout()
        layout.addWidget(label, self.gridlayouts[func]["counter"], 0)
        layout.addWidget(lineedit, self.gridlayouts[func]["counter"], 1)

        self.gridlayouts[func]["counter"] += 1      


class nJobsDialog(QDialog):
    def __init__(self, parent):
        super().__init__()
        self.setLayout(QGridLayout())
        self.setWindowTitle("Parallelization parameters")
        self.njobsLabel = QLabel("# parallel jobs")
        self.threadsLabel = QLabel("# Threads")
        self.njobsLineedit = QLabelWithValidator("5", int, 5)

        self.threadsLineedit = QLabelWithValidator("5", int, 5)
        x,y,x_end, y_end = parent.parent().geometry().getCoords()
        new_x = ((x_end + x) // 2) - 150
        new_y = ((y_end + y) // 2) - 50
        self.setGeometry(new_x, new_y, 300,100)
        self.layout().addWidget(self.njobsLabel, 0, 0)
        self.layout().addWidget(self.njobsLineedit, 0, 1)
        self.layout().addWidget(self.threadsLabel, 1, 0)
        self.layout().addWidget(self.threadsLineedit, 1, 1)

        self.buttonLayout = QHBoxLayout()
        self.confirmButton = QPushButton("Confirm")
        self.confirmButton.clicked.connect(self.confirm)
        
        self.cancelButton = QPushButton("Cancel")
        self.cancelButton.clicked.connect(self.cancel)
        self.buttonLayout.addWidget(self.confirmButton)
        self.buttonLayout.addWidget(self.cancelButton)
    
        self.layout().addLayout(self.buttonLayout, 2,0,1,2)

    def getConfigs(self):
        return {"njobs":self.njobsLineedit.value(), "threads":self.threadsLineedit.value()}
    
    def cancel(self):
        self.reject()

    def confirm(self):
        self.accept()

class runButtonWidget(QPushButton):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Run analysis")
        self.setToolTip("Run analysis for this dataset. Opens up a window in which you can set the run parameters for each dataset.")
        self.clicked.connect(self.runAnalysis)
        self.worker = None
        self.currentThread = None
        self.queue = None
        self.counter_ = 0
        self.stopEvent = None

    @property
    def counter(self):
        self.counter_ += 1
        return self.counter_

    def mainWindow(self):
        return self.parent().parent().parent().parent()
    
    def runAnalysis(self):
        global MESSAGE
        selection = self.mainWindow().datasetListWidget.listWidget.selectedItems()
        selection = [self.mainWindow().datasetListWidget.listWidget.itemWidget(item) for item in selection if not running(self.mainWindow().datasetListWidget.listWidget.itemWidget(item).dataset.name)]

        if any([item.dataset.isZipped for item in selection]):
            MESSAGE(f"Cannot run analysis because some datasets are still zipped.")
            return
        if len(selection) > 0:
            configs = []
            apply_to_all = False
            
            for item in selection:
                if not apply_to_all:
                    parameter_widget = ConfigWidget(self, len(selection) > 1, item.dataset.name, item.dataset.last_run_kwargs)
                    result = parameter_widget.exec()
                    if result == 0:
                        return
                    config = parameter_widget.getConfigs()
                    apply_to_all = parameter_widget.checkBox.isChecked()

                    # config["General"] = {
                    #     "pixel_size":7,
                    #     "only_closed": True,
                    #     "min_size": 200,
                    #     "max_hole_size": 10,
                    #     "micrograph_pixel_size":None,
                    #     "step_size":13
                    #     }


                configs.append(config)
                nonMrcFiles = [path for path in item.dataset.micrograph_paths if Path(path).suffix != ".mrc" and not path in item.dataset.pixelSizes]
                if len(nonMrcFiles) > 0:
                    pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
                    result = pixelSizeWidget.exec()
                    if result == 0:
                      return
                    ps = pixelSizeWidget.getPixelSize()
                    for file in nonMrcFiles:
                        item.dataset.pixelSizes[file] = ps 
                

            parallelWidget = nJobsDialog(self)
            result = parallelWidget.exec()
            if result == 0:
                return
            njobs_threads_dict = parallelWidget.getConfigs()

            if self.currentThread is None:
                thread = QThread()
                queue = Queue()
                stopEvent = mp.get_context("spawn").Event()
                worker = AnalyserWorker(queue, stopEvent)
                worker.moveToThread(thread)
                thread.started.connect(worker.run)
                worker.finished.connect(thread.quit)
                worker.finished.connect(worker.deleteLater)
                thread.finished.connect(thread.deleteLater)
                thread.finished.connect(self.finishedRunning)
                worker.progress.connect(self.progressEmited)
                
                thread.start()   
                self.currentThread = thread
                self.worker = worker
                self.queue = queue
                self.stopEvent = stopEvent
            for item, config in zip(selection, configs):
                self.queue.put((item.dataset, config, njobs_threads_dict))
    
    def stopThread(self, keep_running=False):
        global MESSAGE
        if self.currentThread is not None:
            if keep_running:
                self.queue.put("STOP")
            else:

                self.stopEvent.set()
                MESSAGE("Waiting until analysis is done or thread is closed.")
            
            # while self.currentThread.isRunning():
            #     time.sleep(0.5)


    def finishedRunning(self):
        pass

    def progressEmited(self, emit):
        global CURRENTLY_RUNNING, MESSAGE
        dataset, state = emit
        
        if state == 0:
            CURRENTLY_RUNNING.add(dataset)
        elif state == 1:
            if running(dataset):
                CURRENTLY_RUNNING.remove(dataset)
            else:
                MESSAGE(f"{dataset} missing in Currently running, {CURRENTLY_RUNNING}\n")
        elif state == 2:
            
            MESSAGE(dataset)
        elif state == 3:
            MESSAGE(dataset)

        self.mainWindow().datasetListWidget.listWidget.showInfo()



class MplCanvas(FigureCanvasQTAgg):

    def __init__(self, parent=None, width=5, height=4, dpi=100, rows=1, columns=1):
        fig = Figure(figsize=(width, height), dpi=dpi,constrained_layout=True)
        self.axes = fig.subplots(rows, columns)
        self.fig = fig
        super(MplCanvas, self).__init__(fig)
        self.resizeFonts()
        self.mpl_connect("button_press_event", self.onClick)
        
        
    def resizeFonts(self, fontsize=6):
        for item in ([self.axes.title, self.axes.xaxis.label, self.axes.yaxis.label] +
             self.axes.get_xticklabels() + self.axes.get_yticklabels()):
            item.set_fontsize(fontsize)

    def onClick(self, button):
        
        self.menu = QMenu(self)
        export = QAction('Save as', self)
        export.triggered.connect(self.exportCanvas)
        self.menu.addAction(export)
        # add other required actions
        self.menu.popup(QtGui.QCursor.pos())

    def exportCanvas(self):
        filename,filt = QFileDialog.getSaveFileName(self, "Save file", ".", "Images (*.png, *.jpg)" )
        if filename is None or len(Path(filename).stem) == 0:
            return
        self.fig.savefig(filename)

class graphWidget(QTabWidget):
    def __init__(self, parent=None):
        
        super().__init__(parent)
        self.membraneGraphs = membraneGraphWidget(self)
        self.pointThicknesses = pointsGraphWidget(self, )
        self.pointCurvatures = pointsGraphWidget(self, "Curvature")

        self.addTab(self.membraneGraphs, "Membranes")
        self.addTab(self.pointThicknesses, "Thickness values")
        self.addTab(self.pointCurvatures, "Curvature values")
        self.setMinimumSize(150,150)
        self.setBaseSize(200,200)

class pointsGraphWidget(QWidget):
    def __init__(self, parent, attr="Thickness"):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(width=3.5, height=3.5, dpi=100)
        self.layout().addWidget(self.canvas)
        self.attr = attr

        self.extraLayout = QHBoxLayout()

        self.logYScaleCheckbox = QCheckBox("Log scale Y")
        # self.logYScaleCheckbox.clicked.connect(self.setLogScale)
    
        self.showAllCheckbox = QCheckBox("Show all")
        self.showAllCheckbox.setToolTip("Show data from all tabs")
        # self.showAllCheckbox.clicked.connect(self.showData)

        self.refreshButton = QPushButton("Refresh")
        self.refreshButton.setToolTip("Load the points again. This can take some time, which is why this is not done dynamically.")
        self.refreshButton.clicked.connect(self.showData)

        self.exportButton = QPushButton("Export")
        self.exportButton.setToolTip("Save the current graph as an image file.")
        self.exportButton.clicked.connect(self.exportCanvas)
        self.extraLayout.addWidget(self.showAllCheckbox)
        self.extraLayout.addWidget(self.logYScaleCheckbox)
        self.extraLayout.addWidget(self.refreshButton)
        self.extraLayout.addWidget(self.exportButton)
        

        self.layout().addLayout(self.extraLayout)
        

    def setLogScale(self):
        pass


    def clearData(self):
        self.canvas.axes.cla()
    
    def mainWindow(self):
        return self.parent().parent().parent().parent().parent().parent()

    def showData(self):

        if self.showAllCheckbox.isChecked():
            tabs = self.mainWindow().datasetTabWidget.getAllTabs()
        else:
            if self.mainWindow().datasetTabWidget.currentWidget() is None:
                return
            tabs = [self.mainWindow().datasetTabWidget.currentWidget()]
        if len(tabs) == 0:
            return
        dataset_values = {}
        for tab in tabs:
            dataset = tab.model().dataset
            dataset_values[dataset.name] = []
            data:pd.DataFrame = tab.model().shown_data
            data.value_counts()
            micrographs = data["Micrograph"].unique().tolist()
            for micrograph in micrographs:
                indexes = data[data["Micrograph"] == micrograph]["Index"].tolist()
                wrapper:AnalyserWrapper = get_analyser(micrograph, dataset)
                points = wrapper.json["Points"]
                for index in indexes:
                    dataset_values[dataset.name].extend(points[str(index)][self.attr.capitalize()])
        df = pd.DataFrame(columns = ['Dataset',self.attr])

        for key, value in dataset_values.items():
            patient_df = pd.DataFrame({'Dataset': [key]*len(value),
                                    
                                    self.attr: value,
                                })
            df = pd.concat((df, patient_df), ignore_index = True)

        self.canvas.axes.cla()
        if len(data) == 0:
            return
        self.canvas.axes.set_title(self.attr)
       
        sns.violinplot(df, x="Dataset", y=self.attr, ax=self.canvas.axes, cut=0.1)
        xticks = self.canvas.axes.get_xticklabels()
        for tick in xticks:
            tick.set_rotation(5)
        

        if self.logYScaleCheckbox.isChecked():
            self.canvas.axes.set_yscale("log")
        self.canvas.resizeFonts()
        self.canvas.draw()



        if data is None:
            return
        # self.showData(data, self.currentColumn, self.currentType)


    def exportCanvas(self):
        filename,filt = QFileDialog.getSaveFileName(self, "Save file", ".", "Images (*.png, *.jpg)" )
        if filename is None or len(Path(filename).stem) == 0:
            return
        self.canvas.fig.savefig(filename)


class membraneGraphWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        self.canvas = MplCanvas(width=3.5, height=3.5, dpi=100)
        self.layout().addWidget(self.canvas)
        self.currentColumn = ""
        self.currentType = ""

        self.extraLayout = QHBoxLayout()

        self.logYScaleCheckbox = QCheckBox("Log scale Y")
        self.logYScaleCheckbox.clicked.connect(self.redrawData)
    
        self.showAllCheckbox = QCheckBox("Show all")
        self.showAllCheckbox.setToolTip("Show data from all tabs")
    
        self.showAllCheckbox.clicked.connect(self.redrawData)

        self.percentageBarplotCheckbox = QCheckBox("% for bar plots")
        self.percentageBarplotCheckbox.setToolTip("Show the percentage instead of the count for bar plots.")
        self.percentageBarplotCheckbox.clicked.connect(self.redrawData)

        self.exportButton = QPushButton("Export")
        self.exportButton.setToolTip("Save the current graph as an image file.")
        self.exportButton.clicked.connect(self.exportCanvas)
        self.extraLayout.addWidget(self.showAllCheckbox)
        self.extraLayout.addWidget(self.logYScaleCheckbox)
        self.extraLayout.addWidget(self.percentageBarplotCheckbox)
        self.extraLayout.addWidget(self.exportButton)
        

        self.layout().addLayout(self.extraLayout)
        

    
    def showData(self, data:pd.DataFrame, column:str, t:str):
        self.canvas.axes.cla()
        if len(data) == 0:
            return
        if self.parent().currentIndex() != 0:
            return
        self.canvas.axes.set_title(column)

        

        if t == "violin":
            sns.violinplot(data, x="dataset", y=column, ax=self.canvas.axes, cut=0.1)
            xticks = self.canvas.axes.get_xticklabels()
            for tick in xticks:
                tick.set_rotation(5)
            
                
            # self.canvas.axes.violinplot(data=data.values(), dataset=data.keys())
        elif t == "bar":
            
            values = data[column].unique().tolist()
            datasets = data["dataset"].unique().tolist()
            width = 0.8 / len(datasets)
            value_per_dataset = []
            for dataset in datasets:
                v = data[data["dataset"] == dataset][column].value_counts()
                current_values = [v[key] if key in v else 0 for key in values ]
                value_per_dataset.append(current_values)
            
            ind = np.arange(len(values))
            
            summed_values = [np.sum([current_values[key] for current_values in value_per_dataset]) for key in range(len(values))]

            sorted_idxs = np.argsort(summed_values)[::-1]
            for counter, i in enumerate(sorted_idxs):
                ind[i] = counter

            for counter, (dataset, dataset_value) in enumerate(zip(datasets, value_per_dataset)):
                if self.percentageBarplotCheckbox.isChecked():
                    self.canvas.axes.bar(ind + counter * width, [i/np.sum(dataset_value) for i in dataset_value], width=width, label=dataset,)
                    self.canvas.axes.set_title(f"{column} %")

                else:
                    self.canvas.axes.bar(ind + counter * width, dataset_value, width=width, label=dataset,)
                    self.canvas.axes.set_title(column)
                
                self.canvas.axes.legend()
            self.canvas.axes.set_xticks(ind + width / 2, values)
            xticks = self.canvas.axes.get_xticklabels()
            for tick in xticks:
                tick.set_rotation(10)
        if self.logYScaleCheckbox.isChecked():
            self.canvas.axes.set_yscale("log")

        self.canvas.resizeFonts()
        self.canvas.draw()
        self.currentColumn = column
        self.currentType = t


    def mainWindow(self):
        return self.parent().parent().parent().parent().parent().parent()

    def clearData(self):
        self.canvas.axes.cla()
    
    def redrawData(self):
        if self.currentColumn == "" or self.currentType == "":
            self.clearData()
            return
        data = self.mainWindow().datasetTabWidget.getCurrentData(all=self.showAllCheckbox.isChecked(), column=self.currentColumn)
        if data is None:
            return
        self.showData(data, self.currentColumn, self.currentType)


    def exportCanvas(self):
        filename,filt = QFileDialog.getSaveFileName(self, "Save file", ".", "Images (*.png, *.jpg)" )
        if filename is None or len(Path(filename).stem) == 0:
            return
        self.canvas.fig.savefig(filename)


class ImageViewerWindow(QWidget):
    def __init__(self, wrapper, index):
        super().__init__()

        # self.setWindowTitle("Image Viewer")
# 
        # Create a QLabel widget to display the image
        self.image_label = QLabel()
        data = wrapper.analyser.getResizedMicrograph()
        # Load the image and display it in the label

        # data,_ = load_file(image_path)
        mean, std = np.mean(data), np.std(data)
        data = np.clip(data, mean - 4* std, mean+4*std)
        data -= np.min(data)
        data /= np.max(data)
        data *= 255

        seg = wrapper.analyser.segmentation_stack[index].todense()
        seg_color = np.zeros((*seg.shape, 4))
        seg_color[seg > 0] = (255,0,0,255)

        img = Image.fromarray(data).convert("RGBA")
        seg_img = Image.fromarray(seg_color.astype(np.uint8))
        overlay = Image.blend(img, seg_img, 0.1)
        data = np.array(overlay)
            
        self.setWindowTitle(Path(wrapper.analyser.micrograph_path).name)
        image = q2n.array2qimage(data)

        pixmap = QPixmap.fromImage(image)
        screen_size = QDesktopWidget().availableGeometry(0).size()
        newSize = QSize(int(screen_size.width() * 0.5),int(screen_size.height() * 0.5))
        scaled_pixmap = pixmap.scaled(newSize, aspectRatioMode=Qt.KeepAspectRatio, transformMode=Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

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
        self.thicknessContourGraph = MplCanvas(self, width=2, height=3)
        self.thicknessProfileGraph = MplCanvas(self, width=2, height=3)
        self.imagePixmap = QPixmap(self.shape,self.shape)
        self.maskPixmap = QPixmap(self.shape,self.shape)
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)
        self.imageLabel.adjustSize()
        self.maskLabel.adjustSize()
        self.setLayout(QGridLayout())
        # self.imageLabel.setStyleSheet("border: 1px solid black")
        # self.maskLabel.setStyleSheet("border: 1px solid black")

        # self.placeholder = QWidget()
        # self.placeholder_2 = QWidget()
        # self.placeholder_3 = QWidget()
        # self.placeholder_4 = QWidget()
        
        
        self.layout().addWidget(self.imageLabel,0,0,alignment=Qt.AlignCenter)
        
        
        self.layout().addWidget(self.maskLabel,0,2,alignment=Qt.AlignCenter)
        # self.layout().addWidget(self.placeholder,0,1)
        # self.layout().addWidget(self.placeholder_2, 1,0)
        # self.layout().addWidget(self.placeholder_3, 1,2)
        # self.layout().addWidget(self.placeholder_4, 2,1)

        self.layout().addWidget(self.thicknessContourGraph, 2,0)
        self.layout().addWidget(self.thicknessProfileGraph, 2,2)
        
        # self.layout().setColumnStretch(1,1)
        # self.layout().setRowStretch(1,1)

        self.imageLabel.setToolTip("Original image")
        self.maskLabel.setToolTip("Segmentation")
        self.setMinimumSize(200, 230)


    def getShowFullImage(self, path, index):
        def showFullImage(event):
            
            self.image_viewer = ImageViewerWindow(path, index)
            self.image_viewer.show()


        
        return showFullImage

    def loadImages(self, wrapper:AnalyserWrapper, idx:int, min_thickness, max_thickness, min_curvature, max_curvature):
        self.clearAll()
        self.setToolTip(str(wrapper.directory) + f"\nIndex: {str(idx)}")
        tn, seg = wrapper.get_thumbnails([idx])
        tn = tn[idx]
        seg = seg[idx]

        tn = np.array(tn)
        seg = np.array(seg)
        image = q2n.gray2qimage(tn)
       
        scaled = image.scaled(
            self.shape,
            self.shape,
            aspectRatioMode=Qt.KeepAspectRatio, 
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        self.imagePixmap = QPixmap.fromImage(scaled)
        self.imageLabel.setPixmap(self.imagePixmap)
        self.imageLabel.mousePressEvent = self.getShowFullImage(wrapper, idx)

        
        seg = q2n.gray2qimage(seg)

        scaled = seg.scaled(
            self.shape,
            self.shape,
            aspectRatioMode=Qt.KeepAspectRatio,
        )
        self.maskPixmap = QPixmap.fromImage(scaled)
        self.maskLabel.setPixmap(self.maskPixmap)



        points = wrapper.json["Points"]
        membrane = points[str(idx)]
        thickness = membrane["Thickness"]
        if any([i is not None for i in thickness]):
            thickness = np.array(thickness, dtype=np.float32)
            self.thicknessContourGraph.axes.set_title("Thickness contour")
            self.thicknessContourGraph.axes.plot(thickness)
            self.thicknessContourGraph.axes.set_ylabel("Thickness [Å]")
            self.thicknessContourGraph.axes.set_xlabel("Contour pixel")
            self.thicknessContourGraph.axes.set_ylim(min_thickness, max_thickness)
            self.thicknessContourGraph.resizeFonts()
            self.thicknessContourGraph.draw()



        curvature = membrane["Curvature"]
        if any([i is not None for i in curvature]):
            curvature = np.array(curvature, dtype=np.float32)
            self.thicknessProfileGraph.axes.set_title("Curvature contour")
            self.thicknessProfileGraph.axes.plot(curvature)
            self.thicknessProfileGraph.axes.set_ylabel("Curvature [1/Å]")
            self.thicknessProfileGraph.axes.set_xlabel("Contour pixel")
            self.thicknessProfileGraph.axes.set_ylim(min_curvature, max_curvature)

            self.thicknessProfileGraph.resizeFonts()
            self.thicknessProfileGraph.draw()
    
    def clearAll(self):
        self.imagePixmap.fill(QColor("white"))
        self.maskPixmap.fill(QColor("white"))
        self.imageLabel.setPixmap(self.imagePixmap)
        self.maskLabel.setPixmap(self.maskPixmap)

        self.thicknessContourGraph.axes.cla()
        self.thicknessProfileGraph.axes.cla()



class ImageData:
    def __init__(self, row, dataset, index=None):
        wrapper = get_analyser(row, dataset)
        idx = row["Index"]
        tn, seg = wrapper.get_thumbnails([idx])
        self.index = index
        
        try:
            self.image = q2n.gray2qimage(np.array(tn[idx]))
            self.mask = q2n.gray2qimage(np.array(seg[idx]))
        except KeyError as e:
            raise e


def getMicrographDataMP(micrograph, segmentation, indeces=None):
    return MicrographData(micrograph, segmentation, forMP=True, indeces=indeces)

def fillMicrographDataMp(md, fill):
    md.loadData(fill, True)
    return md

class MicrographData:
    def __init__(self, micrograph, segmentation,forMP=False, fill=False,indeces=None):
        self.micrograph = micrograph
        self.segmentation = segmentation
        self.indeces = indeces
        self.max_index = 0
        self.loadData(fill, forMP)
        

    def applyAfterMP(self):
        
        self.image = q2n.gray2qimage(self.image)
        if self.seg is not None:
            try:
                self.seg = q2n.gray2qimage(self.seg)
            except Exception as e:
                self.seg = None
        
    def loadData(self, fill=False, forMP=False):
        timer = {}
        now = datetime.datetime.now()
        image,_ = load_file(self.micrograph)
        timer["load_micro"] = (datetime.datetime.now() - now).total_seconds()
        now = datetime.datetime.now()
        self.image_shape = image.shape
        median = np.median(image)
        std = np.std(image)
        image = np.clip(image, median - 3*std, median + 3*std)
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        timer["preprocess"] = (datetime.datetime.now() - now).total_seconds()
        now = datetime.datetime.now()
        if self.segmentation is not None:

            seg,_ = load_file(self.segmentation)

            seg = np.array(seg)
            
            if seg.size == 0:
                seg = np.zeros_like(image, dtype=np.uint8)
            else:
                image = resizeMicrograph(image, (seg.shape[-2], seg.shape[-1]))
            timer["load_seg"] = (datetime.datetime.now() - now).total_seconds()
            now = datetime.datetime.now()
            if seg.ndim == 3:
                self.seg_shape = seg.shape[1:]
                click_seg = np.zeros((seg.shape[1], seg.shape[2]), dtype=np.uint8)
                if fill:
                    for counter, s in enumerate(seg):
                        if self.indeces is not None:
                            if counter not in self.indeces:
                                continue
                        seg[counter] = binary_fill_holes(s)
                        click_seg[seg[counter] > 0] = counter + 1
                else:
                    for counter, s in enumerate(seg):
                        if self.indeces is not None:
                            if counter not in self.indeces:
                                seg[counter] = 0
                                continue
                        click_seg[s > 0] = counter + 1 
                seg = (np.sum(seg, 0) > 0) * 255
                self.click_seg = click_seg
                timer["click_seg"] = (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
                
            else:
                timer = {}
                now = datetime.datetime.now()
                self.seg_shape = seg.shape
                try:
                    l, nr = label(seg, np.ones((3,3)))
                except Exception as e:

                    raise e
                timer["label"] = (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
                
                timer["unique"] = (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
                if nr == 0:
                    self.click_seg = None
                else:
                    
                    uniques = np.unique(l)
                    self.click_seg = l
                    if fill:
                        
                        click_seg = np.zeros_like(seg, dtype=np.int32)
                        for u in uniques:
                            if u == 0:
                                continue
                            
                            current_seg = binary_fill_holes((l==u)*1)
                            
                            click_seg[current_seg > 0] = u
                        seg = (click_seg > 0)

                seg = (seg> 0) * 255
            
            if forMP:
                self.seg = seg
            else:
                self.seg = q2n.gray2qimage(seg)
        else:
            self.seg = None
            self.seg_shape = None
            self.click_seg = None
        if self.click_seg is not None:
            self.max_index = np.max(self.click_seg)
            self.click_seg = sparse.as_coo(self.click_seg)
        if forMP:
            self.image = image
        else:
            self.image = q2n.gray2qimage(image)
        self.timer = timer



    def remove_index(self, index):
        if self.click_seg is None:
            return
        ys,xs = np.nonzero(self.click_seg == index,)
        self.click_seg = self.click_seg.todense()
        self.click_seg[self.click_seg == index] = 0
        self.click_seg[self.click_seg > index] -= 1
        self.max_index = np.max(self.click_seg)
        click_color = QColor(0,0,0,255)
        for y,x in zip(ys, xs):
            
            self.seg.setPixelColor(x,y,click_color)
        self.seg = self.seg.convertToFormat(QImage.Format_Grayscale8)
        self.click_seg = sparse.as_coo(self.click_seg)

    def reset_index(self, index):
        if self.click_seg is None:
            return
        ys,xs = np.nonzero(self.click_seg == index,)
        click_color = QColor(255,255,255,255)
        for y,x in zip(ys, xs):
            
            self.seg.setPixelColor(x,y,click_color)
        self.seg:QImage
        self.seg = self.seg.convertToFormat(QImage.Format_Grayscale8)


    def focusIndex(self, index):
        if index != 0:
            if self.seg.format() != QImage.Format_RGBA8888:
                self.seg = self.seg.convertToFormat(QImage.Format_RGBA8888)
            ys,xs = np.nonzero(self.click_seg == index,)
            click_color = QColor(255,0,0,255)
            for y,x in zip(ys, xs):
                
                self.seg.setPixelColor(x,y,click_color)
            return index, True
        return None
        
    



    def clicked(self, coordinate, shape, pad):
        if self.click_seg is None:
            return 0, False
    
        clicked_pos = np.array([int(coordinate.y()), int(coordinate.x())])
        
        ratio = self.seg_shape[0] / self.seg_shape[1]
        shape = np.array([shape[1]//2, shape[0]])
        if clicked_pos[1] > shape[1]:
            clicked_pos[1] = clicked_pos[1] - shape[1] + pad
        starting_pos = np.array([pad, pad])
        if self.seg_shape[1] > self.seg_shape[0]:
            to_add = int((shape[1]) * (1-ratio) / 2)
            starting_pos[0] += to_add
            conversion_ratio = self.seg_shape[1] / (shape[1] - pad)
        else:
            to_add = int(shape[0] * (1-1/ratio) / 2)
            starting_pos[1] *= to_add
            conversion_ratio = self.seg_shape[0] / (shape[0] - pad*2)
        clicked_pos -= starting_pos
        clicked_pos = clicked_pos.astype(np.float32)
        clicked_pos *= conversion_ratio
        clicked_pos = clicked_pos.astype(np.int32)
        if clicked_pos[0] < 0 or clicked_pos[1] < 0 or clicked_pos[0] >= self.click_seg.shape[0] or clicked_pos[1] >= self.click_seg.shape[1]:
            return 0, False
        index = self.click_seg[clicked_pos[0], clicked_pos[1]]
        if index != 0:
            if self.seg.format() != QImage.Format_RGBA8888:
                self.seg = self.seg.convertToFormat(QImage.Format_RGBA8888)
            ys,xs = np.nonzero(self.click_seg == index,)
            click_color = QColor(255,0,0,255)
            for y,x in zip(ys, xs):
                
                self.seg.setPixelColor(x,y,click_color)
            return index, True
        return 0, False

        


class MicrographPreviewDelegate(QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        self.shape = np.array([320,160])
        self.last_index = None
        self.last_data = None
        super().__init__(parent)

    

    def reset_index(self):
        if self.last_data is None :
            return
        self.parent().model.layoutChanged.emit()
        

    def editorEvent(self, event: QEvent, model: QAbstractItemModel, option: QStyleOptionViewItem, index: QModelIndex) -> bool:
        if event.type() == event.MouseButtonRelease and event.button() == Qt.RightButton:
            item_position = option.rect
            mouse_position = event.pos() - item_position.topLeft()
            shape = item_position.height(), item_position.width()
            data:MicrographData = model.data(index,Qt.DisplayRole)
            if self.last_data is not None:
                self.last_data.reset_index(self.last_index)
                self.reset_index()
            if data is not None:
                found_index, inside = data.clicked(mouse_position, shape, CELL_PADDING)
                if inside:
                    self.last_index = found_index
                    self.last_data = data
                else:
                    
                    self.last_index = None
                    self.last_data = None
            else:
                self.last_index = None
                self.last_data = None
        elif event.type() == event.MouseButtonDblClick and event.button() == Qt.LeftButton:
            data:MicrographData = model.data(index,Qt.DisplayRole)
            QApplication.clipboard().setText(str(data.micrograph))

                    

        return super().editorEvent(event, model, option, index)

    def paint(self, painter, option, index):
        
        # data is our preview object
        data:MicrographData = index.model().data(index, Qt.DisplayRole)
        if data is None:
            
            return
        # painter.save()
        # width = option.rect.width() - CELL_PADDING * 2
        # width = (option.rect.width() - CELL_PADDING * 2) // 2
        # height = option.rect.height() - CELL_PADDING * 2
        width = (option.rect.width() - CELL_PADDING * 2) // 2
        height = option.rect.height() - CELL_PADDING * 2

        # option.rect holds the area we are painting on the widget (our table cell)
        # scale our pixmap to fit
        scaled = data.image.scaled(
            width,
            height,
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        # Position in the middle of the area.
        x = int(CELL_PADDING + (width - scaled.width()) / 2)
        y = int(CELL_PADDING + (height - scaled.height()) / 2)



        painter.drawImage(option.rect.x() + x, option.rect.y() + y, scaled)

        if data.seg is not None:
            scaled_mask = data.seg.scaled(width, height, aspectRatioMode=Qt.KeepAspectRatio)
        else:
            scaled :QImage
            scaled_mask = scaled.copy()
            scaled_mask.fill(QColor("white"))
        
        painter.drawImage(option.rect.x() + x + scaled.width(), option.rect.y() + y, scaled_mask)
        # QToolTip.showText(option.rect.topLeft(), str(data.segmentation), self.parent())

        # painter.restore()
    def sizeHint(self, option, index):
        # All items the same size.
        return QSize(self.shape[0], self.shape[1])


class PreviewDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, shape=None) -> None:
        if shape is None:
            self.shape = np.array([320,160])
        else:
            self.shape = np.array(shape)
        super().__init__(parent)

    def paint(self, painter, option, index):
        
        # data is our preview object
        data:ImageData = index.model().data(index, Qt.DisplayRole)
        
        if data is None:
            
            return
        # painter.save()
        # width = option.rect.width() - CELL_PADDING * 2
        # width = (option.rect.width() - CELL_PADDING * 2) // 2
        # height = option.rect.height() - CELL_PADDING * 2
        width = (option.rect.width() - CELL_PADDING * 2) // 2
        height = option.rect.height() - CELL_PADDING * 2

        # option.rect holds the area we are painting on the widget (our table cell)
        # scale our pixmap to fit
        scaled = data.image.scaled(
            width,
            height,
            aspectRatioMode=Qt.KeepAspectRatio,
            transformMode=Qt.TransformationMode.SmoothTransformation
        )
        # Position in the middle of the area.
        x = int(CELL_PADDING + (width - scaled.width()) / 2)
        y = int(CELL_PADDING + (height - scaled.height()) / 2)



        painter.drawImage(option.rect.x() + x, option.rect.y() + y, scaled)

        
        scaled_mask = data.mask.scaled(width, height, aspectRatioMode=Qt.KeepAspectRatio)
        
        painter.drawImage(option.rect.x() + x + scaled.width(), option.rect.y() + y, scaled_mask)
        if hasattr(data, "segmentation"):
            QToolTip.showText(option.rect.center(), data.segmentation, self.parent())
        # painter.restore()
    def sizeHint(self, option, index):
        # All items the same size.
        return QSize(self.shape[0], self.shape[1])



class customSelectionModel(QItemSelectionModel):
    def __init__(self, model):
        self.lastClickedIndex = None
        self.currentlyClickedIndex = None
        super().__init__()
        self.setModel(model)

    def select(self, selection:QItemSelection, flags:QItemSelectionModel.SelectionFlags):

        # Check if the Shift modifier key is pressed
        if QApplication.keyboardModifiers() == Qt.ShiftModifier:
            # Handle the Shift + left-click selection differently
            # Add your custom behavior here
            if self.lastClickedIndex is None:
                super().select(selection, flags)
            else:
                
                twoIndexes = [idx.row() * self.model().columnCount() + idx.column() for idx in [self.currentlyClickedIndex, self.lastClickedIndex]]
                newIndexes = np.arange(np.min(twoIndexes),np.max(twoIndexes)+1,dtype=np.int32)

                selectionSet = set()
                # for idxs in selection.indexes():
                    
                selectionSet.update([(idx.row(), idx.column()) for idx in self.selectedIndexes()])
                newselection = QItemSelection()
                for i in newIndexes:
                    selectionSet.add((i // self.model().columnCount(), i % self.model().columnCount()))
                for i in selectionSet:
                    # newItemSelection = QItemSelection()
                    index = self.model().index(i[0],i[1])
                    newselection.select(index, index)
                    # newselection.append(newItemSelection)
                    # newItemSelection.select()

                super().select(newselection, flags)


        else:
            # Call the base class method for other selection types
            super().select(selection, flags)

    def selectedIndexes(self):
         
        idxs = super().selectedIndexes()
        new_idxs = [idx.row() * self.model().columnCount() + idx.column() for idx in idxs]
        idxs = [idx for idx, new_idx in zip(idxs, new_idxs) if new_idx < len(self.model().previews)]
        return idxs

class PreviewModel(QAbstractTableModel):
    def __init__(self, parent, micrographData=False):
        super().__init__(parent)
        
        # .data holds our data for display, as a list of Preview objects.
        self.previews = []
        self.micrographData = micrographData

    def data(self, index, role):
        try:
            data = self.previews[index.row() * self.columnCount() + index.column() ]
            
        except IndexError:
            return

        if role == Qt.DisplayRole:
            
            return data   # Pass the data to our delegate to draw.

        # if role == Qt.ToolTipRole:
        #     return data.title

    def deleteIdxs(self, idxs):
        self.layoutAboutToBeChanged.emit()
        idxs = sorted(idxs, reverse=True)
        # print(idxs)
        return_idxs = []
        for idx in idxs:
            if not self.micrographData:
                return_idxs.append(self.previews.pop(idx).index)
            else:
                return_idxs.append(self.previews.pop(idx).micrograph)
        self.layoutChanged.emit()
        return return_idxs


    def columnCount(self, index=None):
        
        return max(1,self.parent().size().width() // self.parent().delegate.shape[0])
        
        # return NUMBER_OF_COLUMNS

    def rowCount(self, index=None):
        n_items = len(self.previews)
        return math.ceil(n_items / self.columnCount())


class InspectionTableView(QTableView):
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Get the index of the clicked item
            index = self.indexAt(event.pos())

            # Check if the index is valid
            if index.isValid():
                # Retrieve the data of the clicked item
                self.selectionModel().currentlyClickedIndex = index


        # Call the base class method to perform the default mouse press event handling
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            # Get the index of the clicked item
            index = self.indexAt(event.pos())

            # Check if the index is valid
            if index.isValid():
                # Retrieve the data of the clicked item
                self.selectionModel().lastClickedIndex = index




class MicrographInspectionWindow(QWidget):
    def __init__(self, dataset, parent, startingShape=(320,160), idx=0, number_of_files_to_show=100):
        super().__init__()
        self.customParent = parent
        self.to_remove = None
        self.membranesToRemove = {}
        self.dataset = dataset
        self.startingShape = startingShape
        self.setWindowTitle(dataset.name)
        self.number_of_files_to_show = number_of_files_to_show
        self.idx = idx
        self.view = InspectionTableView(self)
        # self.view.wheelEvent = MicrographInspectionWindow.wheelEvent
        self.view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        
        self.view.horizontalHeader().hide()
        self.view.verticalHeader().hide()
        self.view.setGridStyle(Qt.NoPen)

        self.delegate = MicrographPreviewDelegate(self)
        
        
        self.view.setItemDelegate(self.delegate)
        self.model = PreviewModel(self,True)
        self.view.setModel(self.model)
        
        self.view.setSelectionModel(customSelectionModel(self.model))
        self.view.selectionModel().selectionChanged.connect(self.selectionChanged)

        # palette = QPalette()
        # palette.setColor(QPalette.Highlight, QColor("red"))

        self.scroll_timer = QTimer()
        self.scroll_timer.setInterval(500)  # Adjust the interval as needed
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.on_scroll_stopped)
        self.scrollCounter = 0

        self.buttonLayout = QHBoxLayout()

        self.fillButton = QCheckBox("Fill vesicles")
        self.fillButton.clicked.connect(self.fill)
        self.biggerButton = QPushButton("-")
        self.biggerButton.clicked.connect(lambda: self.changeImageSize(1))
        self.smallerButton = QPushButton("+")
        self.smallerButton.clicked.connect(lambda: self.changeImageSize(-1))

        self.nextMembraneButton = QPushButton(">")
        self.nextMembraneButton.setToolTip("Select next membrane")
        self.previousMembraneButton = QPushButton("<")
        self.previousMembraneButton.setToolTip("Select previous membrane")
        self.previousMembraneButton.clicked.connect(lambda:self.focusNextMembrane(-1))
        self.nextMembraneButton.clicked.connect(lambda:self.focusNextMembrane(1))


        self.indexLayout = QVBoxLayout()
        self.moveIndexLayout = QHBoxLayout()

        self.showIndexLabel = QLabel()
        
        self.moveIndexDownButton = QPushButton("<<")
        self.moveIndexUpButton = QPushButton(">>")
    
        self.moveIndexLayout.addWidget(self.moveIndexDownButton)
        self.moveIndexLayout.addWidget(self.moveIndexUpButton)
        self.moveIndexDownButton.clicked.connect(lambda:self.changeIndex(-1))
        self.moveIndexUpButton.clicked.connect(lambda:self.changeIndex(1))


        self.indexLayout.addWidget(self.showIndexLabel,alignment=Qt.AlignHCenter)
        self.indexLayout.addLayout(self.moveIndexLayout)


        self.removeMicrographButton = QPushButton("Remove micrograph")
        self.removeMicrographButton.clicked.connect(self.removeIndexes)

        self.removeMembraneButton = QPushButton("Remove selected membrane")
        self.removeMembraneButton.clicked.connect(self.addRemoveMembrane)

        self.shapeLabel = QLabel(f"[{self.startingShape[0]}, {self.startingShape[1]}]")

        self.buttonLayout.addWidget(self.fillButton)
        self.buttonLayout.addWidget(self.smallerButton)
        self.buttonLayout.addWidget(self.biggerButton)
        self.buttonLayout.addWidget(self.shapeLabel)
        self.buttonLayout.addLayout(self.indexLayout)

        self.buttonLayout.addWidget(self.nextMembraneButton)
        self.buttonLayout.addWidget(self.previousMembraneButton)
        self.buttonLayout.addWidget(self.removeMicrographButton)
        self.buttonLayout.addWidget(self.removeMembraneButton)
        
        # self.view.setPalette(palette)
        self.setLayout(QVBoxLayout())
        self.layout().addLayout(self.buttonLayout)
        self.layout().addWidget(self.view)
        # self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))

        self.load_files()
    

    def focusNextMembrane(self, direction=1):
        if self.delegate.last_data is not None:
            index = self.delegate.last_index + direction
            
            data = self.delegate.last_data
        
        else:
            idxs = self.view.selectionModel().selectedIndexes()
            if len(idxs) != 1:
                return
            data = self.model.data(idxs[0],Qt.DisplayRole)
            if direction == 1:
                index = 1
            else:
                index = data.max_index
        max_index = data.max_index
        if index > max_index:
            index = 1
        elif index < 1:
            index = data.max_index
        # self.delegate.reset_index()
        data.reset_index(self.delegate.last_index)

        if max_index == 0:
            self.delegate.last_index = None
            self.delegate.last_data = None
        else:
            
            new_index = data.focusIndex(index)
            if new_index is None:
                self.delegate.last_index = None
                self.delegate.last_data = None
            else:
                self.delegate.last_index = new_index[0]
                self.delegate.last_data = data
        
        
        self.model.layoutChanged.emit()

            

    def addRemoveMembrane(self):
        if self.delegate.last_data is not None:
            path = self.delegate.last_data.micrograph
            index = self.delegate.last_index
            if path not in self.membranesToRemove:
                self.membranesToRemove[path] = []
            self.membranesToRemove[path].append(index-1)

            self.delegate.last_data.remove_index(index)
            self.delegate.reset_index()
            self.delegate.last_data = None
            self.delegate.last_index = None
            self.model.layoutChanged.emit()

    
    def changeImageSize(self, direction):
        self.scrollCounter += direction
        new_shape = self.delegate.shape + np.array([10,5]) * self.scrollCounter
        if new_shape[0] < 40 or new_shape[1] <20:
            new_shape = np.array([40,20])
        elif new_shape[0] > 1280 or new_shape[1] >640:
            new_shape = np.array([1280,640])
        self.shapeLabel.setText(f"[{new_shape[0]}, {new_shape[1]}]")
        self.scroll_timer.start()

    def fill(self):
        self.delegate.reset_index()
        self.delegate.last_data = None
        self.delegate.last_index = None

        with mp.get_context("spawn").Pool(max(int(os.environ["CRYOVIA_NJOBS"]), 1)) as pool:
            for data in self.model.previews:
                data:MicrographData
                data.image = None
                data.seg = None
            to_fill = self.fillButton.isChecked()
            result = [pool.apply_async(fillMicrographDataMp, [md, to_fill]) for md in self.model.previews ]
            timer = {"message":0, "rest":0}
            for counter, res in enumerate(result):
                now = datetime.datetime.now()
                # MESSAGE(f"{self.dataset.name}: Filling vesicles in inspection window: {counter + 1}/{len(result)}\n",True)
                timer["message"] += (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
                res = res.get()
                res:MicrographData
                res.applyAfterMP()
        
                self.model.previews[counter ] = res
                timer["rest"] += (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
            


        self.model.layoutChanged.emit()

        self.view.resizeRowsToContents()
        self.view.resizeColumnsToContents()
        

    def on_scroll_stopped(self):
        if self.scrollCounter != 0:
            self.delegate.shape += np.array([10,5]) * self.scrollCounter
            if self.delegate.shape[0] < 40 or self.delegate.shape[1] <20:
                self.delegate.shape = np.array([40,20])
            elif self.delegate.shape[0] > 1280 or self.delegate.shape[1] >640:
                self.delegate.shape = np.array([1280,640])
            self.model.layoutChanged.emit()
            self.view.resizeColumnsToContents()
            self.view.resizeRowsToContents()
            self.scrollCounter = 0

    
          
    def wheelEvent(self, a0: QWheelEvent) -> None:
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if a0.angleDelta().y() > 0:
                self.scrollCounter -= 1
            elif a0.angleDelta().y() < 0:
                self.scrollCounter += 1
            multiplier = 1
            # if abs(self.scrollCounter) % 10 > 0:
            #     multiplier = abs(self.scrollCounter) % 10 +1
            new_shape = self.delegate.shape + np.array([10,5]) * self.scrollCounter * multiplier
            if new_shape[0] < 40 or new_shape[1] <20:
                new_shape = np.array([40,20])
            elif new_shape[0] > 1280 or new_shape[1] > 640:
                new_shape = np.array([1280,640])
            self.shapeLabel.setText(f"[{new_shape[0]}, {new_shape[1]}]")
            self.scroll_timer.start()
            a0.accept()
            return
        return super().wheelEvent(a0)



    def removeIndexes(self):
        idxs = self.view.selectionModel().selectedIndexes()
        if len(idxs) == 0:
            return
        new_idxs = [idx.row() * self.model.columnCount() + idx.column() for idx in idxs]
        pd_idxs = self.model.deleteIdxs(new_idxs)

        if self.to_remove is None:
            self.to_remove = pd_idxs
        else:
            self.to_remove.extend(pd_idxs)

        self.view.selectionModel().clearSelection()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:

            idxs = self.view.selectionModel().selectedIndexes()
            if len(idxs) == 0:
                return
            new_idxs = [idx.row() * self.model.columnCount() + idx.column() for idx in idxs]
            pd_idxs = self.model.deleteIdxs(new_idxs)

            if self.to_remove is None:
                self.to_remove = pd_idxs
            else:
                self.to_remove.extend(pd_idxs)

            self.view.selectionModel().clearSelection()
            # row = self.currentRow()
            # self.removeRow(row)
        else:
            super().keyPressEvent(event)

    # def sizeHint(self):
    #     # All items the same size.
    #     return QSize(1250, 850)
        
    def selectionChanged(self, event=None):
        idxs = self.view.selectionModel().selectedIndexes()


    def load_files(self):
        global MESSAGE
        args = []
        csv = self.dataset.csv
        self.model.previews = []
       
        for micrograph_counter, micrograph in enumerate(self.dataset.micrograph_paths):
            if micrograph_counter >= self.idx and micrograph_counter < self.idx + self.number_of_files_to_show:
                pass
            else:
                continue
            if micrograph in self.dataset.segmentation_paths:
                segmentation_path = self.dataset.segmentation_paths[micrograph]
            else:
                segmentation_path = None
            if micrograph in self.dataset.analysers:
                key = Path(micrograph).stem
                idxs = set(csv[csv["Micrograph"] == key]["Index"])
            else:
                idxs = None
            args.append((micrograph, segmentation_path, idxs))
        njobs = max(int(os.environ["CRYOVIA_NJOBS"]), 1)
        if njobs == 1:
            timer = {"message":0, "get":0, "after":0}
            for counter, (img, seg, idx) in enumerate(args):
                now = datetime.datetime.now()
                
                
                MESSAGE(f"{self.dataset.name}: Load files for inspection: {counter + 1}/{len(args)}\n",True)
                timer["message"] += (datetime.datetime.now() - now).total_seconds()
                now = datetime.datetime.now()
                res = MicrographData(img, seg, False, indeces=idx)
        
                self.model.previews.append(res)
                timer["after"] += (datetime.datetime.now() - now).total_seconds()
                for k,v in res.timer.items():
                    if k not in timer:
                        timer[k] = 0
                    timer[k] += v


        else:
            with mp.get_context("spawn").Pool(njobs) as pool:
                result = [pool.apply_async(getMicrographDataMP, [img, seg, idx]) for (img, seg, idx) in args ]
                current = -1
                timer = {"message":0, "get":0, "after":0}
                for counter, res in enumerate(result):
                    now = datetime.datetime.now()
                    
                    
                    MESSAGE(f"{self.dataset.name}: Load files for inspection: {counter + 1}/{len(result)}\n",True)
                    timer["message"] += (datetime.datetime.now() - now).total_seconds()
                    now = datetime.datetime.now()
                    res = res.get()
                    timer["get"] += (datetime.datetime.now() - now).total_seconds()
                    now = datetime.datetime.now()
                    res:MicrographData
                    res.applyAfterMP()
            
                    self.model.previews.append(res)
                    timer["after"] += (datetime.datetime.now() - now).total_seconds()
                    for k,v in res.timer.items():
                        if k not in timer:
                            timer[k] = 0
                        timer[k] += v


        self.model.layoutChanged.emit()
        self.indexChanged()
        self.view.resizeRowsToContents()
        self.view.resizeColumnsToContents()

    def indexChanged(self):
        total_length = len(self.dataset.micrograph_paths)
        bottom = self.idx
        top = min(self.idx + self.number_of_files_to_show, total_length)

        index_string = f"{bottom} - {top} / {total_length}"
        self.showIndexLabel.setText(index_string)

        if self.idx == 0:
            self.moveIndexDownButton.setEnabled(False)
        else:
            self.moveIndexDownButton.setEnabled(True)

        if self.idx + self.number_of_files_to_show >= len(self.dataset.micrograph_paths):
            self.moveIndexUpButton.setEnabled(False)
        else:
            self.moveIndexUpButton.setEnabled(True)

    def changeIndex(self, direction):
        self.idx += int(direction * 100)
        self.load_files()

    def closeEvent(self, a0) -> None:
        csv:pd.DataFrame = self.dataset.csv
        if self.to_remove is not None:
            self.dataset.removeMicrographPaths(paths=self.to_remove, )
            for path in self.to_remove:
                micrograph = Path(path).stem
                csv.drop(index=csv[csv["Micrograph"] == micrograph].index, inplace=True)
        for key, value in self.membranesToRemove.items():
            
            if key in self.dataset.analysers and self.dataset.analysers[key] is not None:
                micrograph = Path(key).stem
                indexes = set(csv[csv["Micrograph"] == micrograph]["Index"])
                indexes = indexes.difference(set(value))

                a:Analyser = Analyser.load(self.dataset.analysers[key])
                a.remove_all_other_indexes(indexes)
                a.save()
                w:AnalyserWrapper = Analyser.load(self.dataset.analysers[key], type_="Wrapper")
                csv.drop(index=csv[csv["Micrograph"] == micrograph].index, inplace=True)
                current_csv = w.csv
                csv = pd.concat((csv, current_csv))

            
        if self.to_remove is not None or len(self.membranesToRemove.keys()) > 0:
            self.dataset.to_csv(5, csv)
        self.customParent.inspectClosed()
        return super().closeEvent(a0)



    def resizeEvent(self, a0) -> None:

        # self.model.modelReset.emit()
        self.model.layoutChanged.emit()
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()
        return super().resizeEvent(a0)



class InspectionWindow(QWidget):
    def __init__(self, shown_data,dataset, tableview, parent, startingShape=(160,80)):
        super().__init__()
        self.customParent = parent
        self.data:pd.DataFrame = shown_data
        self.to_remove = None
        self.dataset = dataset
        self.tableview = tableview
        
        
        self.view = InspectionTableView()
        self.view.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        
        self.view.horizontalHeader().hide()
        self.view.verticalHeader().hide()
        self.view.setGridStyle(Qt.NoPen)

        self.delegate = PreviewDelegate(shape=startingShape)
        
        self.view.setItemDelegate(self.delegate)
        self.model = PreviewModel(self)
        self.view.setModel(self.model)
        
        self.view.setSelectionModel(customSelectionModel(self.model))
        self.view.selectionModel().selectionChanged.connect(self.selectionChanged)
        self.selectionLabel = QLabel("Selected: 0/0")

        # palette = QPalette()
        # palette.setColor(QPalette.Highlight, QColor("red"))

        # self.view.setPalette(palette)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.selectionLabel)
        self.layout().addWidget(self.view)
        # self.setSizePolicy(QSizePolicy(QSizePolicy.Fixed,QSizePolicy.Fixed))
        self.scroll_timer = QTimer()
        self.scroll_timer.setInterval(500)  # Adjust the interval as needed
        self.scroll_timer.setSingleShot(True)
        self.scroll_timer.timeout.connect(self.on_scroll_stopped)
        self.scrollCounter = 0

        self.load_files()
    

    def on_scroll_stopped(self):
        if self.scrollCounter != 0:
            self.delegate.shape += np.array([10,5]) * self.scrollCounter
            if self.delegate.shape[0] < 40 or self.delegate.shape[1] <20:
                self.delegate.shape = np.array([40,20])
            elif self.delegate.shape[0] > 1280 or self.delegate.shape[1] >640:
                self.delegate.shape = np.array([1280,640])
            self.model.layoutChanged.emit()
            self.view.resizeColumnsToContents()
            self.view.resizeRowsToContents()
            self.scrollCounter = 0
            
    def wheelEvent(self, a0: QWheelEvent) -> None:
        
        modifiers = QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ControlModifier:
            if a0.angleDelta().y() > 0:
                self.scrollCounter -= 1
            elif a0.angleDelta().y() < 0:
                self.scrollCounter += 1
            
            self.scroll_timer.start()
            a0.accept()
            return
        return super().wheelEvent(a0)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:

            idxs = self.view.selectionModel().selectedIndexes()
            
            new_idxs = [idx.row() * self.model.columnCount() + idx.column() for idx in idxs]
            pd_idxs = self.model.deleteIdxs(new_idxs)

            self.data = self.data.drop(index=pd_idxs)
            if self.to_remove is None:
                self.to_remove = pd_idxs
            else:
                self.to_remove.extend(pd_idxs)

            self.view.selectionModel().clearSelection()
            # row = self.currentRow()
            # self.removeRow(row)
        else:
            super().keyPressEvent(event)
        self.selectionChanged(None)

    # def sizeHint(self):
    #     # All items the same size.
    #     return QSize(1250, 850)
        
    def selectionChanged(self, event=None):
        idxs = self.view.selectionModel().selectedIndexes()
        new_text = f"Selected: {len(idxs)}/{len(self.model.previews)}"
        self.selectionLabel.setText(new_text)

    def load_files(self):

        for n, row in enumerate(self.data.iterrows()):
            i, row = row
            item = ImageData(row, self.dataset, index=i)
            self.model.previews.append(item)


        self.model.layoutChanged.emit()

        self.view.resizeRowsToContents()
        self.view.resizeColumnsToContents()


    def closeEvent(self, a0) -> None:
        if self.to_remove is not None:
            self.tableview.removeIdxs(self.to_remove, self.dataset, ret=1)
        self.customParent.inspectClosed()
        return super().closeEvent(a0)



    def resizeEvent(self, a0) -> None:

        # self.model.modelReset.emit()
        self.model.layoutChanged.emit()
        self.view.resizeColumnsToContents()
        self.view.resizeRowsToContents()
        return super().resizeEvent(a0)



class MicrographInspectionButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text)
        self.clicked.connect(self.openInspectionWindow)
        self.InspectionWindow = None
        self.setToolTip("Inspect the micrographs to remove unwanted images.")
        

    def mainWindow(self):
        return self.parent().parent().parent().parent().parent()
    
    def openInspectionWindow(self):
        global MESSAGE
        if self.InspectionWindow is None:
            listwidget : DatasetListWidget= self.parent().parent().listWidget

            items = listwidget.selectedItems()
            if len(items) == 1:
                dataset = items[0].dataset
                if dataset.isZipped:
                    MESSAGE(f"Cannot inspect dataset {dataset.name} because it is still zipped.")
                    return
           
                self.InspectionWindow = MicrographInspectionWindow(dataset,self )
                self.parent().parent().setEnabled(False)
                self.InspectionWindow.show()
        
        

    def inspectClosed(self):
        self.InspectionWindow = None
        self.parent().parent().setEnabled(True)
        



class InspectionButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text)
        self.clicked.connect(self.openInspectionWindow)
        self.InspectionWindow = None
        self.setToolTip("Open up a window to inspect the segmentation from the currently filtered membranes. You can select membranes and delete them with the delete key.")
        

    def mainWindow(self):
        return self.parent().parent().parent().parent().parent()
    
    def openInspectionWindow(self):
        if self.InspectionWindow is None:
            tab = self.mainWindow().datasetTabWidget.currentWidget()
            if tab is None:
                return
            shown_data = tab.model().shown_data
            self.InspectionWindow = InspectionWindow(shown_data, tab.model().dataset, tab, parent=self)
            self.parent().setEnabled(False)
            self.InspectionWindow.show()
        
        

    def inspectClosed(self):
        self.InspectionWindow = None
        self.parent().setEnabled(True)
        


class DatasetGui(QWidget):
    def __init__(self, parent=None, custom_parent=None):
        global MESSAGE
        super().__init__(parent)


        self.setLayout(QVBoxLayout())
        self.horizontalSplitter = QSplitter(Qt.Vertical, self)

        self.rowOne = QSplitter(Qt.Horizontal, self)
        self.rowTwo = QSplitter(Qt.Horizontal, self)
        # self.rowOne.setStyleSheet("QSplitter::handle {background: red;}")

        self.rowOne.setStyleSheet("QSplitter::handle {border: 1px dashed grey;margin: 5px 5px; min-width: 10px;max-width: 10px;};")
        self.rowTwo.setStyleSheet("QSplitter::handle {border: 1px dashed grey;margin: 5px 5px; min-width: 10px;max-width: 10px;};")
        self.horizontalSplitter.setStyleSheet("QSplitter::handle {border: 1px dashed grey;margin: 5px 5px; min-width: 10px;max-width: 10px;};")



        # self.rowThree = QSplitter(Qt.Vertical, self)

        # self.rowOne.splitterMoved.connect(self.moveSplitter)
        # self.rowTwo.splitterMoved.connect(self.moveSplitter)
        # self.rowThree.splitterMoved.connect(self.moveSplitter)


        self.customParent=custom_parent



        self.graphWidget = graphWidget(self)
        self.graphWidgetGroupBox = customGroupBox("Graphs", self)
        self.graphWidgetGroupBox.setLayout(QHBoxLayout())
        self.graphWidgetGroupBox.layout().addWidget(self.graphWidget)


        self.datasetInfoWidget = DatasetInfoWidget(self)
        self.datasetInfoWidgetGroupBox = customGroupBox("Dataset info", self)
        self.datasetInfoWidgetGroupBox.setLayout(QHBoxLayout())
        self.datasetInfoWidgetGroupBox.layout().addWidget(self.datasetInfoWidget)

        self.datasetTabWidget = DatasetTabsWidget(self)
        self.datasetTabWidgetGroupBox = customGroupBox("Dataset attributes", self)
        self.datasetTabWidgetGroupBox.setLayout(QHBoxLayout())
        self.datasetTabWidgetGroupBox.layout().addWidget(self.datasetTabWidget)
        # idxs = [-6,-5,-4,-1]
        # for idx in idxs:
        #     dataset:Dataset = Dataset.load(datasets[idx])
        #     self.datasetTabWidget.addDatasetTab(dataset, 50)

        self.filterwidget = FilterListWidget(self)
        self.ButtonWidget = ListFilterButtons(self, self.filterwidget)

        self.datasetListWidget = DatasetsListPlusButtons(self)
        self.datasetListWidgetGroupBox = customGroupBox("Datasets", self)
        self.datasetListWidgetGroupBox.setLayout(QHBoxLayout())
        self.datasetListWidgetGroupBox.layout().addWidget(self.datasetListWidget)
        
        self.runButtonWidget = runButtonWidget(self)
        

        self.membraneWidget = thumbnailWidget(self)
        self.membraneWidgetGroupBox = customGroupBox("Currently selected membrane", self)
        self.membraneWidgetGroupBox.setLayout(QHBoxLayout())
        self.membraneWidgetGroupBox.layout().addWidget(self.membraneWidget)

        self.openManualInspectionButton = InspectionButton("Manual Inspection")
        self.exportDataButton = QPushButton("Export single table")
        self.exportAllDataButton = QPushButton("Export all tables")
        self.exportDataButton.clicked.connect(self.datasetTabWidget.exportData)
        self.exportAllDataButton.clicked.connect(self.datasetTabWidget.exportAllData)

        self.quickStatsButton = QPushButton("Quick stats")
        self.quickStatsButton.clicked.connect(self.datasetTabWidget.printStats)

        self.lowerButtonLayout = QGridLayout()
        self.lowerButtonLayout.addWidget(self.openManualInspectionButton,1,0)
        self.lowerButtonLayout.addWidget(self.exportDataButton,0,0)
        self.lowerButtonLayout.addWidget(self.exportAllDataButton,0,1)
        self.lowerButtonLayout.addWidget(self.quickStatsButton, 1,1)



        self.messageBoardGroupBox = customGroupBox("Message board", self)
        self.messageBoardGroupBox.setLayout(QHBoxLayout())


        self.messageBoard = QTextEdit(self)
        self.messageBoard.setReadOnly(True)
        self.messageBoard.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        
        self.messageBoard.font().setFamily("Courier")
        self.messageBoard.font().setPointSize(10)
        self.messageBoardGroupBox.layout().addWidget(self.messageBoard)


        # self.applyButton = QPushButton("Apply")
        # self.applyButton.clicked.connect(self.filterwidget.applyFilters)




        self.rowOne.addWidget(self.datasetInfoWidgetGroupBox)
        self.rowOne.addWidget(self.filterwidget)
        self.rowOne.addWidget(self.graphWidgetGroupBox)

        

        self.rowTwoColumnOne = QWidget()
        self.rowTwoColumnOne.setLayout(QVBoxLayout())
        self.rowTwoColumnOne.layout().addWidget(self.runButtonWidget)

        self.rowTwoColumnOne.layout().addWidget(self.datasetListWidgetGroupBox)
        self.rowTwo.addWidget(self.rowTwoColumnOne)

        self.rowTwoColumnTwo = QWidget()
        self.rowTwoColumnTwo.setLayout(QVBoxLayout())
        self.rowTwoColumnTwo.layout().addWidget(self.ButtonWidget)
        self.rowTwoColumnTwo.layout().addWidget(self.datasetTabWidgetGroupBox)

        self.lowerButtonLayoutWidget = QWidget()
        self.lowerButtonLayoutWidget.setLayout(self.lowerButtonLayout)
        self.rowTwoColumnTwo.layout().addWidget(self.lowerButtonLayoutWidget)
        self.rowTwo.addWidget(self.rowTwoColumnTwo)
        self.rowTwo.addWidget(self.membraneWidgetGroupBox)
        
        # self.rowThree.addWidget(self.messageBoardGroupBox)

        self.horizontalSplitter.addWidget(self.rowOne)
        self.horizontalSplitter.addWidget(self.rowTwo)
        self.horizontalSplitter.addWidget(self.messageBoardGroupBox)
        self.layout().addWidget(self.horizontalSplitter)
        # self.layout().addWidget(self.messageBoardGroupBox)

        # self.layout().addWidget(self.datasetListWidgetGroupBox, 2,0,2,1)
        # self.layout().addWidget(self.filterwidget, 0,1)
        # self.layout().addWidget(self.ButtonWidget,1,1)
        # self.layout().addWidget(self.runButtonWidget, 1,0)
        # self.layout().addWidget(self.datasetInfoWidgetGroupBox, 0,0)
        # self.layout().addWidget(self.datasetTabWidgetGroupBox,2,1)
        # self.layout().addWidget(self.graphWidgetGroupBox, 0, 2)
        
        # self.layout().addWidget(self.membraneWidgetGroupBox, 1,2,3,1)
        # self.layout().addLayout(self.lowerButtonLayout, 3, 1)
        # self.layout().addWidget(self.messageBoardGroupBox, 4,0, 1,3)
        MESSAGE = self.newMessage

        # self.layout().setColumnStretch(0,0)
        # self.layout().setColumnStretch(1,1)
        # self.layout().setColumnStretch(2,0)
        
        self.rowOneWidgets = [self.datasetInfoWidgetGroupBox, self.filterwidget, self.graphWidgetGroupBox]
        self.rowTwoWidgets = [self.rowTwoColumnOne, self.rowTwoColumnTwo, self.membraneWidgetGroupBox]

        
        # self.layout().setRowStretch(0,0)
        # self.layout().setRowStretch(1,0)
        # self.layout().setRowStretch(2,1)
        # self.layout().setRowStretch(3,0)

        
    


    def moveSplitter( self, index, pos ):
        for splt in [self.rowTwo, self.rowTwo]:
            if self.sender() == splt:
                continue
            # splt = self._spltA if self.sender() == self._spltB else self._spltB
            splt.blockSignals(True)
            splt.moveSplitter(index, pos)
            splt.blockSignals(False)



    def closeEvent(self, a0) -> None:
        global CURRENTLY_RUNNING
        
        if len(CURRENTLY_RUNNING) > 0:
            qm = QMessageBox()
            qm.setText(f"Analysis is still running.")
            qm.setIcon(qm.Icon.Warning)
            qm.addButton(QPushButton("Close windows but still run in background."),  QMessageBox.ButtonRole.YesRole)

            qm.addButton(QPushButton("Close windows and cancel analysis"), QMessageBox.ButtonRole.AcceptRole)
            # qm.addButton(QPushButton("Remove for future analysis"), QMessageBox.ButtonRole.ActionRole)
            qm.addButton(QPushButton("Cancel"), QMessageBox.ButtonRole.RejectRole)
            # qm.setCheckBox()

            
            ret = qm.exec_()
            if ret == 2:
                a0.ignore()
                return
            if ret == 1:
                self.runButtonWidget.stopThread()
                self.datasetListWidget.listWidget.stopThread()
            elif ret == 0:
                self.runButtonWidget.stopThread(keep_running=True)
                self.datasetListWidget.listWidget.stopThread()
        else:
            self.runButtonWidget.stopThread()
            self.datasetListWidget.listWidget.stopThread()

        if self.customParent is not None:
            self.customParent.child_closed()
        
        return super().closeEvent(a0)

    def newMessage(self, message:str, update=False):
        cursor = self.messageBoard.textCursor()
        cursor.insertText(message)
        self.messageBoard.moveCursor(QTextCursor.End)
        if update:
            QApplication.processEvents()

if __name__ == "__main__":




    app = QApplication(sys.argv)
    view = DatasetGui()
    
    view.show()
    sys.exit(app.exec_())









# class EditableLineEdit(QLineEdit):
#     def __init__(self, text, parent=None):
#         super().__init__(text, parent)
#         self.setReadOnly(True)
#         self.setFrame(False)
#         self.setStyleSheet("QLineEdit { border: none; }")  # Remove the border
#         self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # Make the line edit transparent to mouse events initially

#     def mouseDoubleClickEvent(self, event):
#         self.setAttribute(Qt.WA_TransparentForMouseEvents, False)  # Allow the line edit to receive mouse events
#         self.setReadOnly(False)
#         self.setFocus()
#         self.selectAll()

#     def focusOutEvent(self, event):
#         super().focusOutEvent(event)
#         self.setReadOnly(True)
#         self.setAttribute(Qt.WA_TransparentForMouseEvents, True)  # Make it transparent to mouse events again

# class CustomListItem(QWidget):
#     def __init__(self, text, pixmap):
#         super().__init__()

#         # Create a horizontal layout with no margins and spacing
#         layout = QHBoxLayout()
#         layout.setContentsMargins(0, 0, 0, 0)
#         layout.setSpacing(5)  # Set spacing between widgets

#         # Create an editable line edit for the text and add it to the layout
#         self.editable_line_edit = EditableLineEdit(text)
#         layout.addWidget(self.editable_line_edit)

#         # Create a label for the pixmap and add it to the layout
#         pixmap_label = QLabel()
#         pixmap_label.setPixmap(pixmap)
#         layout.addWidget(pixmap_label)

#         # Add stretch to push the pixmap to the right side
#         layout.addStretch()

#         # Set the layout to the custom widget
#         self.setLayout(layout)

#         # Optionally, set a fixed height to make it more compact
#         self.setFixedHeight(30)

#     def get_text(self):
#         return self.editable_line_edit.text()

#     def set_selected(self, selected):
#         if selected:
#             self.setStyleSheet("background-color: lightblue;")
#         else:
#             self.setStyleSheet("")

# class Example(QWidget):
#     item_selection_changed = pyqtSignal()

#     def __init__(self):
#         super().__init__()
#         self.initUI()

#     def initUI(self):
#         # Create a QListWidget
#         self.list_widget = QListWidget()

#         # Connect the selection change signal
#         self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)

#         # Create a QPixmap
#         pixmap = QPixmap(16, 16)
#         pixmap.fill(QColor('green'))

#         # Create a custom list item with text and pixmap
#         custom_item = CustomListItem("Sample Text", pixmap)

#         # Create a QListWidgetItem
#         list_item = QListWidgetItem(self.list_widget)

#         # Set the size hint of the list item to match the custom widget
#         list_item.setSizeHint(custom_item.sizeHint())

#         # Add the custom widget to the QListWidget
#         self.list_widget.setItemWidget(list_item, custom_item)

#         # Button to get selected items
#         button = QPushButton("Get Selected Items")
#         button.clicked.connect(self.get_selected_items)

#         # Create a vertical layout and add the list widget to it
#         layout = QVBoxLayout()
#         layout.addWidget(self.list_widget)
#         layout.addWidget(button)
#         self.setLayout(layout)

#         self.setWindowTitle('Custom List Widget Item')
#         self.show()

#     def get_selected_items(self):
#         selected_items = self.list_widget.selectedItems()
#         for item in selected_items:
#             custom_widget = self.list_widget.itemWidget(item)
#             if custom_widget:
#                 print(f"Selected item text: {custom_widget.get_text()}")

#     def on_selection_changed(self):
#         for i in range(self.list_widget.count()):
#             item = self.list_widget.item(i)
#             custom_widget = self.list_widget.itemWidget(item)
#             if custom_widget:
#                 custom_widget.set_selected(item.isSelected())