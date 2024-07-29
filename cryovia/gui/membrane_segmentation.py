import sys
# import sparse
from scipy import sparse
from scipy.ndimage import label
from scipy.special import softmax
# from PyQt5.QtCore    import *



import cryovia.gui.segmentation_files.curvature_skeletonizer as cs

from keras.callbacks import Callback
from PyQt5.QtCore import Qt, QPoint, QSize, QByteArray, QModelIndex,pyqtSignal, QObject,QThread
# from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QApplication, QWidget, QAction, QMainWindow, QPushButton, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QComboBox, QListWidget, QGridLayout, QListWidgetItem, QStyle, QGroupBox
from PyQt5.QtWidgets import QInputDialog, QMessageBox, QAbstractItemView, QItemDelegate, QTextEdit, QTableWidget, QTableWidgetItem, QFileDialog, QSpinBox, QCheckBox, QTabWidget, QDialogButtonBox, QDialog, QSizePolicy
# from PyQt5.QtGui     import *
from PyQt5.QtGui import QIcon, QPainter, QPen,QPixmap, QColor, QImage,QValidator, QTextCursor, QIntValidator, QDoubleValidator, QPalette
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import typing
import os
import shutil
import traceback
from cryovia.gui.segmentation_files.segmentation_model import *
from cryovia.gui.napari_seg_helper import SegmentationHelper
# from cryovia.gui.napari_plugin import SegmentationHelper
import pyqtgraph as pyqtg
import multiprocessing
from sklearn.metrics import confusion_matrix  

from cryovia.gui.segmentation_files.prep_training_data import *
from grid_edge_detector import image_gui as ged
from grid_edge_detector.carbon_edge_detector import find_grid_hole_per_file
from sklearn.metrics import classification_report



def resizeSegmentation(image, shape):
    return resize(image, shape, 0)

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

            # QGroupBox::


class IoUModelWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, model, config, gpu, cores, name,testData):
        super().__init__()
        self.model = model

        self.config = config
        self.gpu = gpu
        self.cores = cores
        self.testData = testData
        self.name = name
        self.toStop = False

    def normalize(self, x):
        means = [np.mean(img) for img in x]
        stds = [np.std(img) for img in x]
        
        x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
        return x

    def print(self, msg):
        self.progress.emit(("print", f"{self.name}: {msg}"))

    # def emitCallback(self, epoch, logs):
    #     self.progress.emit(("epoch_logs", (epoch,logs)))


    def stop(self, name):
        self.print(name, self.model.name)
        if name == self.model.name:
            self.print("Stopping...")
            self.toStop = True


    def run(self):
        import tensorflow as tf
        import tensorflow.keras as keras

        def compute_iou(y_pred, y_true):
            # ytrue, ypred is a flatten vector
            y_pred = np.array(y_pred).flatten()
            y_true = np.array(y_true).flatten()
            current = confusion_matrix(y_true, y_pred, labels=[0, 1])
            # compute mean iou
            intersection = np.diag(current)
            ground_truth_set = current.sum(axis=1)
            predicted_set = current.sum(axis=0)
            union = ground_truth_set + predicted_set - intersection
            IoU = intersection / union.astype(np.float32)
            return np.nanmean(IoU)

        def test_batch_size(batch_size, model):
            x_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,1), dtype=np.float32)
            # y_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,2), dtype=np.float32)
            try:
                model.predict(x=x_batch, batch_size=batch_size, verbose=0)
                return True
            except Exception as e:
                return False
        
            



        self.print("Predicting")

        # self.currently_training = True
        
        
        with tf.device(self.gpu):  
            
            
            # optimizer = self.optimizer(learning_rate=self.config.train_learning_rate)
            # loss = self.loss([1.0,5.0],2)

            # self.model.compile(optimizer, loss)

            self.print("Checking batch size")
            current_batch_size = self.config.max_batch_size
            working_batch_size = None
            tried_batch_sizes = set()
            while current_batch_size not in tried_batch_sizes:
                tried_batch_sizes.add(current_batch_size)
                if current_batch_size == 0:
                    self.currently_training = False
                    return 
                if test_batch_size(current_batch_size, self.model):
                    working_batch_size = current_batch_size
                    current_batch_size *= 2
                else:
                    current_batch_size = current_batch_size // 2
            
            self.print(f"Found fitting batch size: {working_batch_size}")
            # optimizer = self.optimizer(learning_rate=self.config.train_learning_rate)
            # self.model.compile(optimizer, loss)
            differences = []
            # for path, seg_path in zip(self.testData["images"], self.testData["segmentations"]):
            turned_patches_total = []


            dataset = customDatasetForPerformance(self.testData["images"], self.testData["segmentations"],working_batch_size, self.config, ".", "IoUTest", njobs=int(os.environ["CRYOVIA_NJOBS"]),stepSize=self.config.input_shape,shuffle=False, flip=False)
            # x_pred, shape = loadPredictData([path], self.config, )

            predictions = []
            xs, ys = [], []

            for i in range(len(dataset)):
                x, y = dataset[i]
                predictions.append(self.model.predict(x, verbose=0))
                xs.append(x)
                ys.append(y)
            
            output_dir = dataset.path / "predictions"
            output_dir.mkdir(exist_ok=True)

            predictions = np.concatenate(predictions)
            
            # predictions = np.squeeze(np.concatenate(predictions))
            # predictions = np.argmax(predictions, -1)
            xs = np.concatenate(xs)
            
            # xs = np.squeeze(np.concatenate(xs))
            # ys = np.concatenate(ys)
            ys = np.squeeze(np.concatenate(ys))
            ys = np.argmax(ys, -1)


            
            dataset.clean()
            gt = []
            predicted_patches = []
            start = 0

            for file_count in dataset.numberOfFilesPerMicrograph:
                for i in range(file_count):
                    preds = [predictions[start + i + j * file_count] for j in range(4)]
                    turned_pred = [np.rot90(pred, i%4,(-3,-2)) for pred,i in zip(preds, range(4,0,-1))]

                    # for rot, rot_ in enumerate(turned_pred):
                    #     plt.imsave(output_dir / f"{start}_{i}_{rot}.png", np.squeeze(np.argmax(rot_,-1)), cmap="gray")
                    prediction = np.sum(turned_pred, 0)
                    prediction = np.squeeze(np.argmax(prediction, -1))
                    predicted_patches.append(prediction)
                    gt.append(ys[start + i])
                # predicted_patches = np.array(predicted_patches)
                start += file_count * 4
            # for counter, (i,j) in enumerate(zip(predicted_patches, gt)):
            #     plt.imsave(output_dir / f"{counter}.png",i , cmap="gray")
            #     plt.imsave(output_dir / f"{counter}_gt.png",j , cmap="gray")
            self.print(str(compute_iou(predicted_patches, gt)))
            gt = np.array(gt)
            predicted_patches = np.array(predicted_patches)
  
                
                
        self.finished.emit()





class PredictModelWorker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, file_paths, model, config, gpu, cores, name, number_of_files, save_dir=None, prediction_params={},pixelSizes={}):
        super().__init__()
        self.file_paths = file_paths
        self.model = model
        # self.optimizer = optimizer
        # self.loss = loss
        self.config = config
        self.gpu = gpu
        self.cores = cores
        self.pixelSizes = pixelSizes

        self.name = name
        self.number_of_files = number_of_files
        self.save_dir = save_dir
        self.prediction_params = prediction_params
        self.toStop = False
        # self.threshold = threshold

    def normalize(self, x):
        means = [np.mean(img) for img in x]
        stds = [np.std(img) for img in x]
        
        x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
        return x

    def print(self, msg):
        self.progress.emit(("print", f"{self.name}: {msg}"))

    # def emitCallback(self, epoch, logs):
    #     self.progress.emit(("epoch_logs", (epoch,logs)))


    def stop(self, name):
        self.print(name, self.model.name)
        if name == self.model.name:
            self.print("Stopping...")
            self.toStop = True


    def run(self):
        import tensorflow as tf
        import tensorflow.keras as keras
        def test_batch_size(batch_size, model):
            x_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,1), dtype=np.float32)
            # y_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,2), dtype=np.float32)
            try:
                model.predict(x=x_batch, batch_size=batch_size, verbose=0)
                return True
            except Exception as e:
                return False
        
            



        self.print("Predicting")

        # self.currently_training = True
        
        
        with tf.device(self.gpu):  
            
            
            # optimizer = self.optimizer(learning_rate=self.config.train_learning_rate)
            # loss = self.loss([1.0,5.0],2)

            # self.model.compile(optimizer, loss)

            self.print("Checking batch size")
            current_batch_size = self.config.max_batch_size
            working_batch_size = None
            tried_batch_sizes = set()
            while current_batch_size not in tried_batch_sizes:
                tried_batch_sizes.add(current_batch_size)
                if current_batch_size == 0:
                    self.currently_training = False
                    return 
                if test_batch_size(current_batch_size, self.model):
                    working_batch_size = current_batch_size
                    current_batch_size *= 2
                else:
                    current_batch_size = current_batch_size // 2
            
            self.print(f"Found fitting batch size: {working_batch_size}")
            # optimizer = self.optimizer(learning_rate=self.config.train_learning_rate)
            # self.model.compile(optimizer, loss)
            differences = []
            for path in self.file_paths:
                turned_patches_total = []

                x_pred, shape = loadPredictData([path], self.config,ps=self.pixelSizes[path] )
        

                x_pred = self.normalize(x_pred)
                x_pred = x_pred[0]
                shape = shape[0]

                for patch in x_pred:
                    turned_patches = [np.rot90(patch, i,(0,1)) for i in range(4)]
                    turned_patches_total.extend(turned_patches)
                turned_patches = np.array(turned_patches_total)
                
                prediction = []
                for i in range(0, len(turned_patches), working_batch_size):
                    prediction.append(self.model.predict(turned_patches[i:i+working_batch_size], batch_size=working_batch_size, verbose=0))

                    gc.collect()
                    if self.toStop:
                        self.finished.emit()
                        return

                prediction = np.concatenate(prediction)


                prediction = self.model.predict(turned_patches, batch_size=working_batch_size, verbose=0)

                if self.for_active_learning:
                    difference = np.sum(np.abs((prediction[..., 0] - prediction[..., 1])))

                    differences.append(difference)
                else:
                    predictions = softmax(prediction, -1)
                    predicted_patches = []
                    for i in range(len(x_pred)):
                        turned_pred = [np.rot90(pred, i%4) for pred,i in zip(predictions[i*4:i*4+4], range(4,0,-1))]

                        prediction = np.sum(turned_pred, 0)

                        predicted_patches.append(prediction)
                    predicted_patches = np.array(predicted_patches)
                    predicted_image = unpatchify(predicted_patches,shape, self.config, threshold=True)
                    mask = None
                    if self.prediction_params["grid_remover"]["apply"]:
                        self.print("Applying grid remover")
                        parameters = self.prediction_params["grid_remover"]["params"]
                        parameters["Pixel spacing"] = self.pixelSizes[path]
                        mask = find_grid_hole_per_file(path, parameters["to_size"], 10000 * parameters["gridsizes"], parameters["threshold"],
                                                                coverage_percentage=parameters["inner_circle_coverage"],outside_coverage_percentage=parameters["outside_circle_coverage"],
                                                                  ring_width=parameters["ring_width"]*10000,detect_ring=parameters["detect_ring"], pixel_size=parameters["Pixel spacing"], wobble=parameters["wobble"],
                                                                  high_pass=parameters["high_pass_filter"])
                        mask = resizeSegmentation(mask, predicted_image.shape)
                        
                        if self.prediction_params["save_intermediate_files"]:
                            save_path = self.save_dir / (path.stem + "_original_segmentation" +  self.prediction_params["suffix"]) 
                            save_file(save_path,predicted_image, self.config.pixel_size)
                        predicted_image *= mask
                        if self.prediction_params["save_intermediate_files"]:
                            save_path = self.save_dir / (path.stem + "_original_segmentation_masked" +  self.prediction_params["suffix"]) 
                            save_file(save_path,predicted_image, self.config.pixel_size)
                        

                    if self.config.filled_segmentation:
                        predicted_image = get_contour_of_filled_segmentation(predicted_image)

                    
                    if self.prediction_params["instance_identification"]["apply"]:
                        if self.config.filled_segmentation:
                            self.print("No need to identify instances with filled segmentations")
                        else:
                            self.print("Applying instance identification")
                            if self.prediction_params["save_intermediate_files"] and not self.prediction_params["grid_remover"]["apply"]:
                                save_path = self.save_dir / (path.stem + "_original_segmentation" +  self.prediction_params["suffix"]) 
                                save_file(save_path,predicted_image, self.config.pixel_size)
                            try:
                                max_nodes = 30
                                if "max_nodes" in self.prediction_params["instance_identification"]:
                                    max_nodes = self.prediction_params["instance_identification"]["max_nodes"]
                                predicted_image, skeletons = cs.solve_skeleton_per_job(predicted_image,mask, only_closed=True, max_nodes=max_nodes)
                            except Exception as e:
                                raise e
                    
                    save_path = self.save_dir / (path.stem + "_segmentation" +  self.prediction_params["suffix"]) 
                    save_file(save_path,predicted_image, self.config.pixel_size)
                    

        if self.for_active_learning:
            idxs = np.argsort(differences)
            self.print(differences)
            self.print(idxs)
            if self.number_of_files > len(differences):
                return_files = self.file_paths
            else:
                return_files = [self.file_paths[idx] for idx in idxs[:self.number_of_files]]
                self.progress.emit(("active_learning_files",return_files))
        self.finished.emit()

    @property
    def for_active_learning(self):
        return self.save_dir is None





class StopCallback(Callback):
    def __init__(self, worker) -> None:
        super().__init__()
        self.worker = worker
        # self.funcs_on_epoch_end_test = funcs_on_epoch_end_test

    @property
    def toStopOnNext(self):
        return self.worker.toStop

    def on_train_batch_end(self, batch, logs=None):
        if self.toStopOnNext:
            self.model.stop_training = True
            


class TrainModelWorker(QObject):

    finished = pyqtSignal()
    progress = pyqtSignal(tuple)
    def __init__(self, file_paths, model, loss, config, gpu, cores, callbacks, name, pixelSizes):
                

        super().__init__()
        self.file_paths = file_paths
        self.model = model
        # self.optimizer = optimizer
        self.loss = loss
        self.config = config
        self.gpu = gpu
        self.cores = cores
        self.callbacks = callbacks
        self.pixelSizes = pixelSizes
        self.toStop = False
        self.callbacks.append(customCallback([self.emitCallback]))
        self.name = name
        self.stopCallback = StopCallback(self)
        self.callbacks.append(self.stopCallback)
        # self.threshold = threshold

    def normalize(self, x):
        means = [np.mean(img) for img in x]
        stds = [np.std(img) for img in x]
        
        x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
        return x

    def print(self, msg):
        self.progress.emit(("print", f"{self.name}: {msg}"))

    def emitCallback(self, epoch, logs):
        self.progress.emit(("epoch_logs", (epoch,logs)))

    def stop(self, name):
        # print(name, self.model.name, self.name)
        if name == self.name:
            self.print("Stopping...")
            self.toStop = True

    def run(self):
        try:
            import tensorflow as tf
            import tensorflow.keras as keras
            from tensorflow.keras.optimizers import Adam
            def test_batch_size(batch_size, model):
                x_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,1), dtype=np.float32)
                y_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,2), dtype=np.float32)
                try:
                    model.fit(x=x_batch, y=y_batch, batch_size=batch_size, epochs=1, verbose=0, validation_split=0)
                    return True
                except Exception as e:
                    return False
            
            def get_print_callback(max_epoch):

                callback = customCallback([], [], [{"name":"epoch_start", "value":0}])

                def print_callback(epoch, logs):
                    starting_date = callback.epoch_start
                    end_date = datetime.datetime.now()
                    took = round((end_date - starting_date).total_seconds(),2)
                    loss = round(logs["loss"], 4)
                    if "val_loss" in logs:

                        valid_loss = round(logs["val_loss"],4)
                    else:
                        valid_loss = None
                    lr = "{:.4e}".format(logs["lr"])
                    print_msg = f"Epoch {epoch+1}/{max_epoch}: {took}s - loss: {loss} - val_loss: {valid_loss} - learning rate: {lr}"
                    self.print(print_msg)
                
                def set_starting_date(epoch, logs):
                    callback.epoch_start = datetime.datetime.now()
                
                
                callback.funcs_on_epoch_begin_train = [set_starting_date]
                callback.funcs_on_epoch_end_train = [print_callback]
                
                return callback
                



            self.print("TRAINING")

            # self.currently_training = True
            self.file_paths["images"] = self.file_paths["images"]
            self.file_paths["segmentations"] = self.file_paths["segmentations"]
            

            # x_train = self.normalize(x_train)
            self.callbacks.append(get_print_callback(self.config.train_epochs))
            with tf.device(self.gpu):  
                # y_train = y_train.astype(np.float32)
                # callbacks = self.createCallbacks()
                # callbacks.extend(callback_functions)
                
                optimizer = Adam(learning_rate=self.config.train_learning_rate)
                loss = self.loss([1.0,5.0],2)

                self.model.compile(optimizer, loss)

                current_batch_size = self.config.max_batch_size
                working_batch_size = None
                tried_batch_sizes = set()
                while current_batch_size not in tried_batch_sizes:
                    tried_batch_sizes.add(current_batch_size)
                    if current_batch_size == 0:
                        self.currently_training = False
                        return 
                    if test_batch_size(current_batch_size, self.model):
                        working_batch_size = current_batch_size
                        current_batch_size *= 2
                    else:
                        current_batch_size = current_batch_size // 2
                    gc.collect()
                    tf.keras.backend.clear_session()
                    if self.toStop:
                        self.finished.emit()
                        return
                if self.config.max_batch_size < working_batch_size:
                    working_batch_size = self.config.max_batch_size
                gc.collect()
                tf.keras.backend.clear_session()
                self.print(f"Found fitting batch size: {working_batch_size}")
                self.print("Creating training files")
                
                train, valid = getTrainingDataForPerformance(self.file_paths["images"], self.file_paths["segmentations"], self.config, self.print, 0,0.25,1, working_batch_size, toStop=self.getShouldIStop(),njobs=self.cores, image_pixel_sizes=self.pixelSizes)
            
                try:
                    optimizer = Adam(learning_rate=self.config.train_learning_rate)
                    self.model.compile(optimizer, loss)
                    # history = self.model.fit(x=x_train[:32], y=y_train[:32], batch_size=working_batch_size,
                    #     epochs=self.config.train_epochs, verbose=0, validation_split=0.2, callbacks=self.callbacks)
                    if self.toStop:
                        train.clean()
                        valid.clean()
                        self.finished.emit()
                        
                        return
                    self.print("Starting training")
                    history = self.model.fit(x=train, batch_size=working_batch_size,
                        epochs=self.config.train_epochs, verbose=0, validation_data=valid, callbacks=self.callbacks)
                    
                    

                    train.clean()
                    valid.clean()
                except Exception as e:
                    train.clean()
                    valid.clean()
                    raise e
        except Exception as e:
            self.print(f"There was an error during training. {traceback.format_exc()}")

        self.finished.emit()

    def getShouldIStop(self):
        def shouldIStop():
            return self.toStop
        return shouldIStop



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


class CustomIntValidator(QIntValidator):
    def __init__(self, bottom, top) -> None:
        super().__init__()
        self.setBottom(bottom)
        self.setTop(top)
        
    def validate(self, a0: str, a1: int) -> typing.Tuple['QValidator.State', str, int]:
        if (self.bottom() > 0 or self.top() < 0) and all([letter == "0" for letter in a0]) and len(a0) > 0:
            return QValidator.State.Invalid, a0, a1
        if len(a0) == 0:
            return QValidator.State.Intermediate, a0, a1
        
        return super().validate(a0, a1)

    def fixup(self, inp: str) -> str:
        if len(inp) > 0 and int(inp) > self.top():
            return str(self.top())
        return str(self.bottom())



class CustomDoubleValidator(QDoubleValidator):
    def __init__(self, bottom, top) -> None:
        super().__init__()
        self.setBottom(bottom)
        self.setTop(top)
        
    def validate(self, a0: str, a1: int) -> typing.Tuple['QValidator.State', str, int]:
    
        if len(a0) == 0:
            return QValidator.State.Intermediate, a0, a1
        try:
            if float(a0) < self.bottom():
                return QValidator.State.Intermediate, a0, a1
            if float(a0) > self.top():
                return QValidator.State.Intermediate, a0, a1
        except:
            pass
        return super().validate(a0, a1)

    def fixup(self, inp: str) -> str:
        if len(inp) > 0 and float(inp) > self.top():
            return str(self.top())
        
        return str(self.bottom())



class GridRemoverParameterWidget(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setLayout(QGridLayout())
        

        self.key_names = [("to_size","Resize to [Å/px]", float, 1,None), ("outside_circle_coverage", "Outside of circle coverage", float,0,1), ("inner_circle_coverage", "Inside of circle coverage", float, 0,1),
                           ("detect_ring", "Detect ring", bool,None, None), ("ring_width", "Ring width [Å]", float, 0.001, None),("wobble", "Wobble", float, 0, 0.2),
                           ("high_pass_filter", "High pass sigma [Å]", int, 0, None), ("threshold", "Threshold", float, None, None), ("gridsizes", "Grid diameter [nm]", list, 0.001,100 )]

        tooltips = {}


        self.configLineEdits = {}
        self.layout().setColumnStretch(0,2)
        self.layout().setColumnStretch(2,1)
        self.layout().setColumnStretch(1,2)
        self.layout().setColumnStretch(3,1)
        for counter, (key, name,t, l,h) in enumerate(self.key_names):
            label = QLabel(name)
            font = label.font()
            
            if t is bool:
                lineedit = QCheckBox()
                lineedit.setChecked(str(ged.CURRENT_CONFIG["parameters"][key]) == "True")
                lineedit.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)

            elif t is float:
                lineedit = QLineEdit(str(ged.CURRENT_CONFIG["parameters"][key]))
                lineedit.setValidator(ged.CorrectDoubleValidator(l, h, ged.CURRENT_CONFIG["parameters"][key]))
                font = label.font()
                lineedit.setFixedWidth(50)
            elif t is list:
                lineedit = QLineEdit(str(ged.CURRENT_CONFIG["parameters"][key][0]))
                lineedit.setValidator(ged.CorrectDoubleValidator(l, h, ged.CURRENT_CONFIG["parameters"][key][0]))
                font = label.font()
                lineedit.setFixedWidth(50)
            elif t is int:
               
                lineedit = QLineEdit(str(ged.CURRENT_CONFIG["parameters"][key]))
                lineedit.setValidator(ged.CorrectIntValidator(l, h))
                lineedit.setFixedWidth(50)
                font = label.font()
                
            self.layout().addWidget(label, counter // 2, (counter % 2)*2)
            self.layout().addWidget(lineedit, counter // 2, (counter % 2)*2 + 1)
            
            self.configLineEdits[key] = (lineedit, t)
            if key in tooltips:
                label.setToolTip(tooltips[key])
                lineedit.setToolTip(tooltips[key])

    
    def getParams(self):
        parameters = {}
        for key, (le, t) in self.configLineEdits.items():
            if t is bool:
                parameters[key] = le.isChecked()
            elif t is list:
                parameters[key] = float(le.text())
            else:
                parameters[key] = t(le.text())
        return parameters

class InstanceIdentificationParameterWidget(QWidget):
    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setLayout(QGridLayout())
        
    def getParams(self):
        params = {

        }

        return params



class TrainWidget(QWidget):
    stopSignal = pyqtSignal(str)
    def __init__(self, parent, listWidget):
        super().__init__(parent)
        self.listWidget:SegmentatorListWidget = listWidget
        self.setLayout(QVBoxLayout())

        self.trainingGroupBox = customGroupBox("Training", self)
        
        self.trainingLayout = QVBoxLayout()
        self.trainingGroupBox.setLayout(self.trainingLayout)
        self.numberOfFilesLabel = QLabel("# of training files:")
        self.listWidget.ListWidget.itemSelectionChanged.connect(self.itemChanged)

        self.filesLayout = QHBoxLayout()

        self.addTrainingSegmentationButton = QPushButton("Add training data")
        self.addTrainingSegmentationButton.font().setBold(True)
        self.addTrainingSegmentationButton.clicked.connect(self.addTrainingSegmentation)
        self.addTrainingSegmentationButton.setToolTip("Add new training images. You will have to first select micrographs and afterwards the segmentations. Both lists of files will be sorted.")

        self.openNapariButton = QPushButton("Open Napari")
        self.openNapariButton.clicked.connect(self.openNapari)
        self.openNapariButton.setToolTip("Opens Napari with a segmentation helper plugin to create new manual segmentations.")

        ########
        self.testLayout = QHBoxLayout()

        self.testPercentageLineedit = QLineEdit("0.05")
        self.testPercentageLineedit.setValidator(CustomDoubleValidator(0,0.95))
        self.testButton = QPushButton("Create test data")
        self.testButton.clicked.connect(self.setTestData)
        self.testButtonIoU = QPushButton("Calculate IoU")
        self.testButtonIoU.clicked.connect(self.testIoU)
        self.resetButton = QPushButton("Reset")
        self.resetButton.clicked.connect(self.reset)

        ########


        self.testLayout.addWidget(self.testPercentageLineedit)
        self.testLayout.addWidget(self.testButton)
        self.testLayout.addWidget(self.testButtonIoU)
        self.testLayout.addWidget(self.resetButton)

        ########
        self.trainButton = QPushButton("TRAIN")
        self.trainButton.font().setBold(True)
        self.trainButton.clicked.connect(self.train)
        self.trainButton.setToolTip("Train the current segmentation model with the added segmentations.")

        self.clearTrainingDataButton = QPushButton("Clear training data")
        self.clearTrainingDataButton.clicked.connect(self.clearTrainingData)
        # self.clearTrainingDataButton.clicked.connect(self.testStuff)
        self.clearTrainingDataButton.setToolTip("Remove all training data from the segmentation model")

        self.activeLearningLayout = QVBoxLayout()

        
        self.activeLearningGroupBox = customGroupBox("Active Learning", self)
        self.numberOfActiveLearningFilesLabel = QLabel("Number of files")

        self.numberOfActiveLearningFilesSpinbox = QSpinBox()
        self.numberOfActiveLearningFilesSpinbox.setMinimum(1)
        self.numberOfActiveLearningFilesLayout = QHBoxLayout()
        self.numberOfActiveLearningFilesLayout.addWidget(self.numberOfActiveLearningFilesLabel)
        self.numberOfActiveLearningFilesLayout.addWidget(self.numberOfActiveLearningFilesSpinbox)

        self.chooseActiveLearningFilesButton = QPushButton("Run active learning")
        self.chooseActiveLearningFilesButton.clicked.connect(self.chooseActiveLearningFiles)
        self.chooseActiveLearningFilesButton.setToolTip("Run active learning. You will chose files and the U-Net will segment the images. It will then ask you to segment the #Number_of_files images it was most unsure of in napari.")

        self.activeLearningLayout.addLayout(self.numberOfActiveLearningFilesLayout)
        self.activeLearningLayout.addWidget(self.chooseActiveLearningFilesButton)
        self.activeLearningGroupBox.setLayout(self.activeLearningLayout)


        self.predictGroupBox = customGroupBox("Prediction", self)

        self.predictGroupBox.setLayout(QVBoxLayout())

        self.predictLayout = QVBoxLayout()
        self.predictButton = QPushButton("Predict images")
        self.predictButton.clicked.connect(self.predictImages)
        self.predictButton.setToolTip("Predict images you chose.")

        self.predictSuffixCombobox= QComboBox()
        # self.predictSuffixCombobox.addItems([".mrc", ".png", ".jpg", ".npz"])
        self.predictSuffixLabel = QLabel("Save as")

        self.predictInstanceCheckbox = QCheckBox("Identify instances")
        self.predictInstanceCheckbox.setToolTip("Whether to try identifying instances. This is useful in very crowded micrographs with overlapping membranes.")
        # self.predictGridRemoverCheckbox = QCheckBox("Mask visible grids")
        # self.predictGridRemoverCheckbox.setToolTip("Whether to try to predict where the metal grid is and remove segmentations on the grid.")
        self.predictInstanceCheckbox.clicked.connect(self.changeAvailableSuffixes)
        self.predictSaveIntermediateFilesCheckbox = QCheckBox("Save intermediate files")

        self.predictImagesLayout = QHBoxLayout()
        self.predictImagesCheckboxLayout = QGridLayout()

        self.predictImagesLayout.addWidget(self.predictButton)
        self.predictImagesLayout.addWidget(self.predictSuffixCombobox)
        self.predictImagesLayout.addWidget(self.predictSuffixLabel)


        self.predictImagesCheckboxLayout.addWidget(self.predictInstanceCheckbox,0,0)
        # self.predictImagesCheckboxLayout.addWidget(self.predictGridRemoverCheckbox,0,1)
        self.predictImagesCheckboxLayout.addWidget(self.predictSaveIntermediateFilesCheckbox,0,1)
        self.predictTabs = QTabWidget()

        self.predictGroupBox.layout().addWidget(self.predictTabs)

        self.predictWidget = QWidget()
        self.predictWidget.setLayout(self.predictLayout)
        self.predictLayout.addLayout(self.predictImagesLayout)
        self.predictLayout.addLayout(self.predictImagesCheckboxLayout)

        # self.gridremoverParameterTab = GridRemoverParameterWidget(self)
        # self.instanceIdentificationParameterTab = InstanceIdentificationParameterWidget(self)

        self.predictTabs.addTab(self.predictWidget, "Predict")
        # self.predictTabs.addTab(self.gridremoverParameterTab, "Grid remover")
        # self.predictTabs.addTab(self.instanceIdentificationParameterTab, "Instance identification")

        self.workers = {}
        self.threads = {}
        


        self.trainingLayout.addWidget(self.numberOfFilesLabel)
        self.filesLayout.addWidget(self.addTrainingSegmentationButton)
        self.filesLayout.addWidget(self.clearTrainingDataButton)
        self.filesLayout.addWidget(self.openNapariButton)
        self.trainingLayout.addLayout(self.filesLayout)
        # self.trainingLayout.addLayout(self.testLayout)#
        self.trainingLayout.addWidget(self.trainButton)
        self.layout().addWidget(self.trainingGroupBox)
        self.layout().addWidget(self.activeLearningGroupBox)
        self.layout().addWidget(self.predictGroupBox)

        self.changeAvailableSuffixes()


    def changeAvailableSuffixes(self):
        current_item = self.predictSuffixCombobox.currentText()
        self.predictSuffixCombobox.clear()
        if self.predictInstanceCheckbox.isChecked():
            suffixes = [".mrc", ".npz"]
        else:
            suffixes = [".mrc", ".png", ".jpg", ".npz"]
        
        self.predictSuffixCombobox.addItems(suffixes)
        if current_item in suffixes:
            self.predictSuffixCombobox.setCurrentIndex(suffixes.index(current_item))

    def testIoU(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            if item.segModel.currently_training:
                item.segModel.print(f"{item.segModel.name}: Is currently training/predicting. Cannot predict.")
                return
            if len(item.segModel.testData["images"]) == 0:
                item.segModel.print(f"{item.segModel.name}: Has no test data set.")
            item.segModel.currently_training = True
            gpu, cores = self.getGPUandCores()
            m :segmentationModel= item.segModel


            self.threads[item.segModel.name] = QThread()

            self.workers[item.segModel.name] = IoUModelWorker(m.load(), m.config, gpu, cores, m.name,m.testData )
            
            self.workers[item.segModel.name].moveToThread(self.threads[item.segModel.name])
            self.threads[item.segModel.name].started.connect(self.workers[item.segModel.name].run)
            self.workers[item.segModel.name].finished.connect(self.threads[item.segModel.name].quit)
            self.workers[item.segModel.name].finished.connect(self.workers[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.threads[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.finishedIoU(m))
            self.workers[item.segModel.name].progress.connect(self.progressEmited(m,[]))
            self.threads[item.segModel.name].start()  





    def setTestData(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            if item.segModel.currently_training:
                item.segModel.print(f"{item.segModel.name}: Is currently training/predicting. Not changing data currently.")
                return
            item.segModel.setTestData(float(self.testPercentageLineedit.text()))


    def reset(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            if item.segModel.currently_training:
                item.segModel.print(f"{item.segModel.name}: Is currently training/predicting. Not changing data currently.")
                return
            item.segModel.build_model(True)


    def predictImages(self):
        from cryovia.gui.datasets_gui import NonMrcFilesPixelSizeWidget

        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            if item.segModel.currently_training:
                item.segModel.print(f"{item.segModel.name}: Is currently training/predicting. Cannot predict.")
                return
            item.segModel.currently_training = True
            dlg = QFileDialog()

            file_suffixes = " *".join([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
            file_suffixes = f"Micrographs (*{file_suffixes})"
            dlg.setFileMode(QFileDialog.ExistingFiles)
            files, filt = dlg.getOpenFileNames(self, "Choose micrographs", ".",filter=file_suffixes)
        
            if len(files) == 0:
                item.segModel.currently_training  = False
                return
            directory = QFileDialog.getExistingDirectory(caption="Choose directory for saving predictions")
            if len(directory) == 0 or directory is None:
                item.segModel.currently_training  = False
                return
            
            gpu, cores = self.getGPUandCores()
            m :segmentationModel= item.segModel
            files = [Path(file) for file in files]
            pixelSizes = {path:None for path in files}
            nonMrcFiles = [path for path in files if path.suffix != ".mrc"]
            if len(nonMrcFiles) > 0:
                pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
                result = pixelSizeWidget.exec()
                if result == 0:
                    return
                ps = pixelSizeWidget.getPixelSize()
                for file in nonMrcFiles:
                    pixelSizes[file] = ps 

            # apply_grid_remover = self.predictGridRemoverCheckbox.isChecked()
            # apply_grid_remover = False
            # predicting_dict = {
            #     "grid_remover":{"apply": apply_grid_remover, "params":self.gridremoverParameterTab.getParams()},
            #     "instance_identification":{"apply":self.predictInstanceCheckbox.isChecked(), "params":self.instanceIdentificationParameterTab.getParams()},
            #     "suffix":self.predictSuffixCombobox.currentText(),
            #     "save_intermediate_files":self.predictSaveIntermediateFilesCheckbox.isChecked()
            #     }
            
            predicting_dict = {
                "grid_remover":{"apply": False, "params":None},
                "instance_identification":{"apply":self.predictInstanceCheckbox.isChecked(), "params":None},
                "suffix":self.predictSuffixCombobox.currentText(),
                "save_intermediate_files":self.predictSaveIntermediateFilesCheckbox.isChecked()
                }
            self.threads[item.segModel.name] = QThread()
            self.workers[item.segModel.name] = PredictModelWorker(files, m.load(), m.config, gpu, cores, m.name,0, Path(directory), predicting_dict, pixelSizes )
            self.stopSignal.connect(self.workers[item.segModel.name].stop)
            self.workers[item.segModel.name].moveToThread(self.threads[item.segModel.name])
            self.threads[item.segModel.name].started.connect(self.workers[item.segModel.name].run)
            self.workers[item.segModel.name].finished.connect(self.threads[item.segModel.name].quit)
            self.workers[item.segModel.name].finished.connect(self.workers[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.threads[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.finishedRunningActiveLearning(m))
            self.workers[item.segModel.name].progress.connect(self.progressEmited(m,[]))
            self.threads[item.segModel.name].start()   


    def chooseActiveLearningFiles(self):
        from cryovia.gui.datasets_gui import NonMrcFilesPixelSizeWidget

        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            if not item.segModel.writable:
                item.segModel.print("Cannot train default model.")
                return
            if item.segModel.currently_training:
                item.segModel.print(f"{item.segModel.name}: Is training/predicting. Cannot predict for active learning.")
                return
            item.segModel.currently_training = True
            dlg = QFileDialog()

            file_suffixes = " *".join([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
            file_suffixes = f"Micrographs (*{file_suffixes})"
            dlg.setFileMode(QFileDialog.ExistingFiles)
            files, filt = dlg.getOpenFileNames(self, "Choose micrographs", ".",filter=file_suffixes)
        
            if len(files) == 0:
                item.segModel.currently_training  = False
                return
            gpu, cores = self.getGPUandCores()
            m :segmentationModel= item.segModel
            files = [Path(file) for file in files]

            pixelSizes = {path:None for path in files}
            nonMrcFiles = [path for path in files if path.suffix != ".mrc"]
            if len(nonMrcFiles) > 0:
                pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
                result = pixelSizeWidget.exec()
                if result == 0:
                    return
                ps = pixelSizeWidget.getPixelSize()
                for file in nonMrcFiles:
                    pixelSizes[file] = ps 

            self.threads[item.segModel.name] = QThread()
            self.workers[item.segModel.name] = PredictModelWorker(files, m.load(), m.config, gpu, cores, m.name, self.numberOfActiveLearningFilesSpinbox.value(), pixelSizes=pixelSizes)
            self.stopSignal.connect(self.workers[item.segModel.name].stop)
            self.workers[item.segModel.name].moveToThread(self.threads[item.segModel.name])
            self.threads[item.segModel.name].started.connect(self.workers[item.segModel.name].run)
            self.workers[item.segModel.name].finished.connect(self.threads[item.segModel.name].quit)
            self.workers[item.segModel.name].finished.connect(self.workers[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.threads[item.segModel.name].deleteLater)
            self.threads[item.segModel.name].finished.connect(self.finishedRunningActiveLearning(m))
            self.workers[item.segModel.name].progress.connect(self.progressEmited(m,[]))
            self.threads[item.segModel.name].start()   
    

    def finishedRunningActiveLearning(self, model):
        def finish():
            model.currently_training=False
            del self.workers[model.name]
            del self.threads[model.name]
            self.itemChanged()
        return finish
        
    
    def finishedIoU(self, model):
        def finish():
            model.currently_training=False
            del self.workers[model.name]
            del self.threads[model.name]
            self.itemChanged()
        return finish

    # def testStuff(self):
    #     item = self.listWidget.ListWidget.currentItem()
    #     if item is not None:
    #         item.segModel.testStuff()

    def itemChanged(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            length = len(item.segModel.trainPaths["images"])
            self.numberOfFilesLabel.setText(f"# of training files: {length}")
            name = item.segModel.name
            if name in self.threads:
                self.trainButton.setText("STOP")
                self.chooseActiveLearningFilesButton.setText("STOP")
                self.predictButton.setText("STOP")
            else:
                self.trainButton.setText("TRAIN")
                self.chooseActiveLearningFilesButton.setText("Run active learning")
                self.predictButton.setText("Predict images")

    def addTrainingSegmentation(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            dlg = QFileDialog()

            file_suffixes = " *".join([".mrc", ".rec", ".MRC", ".REC", ".png", ".jpg", ".jpeg"])
            file_suffixes = f"Micrographs (*{file_suffixes})"
            dlg.setFileMode(QFileDialog.ExistingFiles)
            files, filt = dlg.getOpenFileNames(self, "Choose micrographs", ".",filter=file_suffixes)

        
            if len(files) == 0:
                return
            next_starting_dir = Path(files[0]).parent
            dlg_seg = QFileDialog()
            file_suffixes = " *".join([".mrc", ".MRC", ".png", ".jpg", ".jpeg", ".npz"])
            file_suffixes = f"Micrographs (*{file_suffixes})"
            dlg_seg.setFileMode(QFileDialog.ExistingFiles)
            seg_files, seg_filt = dlg.getOpenFileNames(self, "Choose segmentations", str(next_starting_dir),filter=file_suffixes)

            if len(files) != len(seg_files):
                self.parent().parent().print( f"Number of files ({len(files)}) != number of segmentation files ({len(seg_files)})")
            else:
                files = sorted(files)
                seg_files = sorted(seg_files)
                files = [Path(file) for file in files]
                seg_files = [Path(file) for file in seg_files]
                item.segModel.addTrainPaths(files, seg_files)
                self.itemChanged()
                self.parent().parent().print( f"Added {len(files)} file(s)")
        else:
            self.parent().parent().print("Select a segmentation Model")

    def clearTrainingData(self):
        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            item.segModel.clearTrainPaths()
            self.itemChanged()
            
            self.parent().parent().print(f"Cleared training data from {item.segModel.name}")

    def train(self):
        from cryovia.gui.datasets_gui import NonMrcFilesPixelSizeWidget

        item = self.listWidget.ListWidget.currentItem()
        if item is not None:
            
            m:segmentationModel = item.segModel
            if not m.currently_training:
                
                # callbacks = [customCallback()]
                 #.train(gpu, cores)
                if len(m.trainPaths["images"]) == 0:
                    m.print(f"No training images for {m.name} available.")
                    return
               


                pixelSizes = {path:None for path in m.trainPaths["images"]}
                nonMrcFiles = [path for path in m.trainPaths["images"] if Path(path).suffix != ".mrc"]
                if len(nonMrcFiles) > 0:
                    pixelSizeWidget = NonMrcFilesPixelSizeWidget(self, len(nonMrcFiles))
                    result = pixelSizeWidget.exec()
                    if result == 0:
                        return
                    ps = pixelSizeWidget.getPixelSize()
                    for file in nonMrcFiles:
                        pixelSizes[file] = ps 

                self.trainButton.setText("STOP")
                gpu, cores = self.getGPUandCores()
                m.currently_training = True


                self.threads[item.segModel.name] = QThread()
                self.workers[item.segModel.name] = TrainModelWorker(m.trainPaths, m.load(), segmentationLoss, m.config, gpu, cores, m.createCallbacks(), m.name,pixelSizes)
                
                
                self.stopSignal.connect(self.workers[item.segModel.name].stop)
                self.workers[item.segModel.name].moveToThread(self.threads[item.segModel.name])
                self.threads[item.segModel.name].started.connect(self.workers[item.segModel.name].run)
                self.workers[item.segModel.name].finished.connect(self.threads[item.segModel.name].quit)
                self.workers[item.segModel.name].finished.connect(self.workers[item.segModel.name].deleteLater)
                self.threads[item.segModel.name].finished.connect(self.threads[item.segModel.name].deleteLater)
                self.threads[item.segModel.name].finished.connect(self.finishedRunning(m))
                self.workers[item.segModel.name].progress.connect(self.progressEmited(m, m.createCustomCallbacks()))
                self.threads[item.segModel.name].start()   
                self.itemChanged()
            else:
                self.stopSignal.emit(m.name)

                # m.print(f"{m.name}: Is training/predicting. Cannot train now.")

    def finishedRunning(self, model):
        def finishRunning():
            model.currently_training = False
            model.save()
            self.stopSignal.disconnect(self.workers[model.name].stop)
            del self.workers[model.name]
            del self.threads[model.name]
            self.itemChanged()
        return finishRunning

    def progressEmited(self, model, callbacks):
        def progress(emit):
            t, value = emit
            if t == "print":
                self.parent().parent().print(value)
            elif t == "epoch_logs":
                epoch, logs = value
                for callback in callbacks:
                    callback(epoch, logs)
                self.parent().infoWidget.itemChanged()
            elif t == "active_learning_files":
                self.openNapari(value)
        return progress




    def openNapari(self, files=[]):
        self.parent().parent().print("Opening napari")
        try:
            import napari
            item = self.listWidget.ListWidget.currentItem()
            pixel_size = 7
            if item is None:
                dialog = QDialog()
                dialog.setWindowTitle("No segmentation model selected.")
                dialog.setLayout(QVBoxLayout())
                label = QLabel("You have no segmentation model selected.\nPlease give a pixel size you want to segment in.")
                lineedit = QLineEdit(str(7))
                lineedit.setValidator(DefaultFloatValidator(self.nonMrcPixelSize))
                button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                button_box.accepted.connect(dialog.accept)
                button_box.rejected.connect(dialog.reject)
                layout = dialog.layout()
                layout.addWidget(label)
                layout.addWidget(lineedit)
                layout.addWidget(button_box)
                if dialog.exec_() == QDialog.Accepted:
                    pixel_size = float(lineedit.text())
                else:
                    return

            viewer = napari.Viewer()

            window = viewer.window
            if not isinstance(files, list):
                files = []
            
            if item is not None:
            
                m:segmentationModel = item.segModel
                window.add_dock_widget(SegmentationHelper(viewer, files=files, custom_parent=self, segmentation_model=m,default_pixel_size=pixel_size)) #"cryovia Segmentation Helper"
            else:
                window.add_dock_widget(SegmentationHelper(viewer, files=files, custom_parent=self,default_pixel_size=pixel_size))
            print(window.add_dock_widget)
            viewer.show()

            
            
            
            
            # napari.run(max_loop_level=2)
            # print("Ran napari")
            
        except Exception as e:
            self.parent().parent().print("Error while opening Napari:")
            self.parent().parent().print(traceback.format_exc())
            # self.parent().parent().print(e)
        
    
    # def closedNapari(self, files, seg_model:segmentationModel):
    #     print(files, seg_model)
    #     if seg_model is not None:
    #         seg_model.addTrainPaths(files[0], files[1])

    
    def getGPUandCores(self):
        widget: CpuGpuWindow = self.parent().parent().gpucpuWidget
        gpu = widget.gpuBox.currentText()
        cores = widget.cpuCounter.value()
        return gpu, cores

class DefaultFloatValidator(QDoubleValidator):
    def __init__(self, default):
        super().__init__()
        self.default = default
    def fixup(self, a0: str) -> str:
        if len(a0) == 0:
            return str(self.default)
        if float(a0) <= 0:
            return str(self.default)
        return super().fixup(a0)



class InfoWidget(customGroupBox):
    def __init__(self, parent, listWidget):
        super().__init__("Training loss", parent)
        self.listWidget:SegmentatorListWidget = listWidget
        self.listWidget.ListWidget.itemSelectionChanged.connect(self.itemChanged)
        self.view = pyqtg.PlotWidget(self,"white",labels={"left":("log loss"), "bottom":("epoch")})
        self.view.getPlotItem().setLogMode(False, True)
        self.view.getPlotItem().addLegend()
        # self.view.getPlotItem().getAxis("top").setLabel("log loss")
        # self.view.getPlotItem().getAxis("right").setLabel("epoch")
        self.lossPen = QPen(QColor("blue"))
        self.valLossPen = QPen(QColor("red"))

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setLayout(QHBoxLayout())
        self.layout().addWidget(self.view)


    def itemChanged(self):
        item = self.listWidget.ListWidget.currentItem()
        self.view.clear()
        if item is not None:
            loss = item.segModel.history["loss"]
            valid_loss = item.segModel.history["val_loss"]
            self.view.plot([i for i in range(len(loss))], loss, pen=(0,2), name="loss")
            self.view.plot([i for i in range(len(valid_loss))], valid_loss, pen=(1,2), name="val_loss")
            l = pyqtg.InfiniteLine(item.segModel.bestValLossIndex,name="Best valid loss")
            self.view.addItem(l)    



class TrainAndInfoWidget(QWidget):
    def __init__(self, parent, listWidget):
        super().__init__(parent)
        self.listWidget:SegmentatorListWidget = listWidget

        self.setLayout(QHBoxLayout())

        self.trainWidget = TrainWidget(self, self.listWidget)
        self.infoWidget = InfoWidget(self, self.listWidget)

        self.layout().addWidget(self.trainWidget)
        self.layout().addWidget(self.infoWidget)










class ParameterWindow(customGroupBox):
    COLUMN_SIZE = 3
    def __init__(self, parent, modelWidget):
        super().__init__("Parameters for current segmentation model", parent=parent)
        self.setLayout(QGridLayout())
        self.modelWidget:SegmentatorListWidget = modelWidget

        configs = Config({})
        changableParams = configs.changeableParameterDict

        self.labels = {}
        self.lineedits = {}


        for counter ,(parameter, infos) in enumerate(changableParams.items()):
            t = infos["type"]
            if t is int:
                le = QLineEdit(str(getattr(configs, parameter)))
                le.setValidator(CustomIntValidator(bottom=infos["min"], top=infos["max"]))
            elif t is float:
                le = QLineEdit(str(getattr(configs, parameter)))
                le.setValidator(CustomDoubleValidator(bottom=infos["min"], top=infos["max"]))
            elif t is bool:
                le = QLineEdit(str(getattr(configs, parameter)))
                le.setValidator(QBoolValidator())
            label = QLabel(text=parameter)
            le.editingFinished.connect(self.setParameters)
            self.labels[parameter] = label
            self.lineedits[parameter] = le

            
            self.mEditable:QPalette = self.parent().palette()
            self.mNonEditable:QPalette = QPalette(self.mEditable)
            self.mNonEditable.setColor(QPalette.ColorRole.Base, QColor("grey"))
            self.mNonEditable.setColor(QPalette.ColorRole.Text, QColor("black"))



            self.layout().addWidget(label, int(counter / self.COLUMN_SIZE), (counter % self.COLUMN_SIZE) * 2)
            self.layout().addWidget(le, int(counter / self.COLUMN_SIZE), (counter % self.COLUMN_SIZE) * 2 + 1)


        self.modelWidget.ListWidget.itemSelectionChanged.connect(self.fillParameters)

    def fillParameters(self):
        model =  self.modelWidget.ListWidget.currentItem()
        if model is not None:
            model:segmentationModel = model.segModel
        else:
            return None
        changeable_params = model.config.changeableParameterDict
        for key, le in self.lineedits.items():
            le:QLineEdit 
            le.setText(str(getattr(model.config, key)))
            
            le.setReadOnly(not changeable_params[key]["setable"])
            if changeable_params[key]["setable"] or not model.lockedIn:

                le.setPalette(self.mEditable)
            else:

                le.setPalette(self.mNonEditable)

    
    def setParameters(self):
        self.parent().print("Setting parameters")
        model =  self.modelWidget.ListWidget.currentItem()
        if model is not None:
            model:segmentationModel = model.segModel
        else:
            return None
        changeable_params = model.config.changeableParameterDict
        for key, le in self.lineedits.items():
            le:QLineEdit 
            if changeable_params[key]["setable"] or not model.lockedIn:
                if changeable_params[key]["type"] is bool:
                    value = le.text() == "True"
                else:
                    value = changeable_params[key]["type"](le.text())
                setattr(model.config, key, value)
        model.save()
            
                

class SegmentatorListWidgetItem(QListWidgetItem):
    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent=parent)
        self.segModel:segmentationModel = segmentationModelFactory(**kwargs)
        self.setText(str(self.segModel))
        self.parent = parent
        self.segModel.changed_hooks.append(self.reset_name)

        self.segModel.print = self.parent.parent().parent().parent().print
        
        self.setFlags(self.flags() | Qt.ItemFlag.ItemIsEditable)
        
    def __repr__(self) -> str:
        
        return self.segModel.name

    def reset_name(self):
        self.setText(str(self.segModel))

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
        self.customParent:SegmentatorListWidget = parent
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
            self.parent().selectedItems()[0].segModel.rename(editor.text())
        editor = QLineEdit(parent)
        parent:SegmentatorListWidget
        editor.setValidator(NotAllowedValidator(editor, get_all_segmentation_model_names, None))
        editor.editingFinished.connect(editingFinishedFunc)
        return editor 
    



class SegmentatorListWidget(QWidget, ):
    def __init__(self, parent,) -> None:
        super().__init__(parent)
        
        self.setLayout(QVBoxLayout())
        
        self.ListWidget = QListWidget(self)
        # self.ListWidget.doubleClicked.connect(self.test)
        self.ListWidget.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self.ListWidget.setItemDelegate(CustomItemDelegate(self.ListWidget))
        
        self.segModelGroupBox = customGroupBox("Available models", self)
        self.segModelGroupBox.setLayout(QVBoxLayout())
        # self.segModelLabel = QLabel("Available models")

        self.addButton = QPushButton("+")
        self.addButton.font().setBold(True)
        self.addButton.clicked.connect(self.openModelDialog)
        self.addButton.setFixedSize(23,23)
        self.addButton.setToolTip("Create a new segmentation model.")

        self.removeButton = QPushButton()
        self.removeButton.setIcon(self.removeButton.style().standardIcon(QStyle.SP_DialogDiscardButton))
        self.removeButton.clicked.connect(self.removeModel)
        self.removeButton.setFixedSize(23,23)
        self.removeButton.setToolTip("Remove this segmentation model.")
        

        self.copyButton = QPushButton()
        self.copyButton.setIcon(self.copyButton.style().standardIcon(QStyle.SP_DialogResetButton))
        self.copyButton.clicked.connect(self.copyModel)
        self.copyButton.setFixedSize(23,23)
        self.copyButton.setToolTip("Create a copy of this segmentation model.")

        self.trainButton = QPushButton()
        self.trainButton.setIcon(self.trainButton.style().standardIcon(QStyle.SP_ComputerIcon))
        self.trainButton.clicked.connect(self.trainModel)
        self.trainButton.setFixedSize(23,23)
        self.trainButton.setToolTip("Train this segmentation model")

        self.setParametersButton = QPushButton()
        self.setParametersButton.setIcon(self.setParametersButton.style().standardIcon(QStyle.SP_FileDialogListView))
        self.setParametersButton.clicked.connect(self.parent().setParameters)
        self.setParametersButton.setFixedSize(23,23)
        self.setParametersButton.setToolTip("Set the parameters for this model.")

        # self.showConfusionMatrixButton = QPushButton("#")
        # self.showConfusionMatrixButton.font().setBold(True)
        # self.showConfusionMatrixButton.clicked.connect(self.showConfusionMatrix)

        # self.renameButton = QPushButton()
        # self.renameButton.setIcon(self.renameButton.style().standardIcon(QStlye.))

        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.addWidget(self.removeButton)
        self.buttonLayout.addWidget(self.addButton)
        self.buttonLayout.addWidget(self.copyButton)
        self.buttonLayout.addWidget(self.trainButton)
        self.buttonLayout.addWidget(self.setParametersButton)
        # self.buttonLayout.addWidget(self.showConfusionMatrixButton)

        self.segModelGroupBox.layout().addWidget(self.ListWidget)
        self.segModelGroupBox.layout().addLayout(self.buttonLayout)
        self.layout().addWidget(self.segModelGroupBox)
        # self.layout().addWidget(self.ListWidget)
        # self.layout().addLayout(self.buttonLayout)
        # self.removeButton.font().setBold(True)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.createItems()



    # def showConfusionMatrix(self):
    #     items = self.ListWidget.selectedItems()
    #     if len(items) == 1:
    #         segModel:segmentationModel = items[0].segModel
    #         # self.confusionMatrixWidget = ConfusionTableWidget(self, segModel)
            # self.confusionMatrixWidget.show()

    def createItems(self):
        self.ListWidget.clear()
        # for item in self.ListWidget.items():
        #     self.ListWidget.removeItemWidget(item)
        
        paths = sorted(get_all_segmentation_model_paths())
        for path in paths:
            self.ListWidget.addItem(SegmentatorListWidgetItem(self.ListWidget, filepath=path))

    def openModelDialog(self):
        self.dialog = NewNameDialog(self, get_all_segmentation_model_names, "New model name", "New model", self.createNewModel)
        self.dialog.show()

    def createNewModel(self, name):
        self.ListWidget.addItem(SegmentatorListWidgetItem(self.ListWidget, parameter_dict={"name":name}))
        self.createItems()



    def copyModel(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            
            segModel:segmentationModel = items[0].segModel
            copied_segModel = segModel.create_copy()
            self.ListWidget.addItem(SegmentatorListWidgetItem(self.ListWidget, model=copied_segModel))
            self.createItems()

    def trainModel(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            segModel:segmentationModel = items[0].segModel
            segModel.train()

    def removeModel(self):
        items = self.ListWidget.selectedItems()
        if len(items) == 1:
            segModel:segmentationModel = items[0].segModel
            if segModel.writable:
                self.createWarningMessage(f"Remove {segModel.name} segmentation model? You cannot undo this.", segModel.remove)
            else:
                self.cannotDelete()
                # self.createWarningMessage("You cannot remove the Default segModel.", None)

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
        message = "You cannot remove the Default segmentation model."
        reply = messageBox.question(self, title, message, messageBox.Cancel, messageBox.Cancel)






class CpuGpuWindow(customGroupBox):
    def __init__(self, parent):
        super().__init__("Processing parameters", parent)

        self.gpuBox = QComboBox()
        self.gpuLabel =QLabel("GPU/CPU to use")
        
        self.cpuCounter = QSpinBox()
        self.cpuCounterLabel = QLabel("#Cores")

        

        self.setLayout(QGridLayout())

        self.layout().addWidget(self.cpuCounterLabel, 0,0, Qt.AlignmentFlag.AlignTop)
        self.layout().addWidget(self.cpuCounter, 0,1,Qt.AlignmentFlag.AlignTop)
        self.layout().addWidget(self.gpuLabel, 1,0,Qt.AlignmentFlag.AlignTop)
        self.layout().addWidget(self.gpuBox, 1,1,Qt.AlignmentFlag.AlignTop)

        self.fillComboBox()
        self.configureSpinBox()

    def fillComboBox(self):
        from tensorflow import config
        self.gpuBox.clear()
        gpus = config.list_logical_devices('GPU')
        
        cpus = config.list_logical_devices('CPU')
        for gpu in gpus:
            self.gpuBox.addItem(gpu.name)
        for cpu in cpus:
            self.gpuBox.addItem(cpu.name)

    def configureSpinBox(self):
        cpu_count = multiprocessing.cpu_count()
        self.cpuCounter.setMinimum(1)
        self.cpuCounter.setValue(1)
        self.cpuCounter.setMaximum(cpu_count)


class SegmentationWindow(QWidget):
    def __init__(self, parent=None, custom_parent=None) -> None:
        super().__init__(parent=parent)
        ged.CURRENT_CONFIG = ged.load_config_file()
        self.customParent = custom_parent
        self.setLayout(QGridLayout())
        # self.layout().setContentsMargins(0,0,0,0)
        # self.drawWindow = ShapeDrawingWindow(self)
        self.listWidget = SegmentatorListWidget(self)
        self.trainAndInfoWidget = TrainAndInfoWidget(self, self.listWidget)
        # self.shapesListWidget = ShapesListWidget(self, self.listWidget.ListWidget)
        # self.sideWindow = SideWindow(self, self.drawWindow, self.shapesListWidget)
        self.parameterWidget = ParameterWindow(self, self.listWidget)
        self.gpucpuWidget = CpuGpuWindow(self)

        self.messageBoardBox = customGroupBox("Message board", self)
        self.messageBoardBox.setLayout(QVBoxLayout())
        self.messageBoard = QTextEdit(self)
        self.messageBoard.setReadOnly(True)
        self.messageBoard.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        self.messageBoardBox.layout().addWidget(self.messageBoard)

        # self.layout().addWidget(self.drawWindow,0,1)
        self.layout().setColumnStretch(1,2)
        self.layout().addWidget(self.trainAndInfoWidget, 0,1)
        
        self.layout().addWidget(self.parameterWidget,1,1)
        self.layout().addWidget(self.listWidget,0,0, Qt.AlignmentFlag.AlignBottom)
        self.layout().addWidget(self.gpucpuWidget, 1,0,Qt.AlignmentFlag.AlignTop)
        self.layout().addWidget(self.messageBoardBox, 2,1, Qt.AlignmentFlag.AlignTop)
        # self.layout().addWidget(self.shapesListWidget,1,0) 


    def setParameters(self):
        self.parameterWidget.setParameters()

    def print(self, msg):
        msg = str(msg)
        self.messageBoard.moveCursor(QTextCursor.End)
        cursor = self.messageBoard.textCursor()
        cursor.insertText(msg)
        cursor.insertText("\n")
        self.messageBoard.moveCursor(QTextCursor.End)
        sb = self.messageBoard.horizontalScrollBar()
        sb.setValue(sb.minimum())
        
        # self.resize(self.drawWindow.size() + QSize(100,0))
    
    # def sizeHint(self) -> QSize:
    #     # other_qsize = QSize(self.sideWindow.sizeHint())
    #     # other_qsize.setWidth(0)
        

    #     return self.drawWindow.sizeHint() + self.listWidget.sizeHint() 
        

    def closeEvent(self, a0) -> None:
        if self.customParent is not None:
            self.customParent.child_closed()
        return super().closeEvent(a0)
    

if __name__ == "__main__":
    pass