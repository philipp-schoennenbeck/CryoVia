import os
from pathlib import Path
import json
# import utils
from tqdm import tqdm
import queue
# from .custom_config import custom_config
# from custom_config import custom_config
from cryovia.gui.segmentation_files.losses import segmentationLoss
# from cryovia.gui.segmentation_files.grid_remover import mask_carbon_edge_per_file
from grid_edge_detector.carbon_edge_detector import find_grid_hole_per_file
from grid_edge_detector.image_gui import mask_file

from cryovia.gui.segmentation_files.curvature_skeletonizer import solve_skeleton_per_job

# from tensorflow.keras.callbacks import TerminateOnNaN
from scipy.special import softmax
import gc

import numpy as np
import traceback
import pickle


import multiprocessing as mp

import sparse
from cryovia.cryovia_analysis.custom_utils import resizeSegmentation
from cryovia.gui.segmentation_files.prep_training_data import getTrainingData, loadPredictData, unpatchify, getTrainingDataForPerformance, predictMiddle, get_contour_of_filled_segmentation, load_file
import datetime
import copy
import shutil
from cryovia.gui.Unpickler import CustomUnpickler
import sys
from matplotlib import pyplot as plt
from cryovia.gui.path_variables import SEGMENTATION_MODEL_DIR

# SEGMENTATION_MODEL_DIR = Path().home() / ".cryovia" / "SegmentationModels"




from keras.callbacks import Callback



def logical_process(pipe, device="GPU"):
    import tensorflow as tf
    pipe.send(tf.config.list_logical_devices("GPU"))

def get_logical_devices(device="GPU"):
    con1, con2 = mp.get_context("spawn").Pipe()
    process = mp.get_context("spawn").Process(target=logical_process, args=[con1, device])
    process.start()
    
    result = con2.recv()
    return result



class customCallback(Callback):
    def __init__(self, funcs_on_epoch_end_train=[], funcs_on_epoch_begin_train=[], extra_attr=[]) -> None:
        super().__init__()
        self.funcs_on_epoch_end_train = funcs_on_epoch_end_train
        self.funcs_on_epoch_begin_train = funcs_on_epoch_begin_train
        for extra_a in extra_attr:
            setattr(self, extra_a["name"], extra_a["value"])
        # self.funcs_on_epoch_end_test = funcs_on_epoch_end_test



    def on_epoch_begin(self, epoch, logs=None):
        for func in self.funcs_on_epoch_begin_train:
            func(epoch, logs)
        # keys = list(logs.keys())
        

    def on_epoch_end(self, epoch, logs=None):
        for func in self.funcs_on_epoch_end_train:
            func(epoch, logs)
        # keys = list(logs.keys())



def create_dir(path):
    
    if path.exists():
        return
    path.mkdir(parents=True)



def get_all_segmentation_model_paths():
    global SEGMENTATION_MODEL_DIR
    create_dir(SEGMENTATION_MODEL_DIR)
    segmentation_model_paths = []
    for directory in os.listdir(SEGMENTATION_MODEL_DIR):
        directory = SEGMENTATION_MODEL_DIR / directory

        if directory.is_dir() and (directory / "Segmentator.pickle").exists():
            segmentation_model_paths.append(directory / "Segmentator.pickle")
    
    return segmentation_model_paths


def get_segmentation_dict():
    all_paths = get_all_segmentation_model_paths()
    names = {}
    for path in all_paths:
        name = path.parent.name
        names[name] = path
    return names


def get_all_segmentation_model_names():
    all_paths = get_all_segmentation_model_paths()
    names = [path.parent.name for path in all_paths]

    return set(names)


def custom_meaniou(classes):
    import tensorflow as tf
    loss = tf.keras.metrics.MeanIoU(classes)
    def custom_loss(y_true, y_pred):
        y_pred = tf.keras.activations.softmax(y_pred, axis=-1)
        y_pred = tf.argmax(y_pred, -1)
        y_true = tf.argmax(y_true, -1)
        return loss(y_true, y_pred)
    return custom_loss




def conv_block2(n_filter, n1,
                activation="relu",
                border_mode="same",
                dropout=0.0,
                batch_norm=False,
                init="glorot_uniform",
                **kwargs):
    from tensorflow.keras.layers import Conv2D, Activation, Dropout, BatchNormalization

    def _func(lay):
        if batch_norm:
            s = Conv2D(n_filter, n1, padding=border_mode, kernel_initializer=init, **kwargs)(lay)
            s = BatchNormalization()(s)
            s = Activation(activation)(s)
        else:
            s = Conv2D(n_filter, n1, padding=border_mode, kernel_initializer=init, activation=activation, **kwargs)(lay)
        if dropout is not None and dropout > 0:
            s = Dropout(dropout)(s)
        return s

    return _func



def unet_block(n_depth=2, n_filter_base=16, kernel_size=(3,3), n_conv_per_depth=2,
               activation="relu",
               batch_norm=False,
               dropout=0.0,
               last_activation=None,
               pool=(2,2),
               kernel_init="glorot_uniform",
               prefix=''):

    from keras.layers import Concatenate,  MaxPooling2D, UpSampling2D
    # if len(pool) != len(kernel_size):
    #     raise ValueError('kernel and pool sizes must match.')
    n_dim = 2
    if n_dim not in (2,3):
        raise ValueError('unet_block only 2d or 3d.')

    conv_block = conv_block2  #if n_dim == 2 else conv_block3
    pooling    = MaxPooling2D #if n_dim == 2 else MaxPooling3D
    upsampling = UpSampling2D #if n_dim == 2 else UpSampling3D

    if last_activation is None:
        last_activation = activation

    channel_axis = -1 #if backend_channels_last() else 1

    def _name(s):
        return prefix+s

    def _func(input):
        skip_layers = []
        layer = input

        # down ...
        for n in range(n_depth):
            for i in range(n_conv_per_depth):
                layer = conv_block(n_filter_base * 2 ** n, kernel_size,
                                   dropout=dropout,
                                   activation=activation,
                                   init=kernel_init,
                                   batch_norm=batch_norm, name=_name("down_level_%s_no_%s" % (n, i)))(layer)
            skip_layers.append(layer)
            layer = pooling(pool, name=_name("max_%s" % n))(layer)

        # middle
        for i in range(n_conv_per_depth - 1):
            layer = conv_block(n_filter_base * 2 ** n_depth, kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation,
                               batch_norm=batch_norm, name=_name("middle_%s" % i))(layer)

        layer = conv_block(n_filter_base * 2 ** max(0, n_depth - 1), kernel_size,
                           dropout=dropout,
                           activation=activation,
                           init=kernel_init,
                           batch_norm=batch_norm, name=_name("middle_%s" % n_conv_per_depth))(layer)

        # ...and up with skip layers
        for n in reversed(range(n_depth)):
            layer = Concatenate(axis=channel_axis)([upsampling(pool)(layer), skip_layers[n]])
            for i in range(n_conv_per_depth - 1):
                layer = conv_block(n_filter_base * 2 ** n, kernel_size,
                                   dropout=dropout,
                                   init=kernel_init,
                                   activation=activation,
                                   batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, i)))(layer)

            layer = conv_block(n_filter_base * 2 ** max(0, n - 1), kernel_size,
                               dropout=dropout,
                               init=kernel_init,
                               activation=activation if n > 0 else last_activation,
                               batch_norm=batch_norm, name=_name("up_level_%s_no_%s" % (n, n_conv_per_depth)))(layer)

        return layer

    return _func





def custom_unet(input_shape,
                last_activation,
                n_depth=2,
                n_filter_base=16,
                kernel_size=3,
                n_conv_per_depth=2,
                activation="relu",
                batch_norm=False,
                dropout=0.2,
                pool_size=2,
                n_channel_out=1,
                residual=False,
                prob_out=False,
                eps_scale=1e-3,
                 half_size_output=False):
    """ TODO """
    
    from keras.layers import Conv2D, Input, Activation, Lambda, Concatenate
    from keras import Model

    if last_activation is None:
        raise ValueError("last activation has to be given (e.g. 'sigmoid', 'relu')!")

    # all((s % 2 == 1 for s in kernel_size)) or _raise(ValueError('kernel size should be odd in all dimensions.'))

    channel_axis = -1 #if backend_channels_last() else 1

    n_dim = 2
    conv = Conv2D #if n_dim==2 else Conv3D

    input = Input((input_shape, input_shape, 1), name = "input")
    unet = unet_block(n_depth, n_filter_base, kernel_size,
                      activation=activation, dropout=dropout, batch_norm=batch_norm,
                      n_conv_per_depth=n_conv_per_depth, pool=pool_size)(input)
    # if half_size_output:

    #     # final = conv(n_channel_out, (2,)*n_dim,strides=2,padding="valid", activation='linear')(unet)
    #     final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    #     # final = Crop(input_shape[0] // 4)(final)
        
    # else:
    final = conv(n_channel_out, (1,)*n_dim, activation='linear')(unet)
    
    # if residual:
    #     if not (n_channel_out == input_shape[-1] if backend_channels_last() else n_channel_out == input_shape[0]):
    #         raise ValueError("number of input and output channels must be the same for a residual net.")
    #     final = Add()([final, input])
    final = Activation(activation=last_activation)(final)

    if prob_out:
        scale = conv(n_channel_out, (1,)*n_dim, activation='softplus')(unet)
        scale = Lambda(lambda x: x+np.float32(eps_scale))(scale)
        final = Concatenate(axis=channel_axis)([final,scale])

    return Model(inputs=input, outputs=final)




class Config:
    def __init__(self, parameter_dict) -> None:
        global SEGMENTATION_MODEL_DIR
        self.n_dim = 2
        self.nr_classes = 2
        self.max_batch_size = 64
        self.input_shape = 256
        self.thin_segmentation = True
        self.dilation = 1
        self.last_activation = "linear"
        self.n_depth = 4
        self.n_filter_base = 32
        self.kernel_size = 3
        self.n_conv_per_depth = 2
        self.activation = "relu"
        self.batch_norm = True
        self.dropout = 0.2
        self.pool_size = 2
        self.pixel_size = 7
        self.high_pass_filter = 0
        self.std_clip = 4
        self.n_channel_out = 2
        self.eps_scale = 0.001 
        self.train_learning_rate = 0.0004
        self.train_reduce_lr = True
        self.relative_weights = [1.0, 5.0]
        self.changed_ = False
        self.name = None
        self.determine_batch_size = True
        self.train_epochs = 5
        self.filled_segmentation = False
        self.only_use_improved = True

        self.changeableParameterDict = {
            # "nr_classes": {"type":int, "min":2, "max":100, "setable":False},
            "max_batch_size": {"type":int, "min":1, "max":1024, "setable":True},
            "input_shape": {"type":int, "min":64, "max":1024, "setable":False},
            "n_depth": {"type":int, "min": 2, "max":6, "setable":False},
            "n_filter_base": {"type":int, "min":4, "max":256, "setable":False},
            "kernel_size": {"type":int, "min":3 ,"max":15, "setable":False},
            "n_conv_per_depth": {"type":int, "min":1 ,"max":16, "setable":False},
            "dropout": {"type":float, "min": 0,"max":0.9, "setable":False},
            "train_learning_rate": {"type":float, "min": 0.000001,"max":0.1, "setable":True},
            "train_reduce_lr":{"type":bool, "setable":True},
            "thin_segmentation":{"type":bool, "setable":True},
            "dilation":{"type":int, "min":0, "max":99, "setable":True},
            # "determine_batch_size":{"type":bool, "setable":True},
            "train_epochs":{"type":int, "min":1, "max":1000, "setable":True},
            "filled_segmentation":{"type":bool, "setable":True},
            "only_use_improved":{"type":bool, "setable":True},
            "pixel_size":{"type":float, "min":0.5, "max":9999, "setable":True},
            "high_pass_filter":{"type":float, "min":0, "max":99999, "setable":True},
            "std_clip":{"type":float, "min":0, "max":99999, "setable":True}
        }


        for k in parameter_dict.keys():
            setattr(self, k, parameter_dict[k])

        # if self.save_dir is None and self.name is None:
        #     raise ValueError("save_dir or name should not be None")
        if self.name is not None:
            
            if not self.save_dir.exists():
                self.save_dir.mkdir(parents=True)



    

    @property
    def save_dir(self):
        global SEGMENTATION_MODEL_DIR
        if self.name is not None:
            
            (SEGMENTATION_MODEL_DIR / self.name).mkdir(parents=True, exist_ok=True)
            return SEGMENTATION_MODEL_DIR / self.name
        return None

    def __setattr__(self, __name: str, __value) -> None:
        if __name != "name" and __name != "changed_":
            self.changed_ = True
        super().__setattr__(__name, __value)


def segmentationModelFactory(filepath=None, parameter_dict=None, to_copy =None, model=None):
    if filepath is None and parameter_dict is None and to_copy is None and model is None:
        raise ValueError("SegmentationModelFactory: One of the parameters has to be not None.")
    if filepath is not None:
        with open(filepath, "rb") as f:
            inst = CustomUnpickler(f).load()
            if not hasattr(inst, "testData"):
                setattr(inst, "testData", {"images":[], "segmentations":[]})
            if not hasattr(inst, "pixel_sizes"):
                pixel_sizes = {}
                for segmentation in inst.trainPaths["segmentations"]:
                    pixel_sizes[segmentation] = inst.config.pixel_size
                for segmentation in inst.testData["segmentations"]:
                    pixel_sizes[segmentation] = inst.config.pixel_size
                inst.pixel_sizes = pixel_sizes
                
            inst.print = print
    elif parameter_dict is not None:
        inst = segmentationModel(parameter_dict)
    elif to_copy is not None:
        inst = to_copy.create_copy()
    elif model is not None:
        inst = model
    
    default_config = Config({})
    inst:segmentationModel
    for key, value in vars(default_config).items():
        if not hasattr(inst.config, key):
            setattr(inst.config, key, value)
    
    for key, value in default_config.changeableParameterDict.items():
        if key not in inst.config.changeableParameterDict:
            inst.config.changeableParameterDict[key] = value
    inst.currently_training = False
    return inst



class segmentationModel:
    def __init__(self, parameter_dict:dict):
        self.config = Config(parameter_dict)
        self.history = {"loss":[], "val_loss":[]}
        self.epochs = 0
        self.bestValLossIndex = 0
        self.currently_training = False
        self.creationTime_ = datetime.datetime.now().timestamp()
        self.changedTime_ = copy.copy(self.creationTime_)
        self.changed_hooks = []
        self.trainPaths = {"images":[], "segmentations":[]}
        self.testData = {"images":[], "segmentations":[]}
        self.pixel_sizes = {}
        # self.activeLearningFiles = []
        self.print_ = print
        self.save()

        # self.build_model()

    def rename(self, new_name):
        global SEGMENTATION_MODEL_DIR
        all_names = get_all_segmentation_model_names()
        if new_name in all_names:
            raise FileExistsError(new_name)
        shutil.move(self.config.save_dir, SEGMENTATION_MODEL_DIR / new_name)
        self.name = new_name
        self.save()


    def setTestData(self, perc=0.05):
        if len(self.testData["images"]) > 0:
            self.trainPaths["images"].extend(self.testData["images"])
            self.trainPaths["segmentations"].extend(self.testData["segmentations"])
            self.testData = {"images":[], "segmentations":[]}

        idxs = np.arange(len(self.trainPaths["images"]))
        np.random.shuffle(idxs)
        
        max_idx = int(len(idxs) * perc)
        if max_idx <= 0:
            self.print("Too few images to set aside test data.")
            return
        test_idxs = idxs[:max_idx]
        test_idxs = sorted(test_idxs)[::-1]
        for idx in test_idxs:
            self.testData["images"].append(self.trainPaths["images"].pop(idx))
            self.testData["segmentations"].append(self.trainPaths["segmentations"].pop(idx))
        self.print(f"Declared {len(test_idxs)} files as test data.")
        self.save()
        

    def addOnePath(self, img, seg, pixelSize=1):
        data, ps = load_file(img)
        seg_data, seg_ps = load_file(seg)
        if seg_ps is not None:
            self.pixel_sizes[seg] = seg_ps
        elif ps is not None:
            seg_ps = ps * data.shape[-1] / seg_data.shape[-1]
            self.pixel_sizes[seg] = seg_ps
        else:
            self.pixel_sizes[seg] = pixelSize

    def addTrainPaths(self, image_paths, segmentation_paths, pixelSize=1):
        assert len(image_paths) == len(segmentation_paths)
        self.updateTrainPaths(image_paths, segmentation_paths, pixelSize)
        


    def updateTrainPaths(self, image_paths, segmentation_paths, pixelSize=1):
        assert len(image_paths) == len(segmentation_paths)
        for im, seg in zip(image_paths, segmentation_paths):
            if im in self.trainPaths["images"]:
                idx = self.trainPaths["images"].index(im)
                self.trainPaths["segmentations"][idx] = seg

            elif Path(im) in self.trainPaths["images"]:
                idx = self.trainPaths["images"].index(Path(im))
                self.trainPaths["segmentations"][idx] = seg
            else:
                self.trainPaths["images"].append(im)
                self.trainPaths["segmentations"].append(seg)
            self.addOnePath(im, seg, pixelSize)
        self.save()
    
    def clearTrainPaths(self):
        self.trainPaths = {"images":[], "segmentations":[]}
        self.testData = {"images":[], "segmentations":[]}
        self.pixel_sizes = {}
        self.save()

    @property
    def lockedIn(self):
        return (self.config.save_dir / "best_weights.h5").exists()

    def createCustomCallbacks(self):
        starting_epoch = self.epochs
        def addToHistory(epoch, logs):
            self.history["loss"].append(logs["loss"])
            self.history["val_loss"].append(logs["val_loss"])
            


        def addEpoch(epoch, logs):
            self.epochs += 1
        
        def checkBestValLoss(epoch, logs):
            last_val_losses = self.history["val_loss"][-(epoch + 1):]
            self.bestValLossIndex = starting_epoch + np.argmin(last_val_losses)
           
        # callbacks = [customCallback([addToHistory, addEpoch, checkBestValLoss])]
        return addToHistory, addEpoch, checkBestValLoss
        # return callbacks

    def create_copy(self):
        def find_new_name():
            names = get_all_segmentation_model_names()
            counter = 1
            while True:
                new_name = f"{self.name}_{counter}"
                if new_name not in names:
                    return new_name
                counter += 1 
        hooks = self.changed_hooks
        print_func = self.print
        self.print = None
        self.changed_hooks = []
        new_copy = copy.deepcopy(self)
        self.changed_hooks = hooks
        self.print = print_func
        new_copy.name = find_new_name()
        
        self.print(new_copy.name)
        new_copy.save()
        if (self.config.save_dir / "best_weights.h5").exists():
            shutil.copy((self.config.save_dir / "best_weights.h5"), new_copy.config.save_dir / "best_weights.h5")
            
        return new_copy
        
    def build_model(self, overwrite=False):
        model = custom_unet(
            input_shape=self.config.input_shape,
            last_activation=self.config.last_activation,
            n_depth=self.config.n_depth,
            n_filter_base=self.config.n_filter_base,
            kernel_size=self.config.kernel_size,
            n_conv_per_depth=self.config.n_conv_per_depth,
            activation=self.config.activation,
            batch_norm=self.config.batch_norm,
            dropout=self.config.dropout,
            pool_size=self.config.pool_size,
            n_channel_out=self.config.n_channel_out,
            eps_scale=self.config.eps_scale,
        )
        if self.config.save_dir is not None:
            if not (Path(self.config.save_dir) / "best_weights.h5").exists() or overwrite:
                model.save_weights((Path(self.config.save_dir) / "best_weights.h5"))
        else:
            self.print("Savedir in config is None")
        return model


    def createCallbacks(self):
        from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping

        callbacks = []    
        if self.config.train_reduce_lr:
            callbacks.append(ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10))
        callbacks.append(EarlyStopping(patience=10))
        callbacks.append(
                ModelCheckpoint(str(Path(self.config.save_dir) / "best_weights.h5"), save_best_only=True,
                                save_weights_only=True, monitor="val_loss", mode="min"))

        # callbacks.extend(self.createCustomCallbacks())
        return callbacks

    


    def normalize(self, x):
        means = [np.mean(img) for img in x]
        stds = [np.std(img) for img in x]
        
        x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
        return x
 


    def load(self, loaddir=None):

        model = self.build_model()
        if loaddir is None:
            if self.config.save_dir is not None and (Path(self.config.save_dir) / "best_weights.h5").exists():
                loaddir = (Path(self.config.save_dir) / "best_weights.h5")
            else:
                return None
        if loaddir.exists():
                       
            # self.model.load_weights(str(Path(self.config.save_dir) / "best_weights.h5"))
            model.load_weights(loaddir)
        return model   
        

    def save(self, overrule=False):
        if not self.writable and not overrule:
            return
        if self.config.save_dir is not None:
            path = self.config.save_dir / "Segmentator.pickle"
            hooks = self.changed_hooks
            print_func = self.print
            self.print = None
            self.changed_hooks = []
            with open(path, "wb") as f:
                pickle.dump(self, f)
            self.changed_hooks = hooks
            self.print = print_func
        else:
            self.print("Could not save. Savedir is None")



    def remove(self):
        if not self.writable:
            self.print(f"Cannot remove default model.")
            return
        if not self.currently_training:
            self.print(f"Removing directory {self.config.save_dir}")
            shutil.rmtree(self.config.save_dir)
            # os.removedirs(self.config.save_dir)
        else:
            self.print(f"Cannot remove {self.name}. It is currently training.")
        

    @property
    def name(self):
        if self.config.save_dir is not None:
            return self.config.name
        return None
    
    @name.setter
    def name(self, value):
        self.config.name = value


    @property
    def writable(self):
        return self.name != "Default" and self.name != "Default_thin" 

    def __str__(self) -> str:
        return self.name

    

    @property
    def print(self):
        if self.print_ is None:
            return print
        else:
            return self.print_

    @print.setter
    def print(self, value):
        self.print_ = value


    
    def predict_multiprocessing(self, file_paths, pixelSizes, gpu=None,  kwargs={}, njobs=2, threads=10,tqdm_file=sys.stdout,dataset_name="", stopEvent=None, seg_path=None, mask_path=None ):
        if stopEvent is None:
            stopEvent = mp.get_context("spawn").Event()
        
        if isinstance(gpu, str):
            gpu = [gpu]

        to_mask = False
        if kwargs["run"]["maskGrid"]:
            if "remove_segmentation_on_edge" in kwargs["maskGrid"] and kwargs["maskGrid"]["remove_segmentation_on_edge"]:
                to_mask = True

        # import tensorflow as tf
        filePathsQueue = mp.get_context("spawn").Queue()
        loadInQueue = mp.get_context("spawn").Queue(threads)
        predictionQueue = mp.get_context("spawn").Queue(threads)
        outputQueue = mp.get_context("spawn").Queue()
        njobs = threads * njobs

        number_of_loader_proccesses = max(1, int((njobs - len(gpu)) * 1/3))
        number_of_unpatchifyer_procceses = max(1, njobs - len(gpu) - number_of_loader_proccesses)
        [filePathsQueue.put((path, ps)) for (path, ps) in zip(file_paths, pixelSizes)]
        loaderFinishedEvent = mp.get_context("spawn").Event()
        predictorFinishedEvent = mp.get_context("spawn").Event()
        
        loaders = [mp.get_context("spawn").Process(target=loadPathsProcess, args=(filePathsQueue, loadInQueue, self.config)) for _ in range(number_of_loader_proccesses)]


        predictors = [mp.get_context("spawn").Process(target=predictProcess, args=(self,gpu, gpu_idx,  loadInQueue, predictionQueue,loaderFinishedEvent )) for gpu_idx in range(len(gpu))]

        kwargs["segmentation"]["filled_segmentation"] = self.config.filled_segmentation

        unpatchifyers = [mp.get_context("spawn").Process(target=unpatchifyProcess, args=(kwargs, self.config, predictionQueue, outputQueue, predictorFinishedEvent, seg_path, mask_path)) for _ in range(number_of_unpatchifyer_procceses)]

        if stopEvent.is_set():
            return {}
        [loader.start() for loader in loaders]
        [predictor.start() for predictor in predictors]
        [unpatchifyer.start() for unpatchifyer in unpatchifyers]

        results = {}
        with tqdm(total=len(file_paths), desc=f"{dataset_name}: Segmentation", smoothing=0, file=tqdm_file) as pbar:
            while True:
                if stopEvent.is_set():
                    loaderFinishedEvent.set()
                    predictorFinishedEvent.set()
                    for q in [filePathsQueue, loadInQueue, predictionQueue, outputQueue]:
                        while True:
                            try:
                                q.get_nowait()
                            except:
                                break
                    while any([loader.is_alive() for loader in loaders]):
                        time.sleep(0.1)
                    while any([predictor.is_alive() for predictor in predictors]):
                        time.sleep(0.1)
                    while any([unpatchifyer.is_alive() for unpatchifyer in unpatchifyers]):
                        time.sleep(0.1)
                    
                    return {}
                if not any([loader.is_alive() for loader in loaders]):
                    if not loaderFinishedEvent.is_set():
                        loaderFinishedEvent.set()
                if not any([predictor.is_alive() for predictor in predictors]):
                    if not predictorFinishedEvent.is_set():
                        predictorFinishedEvent.set()

                try:
                    path, prediction = outputQueue.get(False)
                    results[path] = prediction
                    try:
                        pbar.update(1)
                    except:
                        pass
                
                except queue.Empty:
                    time.sleep(0.1)
                    pass
                
                if not any([unpatchifyer.is_alive() for unpatchifyer in unpatchifyers]) and outputQueue.empty():
                    
                    break
           
        gc.collect()
        # tf.keras.backend.clear_session()
        return results




def normalize(x):
    means = [np.mean(img) for img in x]
    stds = [np.std(img) for img in x]
    
    x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
    return x


def loadPathsProcess(inputqueue, outputqueue, config):
    
    # import tensorflow as tf
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # # tf.config.set_visible_devices([], 'GPU')
    # with tf.device('/CPU:0'):
    while True:
        try:
            path,ps = inputqueue.get(timeout=5)
        except queue.Empty:
            if inputqueue.empty():
                return
            continue 
        
        turned_patches_total = []

        x_pred, shape = loadPredictData([path], config, ps)


        x_pred = normalize(x_pred)
        x_pred = x_pred[0]
        shape = shape[0]

        for patch in x_pred:
            turned_patches = [np.rot90(patch, i,(0,1)) for i in range(4)]
            turned_patches_total.extend(turned_patches)
        turned_patches = np.array(turned_patches_total)
        # turned_patches = tf.constant(turned_patches)
       
        outputqueue.put((path, turned_patches, shape))
    

import time

def predictProcess(segmentationModel, gpus, gpu_idx,  inputqueue, outputqueue, event ):
    try:

        os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu_idx)   
        gpu = gpus[gpu_idx] 
        import tensorflow as tf

        def test_batch_size(batch_size, model):
            x_batch = np.zeros((batch_size, segmentationModel.config.input_shape, segmentationModel.config.input_shape,1), dtype=np.float32)
            # y_batch = np.zeros((batch_size, self.config.input_shape, self.config.input_shape,2), dtype=np.float32)
            try:
                result = model.predict(x=x_batch, batch_size=batch_size, verbose=0)
                return True
            except Exception as e:
                return False

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for g in gpus:
                    tf.config.experimental.set_memory_growth(g, True)
                logical_gpus = tf.config.list_logical_devices('GPU')

            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)



        if gpu is None:
            gpus = tf.config.list_logical_devices('GPU')
            if len(gpus) > 0:
                gpu = gpus[0]
            else:
                cpus = tf.config.list_logical_devices('CPU')
                gpu = cpus[0]

        

        with tf.device(gpu) as device:


            model = segmentationModel.load()
            current_batch_size = segmentationModel.config.max_batch_size
            working_batch_size = None
            tried_batch_sizes = set()

            while current_batch_size not in tried_batch_sizes:
                
                tried_batch_sizes.add(current_batch_size)
                if current_batch_size == 0:
                    tf.keras.backend.clear_session()
                    return 
                if test_batch_size(current_batch_size, model):
                    working_batch_size = current_batch_size
                    current_batch_size *= 2
                else:
                    current_batch_size = current_batch_size // 2
                gc.collect()
                tf.keras.backend.clear_session()
                if current_batch_size > segmentationModel.config.max_batch_size:
                    current_batch_size = segmentationModel.config.max_batch_size
                    break

            tf.keras.backend.clear_session()
            while True:
                prediction = []
                try:
                    path, turned_patches, shape = inputqueue.get(timeout=5)
                except queue.Empty:
                    if event.is_set():
                        tf.keras.backend.clear_session()
                        return
                    continue
                turned_patches = tf.convert_to_tensor(turned_patches)

                for i in range(0, len(turned_patches), working_batch_size):
                    prediction.append(model.predict(turned_patches[i:i+working_batch_size], batch_size=working_batch_size, verbose=0))
                    gc.collect()

                prediction = np.concatenate(prediction)
                outputqueue.put((path, prediction, shape))
    except Exception as e:
        tf.keras.backend.clear_session()
        print(traceback.format_exc())

        raise e


def unpatchifyProcess(kwargs, config, inputqueue, outputqueue, event, dataset_path, mask_path):
    while True:
        try:
            path, prediction, shape = inputqueue.get(timeout=5)
        except queue.Empty:
            if event.is_set():
                return
            continue
            

        predictions = softmax(prediction, -1)
        predicted_patches = []
        for i in range(len(prediction)//4):
            turned_pred = [np.rot90(pred, i%4) for pred,i in zip(predictions[i*4:i*4+4], range(4,0,-1))]

            prediction = np.sum(turned_pred, 0)

            predicted_patches.append(prediction)
        predicted_patches = np.array(predicted_patches)
        predicted_image, confidence = unpatchify(predicted_patches, shape, config, threshold=True, both=True)
        mask = None
        if kwargs["run"]["maskGrid"]:
            
            if "use_existing_mask" in kwargs["maskGrid"] and kwargs["maskGrid"]["use_existing_mask"]:
                current_mask_path = mask_path / (Path(path).stem + "_mask.pickle")
                if current_mask_path.exists():
                    if current_mask_path.suffix == ".pickle":
                        mask = mask_file.load(current_mask_path).create_mask()
                    else:
                        mask,_ = load_file(current_mask_path)
            else:
                mask = find_grid_hole_per_file(path, **kwargs["maskGrid"])
            if mask is not None:
                mask = resizeSegmentation(mask,predicted_image.shape).todense()

        if kwargs["segmentation"]["filled_segmentation"]:
            predicted_image = get_contour_of_filled_segmentation(predicted_image)
        else:
            if kwargs["segmentation"]["identify_instances"]:
                try:
                    max_nodes = 30
                    if "max_nodes" in kwargs["segmentation"]:
                        max_nodes = kwargs["segmentation"]["max_nodes"]
                    predicted_image, skeletons = solve_skeleton_per_job(predicted_image,mask, kwargs["general"]["use_only_closed"], name=path, max_membrane_thickness=10, connect_parts=kwargs["segmentation"]["combine_snippets"], max_nodes=max_nodes)
                except Exception as e:
                    raise e

        predicted_image = sparse.as_coo(predicted_image)

        seg_path = dataset_path / (Path(path).stem + "_labels.npz")
        sparse.save_npz(seg_path, predicted_image)

        outputqueue.put((path, seg_path))


if __name__ == "__main__":
    

    pass