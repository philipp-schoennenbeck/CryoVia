from pathlib import Path
import os
from cryovia.cryovia_analysis.analyser import Analyser , AnalyserWrapper
import pickle
from cryovia.cryovia_analysis.shape_classifier import get_classifier_dict, ShapeClassifierFactory, ShapeClassifier
from cryovia.gui.segmentation_files.segmentation_model import get_segmentation_dict, segmentationModelFactory, segmentationModel, Config
# from cryovia.gui.datasets_gui import DEFAULT_CONFIGS
import numpy as np
import difflib
import sparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import multiprocessing as mp
from tqdm import tqdm
from cryovia.gui.Unpickler import CustomUnpickler
import sys
import gc
import traceback
import pandas as pd
import shutil
from copy import deepcopy
from datetime import datetime
import time
from collections import OrderedDict
import tarfile

from cryovia.cryovia_analysis.shape_classifier import get_all_classifier_names
from cryovia.gui.membrane_segmentation_gui import get_all_segmentation_model_names
from grid_edge_detector.carbon_edge_detector import find_grid_hole_per_file
from cryovia.gui.segmentation_files.prep_training_data import load_file
from cryovia.cryovia_analysis.custom_utils import resizeSegmentation
from grid_edge_detector.image_gui import mask_file
from cryovia.gui.path_variables import DATASET_PATH


if os.name == 'nt':
    MP_START_METHOD = "spawn"
else:
    MP_START_METHOD = "spawn"



DEFAULT_CONFIGS = OrderedDict([
    ("general",OrderedDict([
        ("use_only_closed",True),
        ("rerun",False),
        ("step_size",13),
        ("min_size",50),
        ("only_run_for_new_data",False),
        ("estimate_middle_plane",True),
        ("dark_mode", True)
   ])),
    ("segmentation",OrderedDict([
        ("rerun_segmentation", False),
        ("identify_instances",True),
        ("segmentation_model",get_all_segmentation_model_names),
        ("only_segmentation", False),
        ("combine_snippets", False),
        ("max_nodes", 30)
    ])),
    ("maskGrid", OrderedDict([
        ("use_existing_mask", False),
        ("to_size",100),
        ("diameter",12000), 
        ("threshold",0.005), 
        ("coverage_percentage",0.5), 
        ("outside_coverage_percentage",0.05),
        ("detect_ring",True), 
        ("ring_width",200), 
        ("wobble",0), 
        ("high_pass",0),
        ("distance", 0),
        ("return_ring_width", 400),
        ("crop", 500),

    ])),
    ("estimateThickness",OrderedDict([
        ("max_neighbour_dist",300.0),
        ("min_thickness",20.0),
        ("max_thickness",70.0),
        ("sigma",2),
        ("smooth_contour", False)
    ])),

    ("estimateCurvature",OrderedDict([
         ("max_neighbour_dist",300.0),
         ("adaptive", False),
         ("max_distance", 1500),
         ("min_distance", 35),
         ("threshold", 2),
         ("step",100)

])),

    ("shapePrediction",OrderedDict([
       ( "shape_classifier",get_all_classifier_names)
    ])),
    ("identifyIceContamination",OrderedDict([
       
])),
    ("enclosed",OrderedDict([
      
    ])),
    ("identifyIceContamination",OrderedDict([
       
])),

])



def logical_process(pipe, device="GPU"):
    import tensorflow as tf
    from cryovia.gui.starting_menu import changeToDebug
    if os.environ["CRYOVIA_MODE"] is not None and int(os.environ["CRYOVIA_MODE"]) == 1:
        changeToDebug()
    pipe.send(tf.config.list_logical_devices(device))

def get_logical_devices(device="GPU"):
    global MP_START_METHOD
    con1, con2 = mp.get_context(MP_START_METHOD).Pipe()
    process = mp.get_context(MP_START_METHOD).Process(target=logical_process, args=[con1, device])
    process.start()
    result = con2.recv()
    return result


def create_dir(path):
    
    if path.exists():
        return
    path.mkdir(parents=True)



def get_all_dataset_paths():
    """
    Extracts all the paths of the available datasets.
    Parameters
    ----------


    Returns
    -------
    dataset_paths : list of all paths
    """
    global DATASET_PATH
    create_dir(DATASET_PATH)
    dataset_paths = []
    for directory in os.listdir(DATASET_PATH):
        directory = DATASET_PATH / directory
        if directory.is_dir() and (directory / "dataset.pickle").exists():
            dataset_paths.append(directory)
    
    return dataset_paths


def get_all_dataset_names():
    """
    Extracts the names of all the available datasets.
    Parameters
    ----------


    Returns
    -------
    names : set of dataset names
    """
    all_paths = get_all_dataset_paths()
    names = []
    
    for path in all_paths:
        name = path.name
        
        names.append(name)
    return set(names)




def get_default_values():
    return DEFAULT_CONFIGS

def dataset_factory(new=False, copy=False, **kwargs):
    """
    Loads a dataset or creates a new one.
    Parameters
    ----------
    new    : bool, whether to create a new Dataset
    copy   : bool, whether to copy the given dataset
    kwargs : kwargs for the dataset creation
    Returns
    -------
    
    """
    global DATASET_PATH
    if new:
        return Dataset(**kwargs)
    if copy:
        raise NotImplementedError
    return Dataset.load(DATASET_PATH / kwargs["name"] / "dataset.pickle")

class Dataset:
    def __init__(self, name, path):
        
        self.path = Path(path)
        self.name = name
        names = get_all_dataset_names()
        if name in names:
            raise ValueError(f"{name} already exists as a Dataset.")

        # for key, default_value in get_default_values().items():
        #     setattr(self, key.replace(" ","_").lower(), default_value)

        self.micrograph_paths = []
        self.segmentation_paths = {}
        self.pixelSizes = {}
        self.analysers = {}
        now = datetime.now()
        self.last_run_kwargs = {}
        self.times = {"Created":now.strftime("%Y-%m-%d, %H:%M:%S"),
                      "Last changed":now.strftime("%Y-%m-%d, %H:%M:%S"),
                      "Last run":""}
        self.save()
        


    def save(self):
        """
        Save this dataset
        Parameters
        ----------


        Returns
        -------
        
        """
        self.changed()
        create_dir(self.dataset_path)
        create_dir(self.pickel_path)
        path = self.pickel_path / "dataset.pickle"
        with open(path, "wb") as f:
            pickle.dump(self, f)
        

    def changed(self):
        now = datetime.now()
        self.times["Last changed"] = now.strftime("%m/%d/%Y, %H:%M:%S")
    
    def addMicrographPaths(self, paths):
        """
        Adds the paths as new micrograph paths for analysis.
        Parameters
        ----------


        Returns
        -------
        
        """
        known_paths = set(self.micrograph_paths)
        for path in paths:
            if path not in known_paths:
                self.micrograph_paths.append(path)
        
                self.segmentation_paths[path] = None
                
        self.save()
    
    def addSegmentationPaths(self, paths, replace=None):
        """
        Adds new segmentation paths to the micrograph paths.
        Parameters
        ----------
        paths : the paths to add
        replace : a tuple of strings, second value gets replaced by the first to look for micrograph names. If this is none, a similarity search is performed.

        Returns
        -------
        
        """
        
        seg_files = [path for path in self.segmentation_paths.values() if path is not None]
        seg_files.extend([Path(path) for path in paths])

      
        if len(self.micrograph_paths) < len(seg_files):
            print("Got more segmentation files than needed")
            print(len(self.micrograph_paths), len(seg_files))
            return
        if replace is None:
            similarity_matrix = np.zeros((len(self.micrograph_paths), len(seg_files)))
            for counter_1, file in enumerate(self.micrograph_paths):
                for counter_2, seg_file in enumerate(seg_files):

                    similarity_matrix[counter_1, counter_2] = difflib.SequenceMatcher(None, Path(file).stem, Path(seg_file).stem ).ratio()
            argmaxs = np.argmax(similarity_matrix, 0)
            
            for counter, argmax in enumerate(argmaxs):
                self.segmentation_paths[self.micrograph_paths[argmax]] = seg_files[counter]
        else:
            micrograph_names = {path.name: path for path in self.micrograph_paths}
            segmentation_names = {path.name: path for path in paths}
            for key, value in segmentation_names.items():
                replaced_name = key.replace(replace[1], replace[0])
                if replaced_name in micrograph_names:
                    self.segmentation_paths[micrograph_names[replaced_name]] = value

        self.save()


    def isHealthy(self, stopEvent=None):

        
        missingFiles = {"micrographs":0, "segmentations":0, "analysers":0, "csv":0}

        filesAreMissing = False
        for i in self.micrograph_paths:
            if stopEvent is not None:
                if stopEvent.is_set():
                    return True, {}
            if not os.access(i, os.F_OK):
            
                missingFiles["micrographs"] += 1
                filesAreMissing = True
        for i in self.segmentation_paths.values():
            if stopEvent is not None:
                if stopEvent.is_set():
                    return True, {}
            if i is None:
                continue
            if not os.access(i, os.F_OK):
                missingFiles["segmentations"] += 1
                filesAreMissing = True
        for i in self.analysers.values():
            if stopEvent is not None:
                if stopEvent.is_set():
                    return True, {}
            if not os.access(i, os.F_OK):
                missingFiles["analysers"] += 1
                filesAreMissing = True

        if not (self.pickel_path / "membranes.csv").exists():
            missingFiles["csv"] += 1

        return not filesAreMissing, missingFiles

    @property
    def csv(self):
        """
        Returns the csv dataframe with all information about the membranes
        """
        csv_path = self.pickel_path / "membranes.csv"
        try:
            df = pd.read_csv(csv_path, header=0)
        except FileNotFoundError as e:
            
            df = pd.DataFrame([], columns=["Circumference","Diameter", "Area","Shape","Shape probability","Thickness","Closed","Min thickness","Max thickness",
                                           "Min curvature","Max curvature","Is probably ice","Circularity", "Is enclosed","Enclosed distance","Index","Micrograph"])
        df["Micrograph"] = df["Micrograph"].astype(str)
        return df 

    def to_csv(self, njobs=1, csv=None):
        """
        Loads the data and writes the csv files.
        Parameters
        ----------
        njobs : int, number of processes to load data
        csv   : if None, loads the data and writes csv file, else writes csv in the csv file

        Returns
        -------
        csv   : pandas dataframe
        """
        
        if csv is None:
            if self.isZipped:
                raise ValueError(f"Cannot perform actions because {self.name} is still zipped. Unzip the dataset first.")
            csv = self.load_data(njobs, "csv")
            if len(csv) == 0:
                return
            csv = pd.concat(csv)
        csv.to_csv(self.pickel_path / "membranes.csv" ,index=False)
        csv["Micrograph"] = csv["Micrograph"].astype(str)
        return csv
        

    def complete_run_kwargs(self, run_kwargs):
        """
        Takes run arguments dictionary and completes the missing entries.
        Parameters
        ----------
        run_kwargs : the run arguments to complete

        Returns
        -------
        run_kwargs : the completed arguments dict
        """
        default = get_default_values()
        for func, params in default.items():
            if func not in run_kwargs:
                run_kwargs[func] = {}
            for key, value in params.items():
                if key not in run_kwargs[func]:
                    if func in self.last_run_kwargs and key in self.last_run_kwargs[func]:
                        run_kwargs[func][key] = self.last_run_kwargs[func][key]
                    else:
                        run_kwargs[func][key] = value
        return run_kwargs

    def run(self, njobs=1,threads=10, gpu=None, only_segmentation=False, run_kwargs={}, use_csv=True, tqdm_file=sys.stdout, stopEvent=None):
        """
        Runs the analysis on all available micrographs with the given parameters
        Parameters
        ----------
        njobs       : number of parallel processes
        threads     : number of threads
        gpu         : gpus to use as given by "get logical devices"
        only_segmentation : Whether to run only the segmentation
        run_kwargs  : the arguments for the analysis
        use_csv     : bool, whether to look up the csv file and remove the missing entries also in the analysis
        tqdm_file   : where to print out tqdm progress bar
        stopEvent   : an event to stop the analysis

        Returns
        -------
        
        """
        if self.isZipped:
            raise ValueError(f"Cannot perform actions because {self.name} is still zipped. Unzip the dataset first.")
        from keras.backend import clear_session
        if stopEvent is None:
            stopEvent = mp.get_context(MP_START_METHOD).Event()
        analysers = []
        args = []
        run_kwargs = deepcopy(run_kwargs)
        run_kwargs = self.complete_run_kwargs(run_kwargs)
        self.last_run_kwargs = run_kwargs
        # if isinstance(self.segmentation_model, (str, Path)):
        segDict = get_segmentation_dict()
        #     print(segDict[self.segmentation_model])
        segModel = segmentationModelFactory(filepath=segDict[run_kwargs["segmentation"]["segmentation_model"]])
        # else:
        #     segModel = self.segmentation_model

        

        
       
        classDict = get_classifier_dict()
        shapeClassifier = ShapeClassifierFactory(filepath=classDict[run_kwargs["shapePrediction"]["shape_classifier"]])
        rerun = run_kwargs["general"]["rerun"]
        del run_kwargs["general"]["rerun"]
        if not run_kwargs["segmentation"]["rerun_segmentation"]:
            to_segment = [micrograph for micrograph in self.micrograph_paths if micrograph not in self.segmentation_paths or self.segmentation_paths[micrograph] is None or not Path(self.segmentation_paths[micrograph]).exists()]
        else:
            to_segment = [micrograph for micrograph in self.micrograph_paths]
            rerun = True
            use_csv = False

        if len(to_segment) > 0:
            if gpu is None:
                gpu = get_logical_devices("GPU")

                if len(gpu) == 0:
                    gpu = get_logical_devices("CPU")
            pixelSizes = [self.pixelSizes[m] if m in self.pixelSizes else None for m in to_segment ]
            results = segModel.predict_multiprocessing( to_segment, pixelSizes, gpu, run_kwargs, njobs=njobs,threads=threads,tqdm_file=tqdm_file, dataset_name=self.name, stopEvent=stopEvent, seg_path=self.dataset_path, mask_path=self.mask_path)
            if stopEvent.is_set():
                self.save()
                return
            if results is None:
                self.save()
                return
            for key, value in results.items():
                self.segmentation_paths[key] = value
                
            
        gc.collect()
        clear_session()
        gc.collect()
        self.save()
        if run_kwargs["segmentation"]["only_segmentation"]:
            return
        if only_segmentation:
            
            return


        empty_micrographs = []
        segmentation_paths = [self.segmentation_paths[micrograph] if micrograph in self.segmentation_paths else None for micrograph in self.micrograph_paths]
        analyser_paths = [self.analysers[micrograph] if micrograph in self.analysers and Path(self.analysers[micrograph]).exists() else None for micrograph in self.micrograph_paths ]
        

        if use_csv:
            csv = self.csv

            csvs = [csv[csv["Micrograph"] == Path(micrograph).stem]["Index"].to_list() for micrograph in self.micrograph_paths]
        else:
            csvs = [None for _ in self.micrograph_paths]

        time_used = {}

        manager = mp.Manager()
        pixelSizes = [self.pixelSizes[m] if m in self.pixelSizes else None for m in self.micrograph_paths ]


        input_queue = manager.Queue()
        output_queue =  manager.Queue()
        error_queue = manager.Queue()
        # input_queue = Queue()

        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_gpus = os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            visible_gpus = None
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        

        lock = manager.Lock()
        analyser_proccesses = [mp.get_context(MP_START_METHOD).Process(target=run_analysis, args=[input_queue, output_queue, stopEvent, njobs,q, lock, self.mask_path, error_queue]) for q in range(threads)]
        [input_queue.put((micrograph, segmentation, analyser, rerun, run_kwargs, counter, self.dataset_path, shapeClassifier, csv, ps)) for counter, (micrograph, segmentation, analyser, csv, ps) in
                        enumerate(zip(self.micrograph_paths, segmentation_paths, analyser_paths, csvs, pixelSizes))]
        
        [p.start() for p in analyser_proccesses]
        
            # with mp.get_context(MP_START_METHOD).Pool(njobs) as pool:
            
        with tqdm(total=len(analyser_paths), desc=f"{self.name}: Analyzed files ", smoothing=0,file=tqdm_file ) as pbar:
            while any([process.is_alive() for process in analyser_proccesses]):
                
                if stopEvent.is_set():

                    [res.terminate() for res in analyser_proccesses]
                    self.save()
                    while any([res.is_alive() for res in analyser_proccesses]):
                        time.sleep(0.1)
                    return
                
                # if all([res.done() for res in results]) and output_queue.empty():
                    
                #     break
                try:
                    micrograph, path, seg_path, times = output_queue.get(timeout=5)
                except Exception as e:

                    continue

                try:
                    pbar.update(1)
                except:
                    pass

                if micrograph is None:
                    continue
            
                if path is None:
                    empty_micrographs.append(micrograph)
                    continue

                if times is not None:
                    for key, value in times.items():
                        if key not in time_used:
                            time_used[key] = 0
                        time_used[key] += value
                self.analysers[micrograph] = path
                self.segmentation_paths[micrograph] = seg_path
                analysers.append(path)
                self.save()
        if visible_gpus is None:
            del os.environ["CUDA_VISIBLE_DEVICES"]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible_gpus

        foundErrors = any([process.exitcode > 0 for process in analyser_proccesses])


        if foundErrors:
            tqdm_file.write("Something went wrong during analysis. Following errors occurred:\n")
            while not error_queue.empty():
                tqdm_file.write(error_queue.get())
                tqdm_file.write("\n")
            return
        self.to_csv(njobs)
        now = datetime.now()
        self.times["Last run"] = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.save()
        return analysers, args
    
    def remove(self, remove_data=False):
        """
        Removes this dataset.
        Parameters
        ----------
        remove_data : bool, whether to also remove all the analysed data or only the dataset pickle file and the csv file

        Returns
        -------
        
        """

        if remove_data:
            if self.isZipped:
                os.remove(self.zip_path)
            else:
                for m in self.micrograph_paths:
                    if m in self.analysers:
                        wrapper = Analyser.load(self.analysers[m], type_="Wrapper")
                        wrapper.remove()
                    elif m in self.segmentation_paths and self.segmentation_paths[m] is not None:

                        if Path(self.segmentation_paths[m]).parent == self.dataset_path:
                            shutil.rmtree(self.segmentation_paths[m]) 
                shutil.rmtree(self.mask_path)
            if len(os.listdir(self.dataset_path)) == 0:
                shutil.rmtree(self.dataset_path)
        
        remove_path = self.pickel_path
        shutil.rmtree(remove_path)

    @property
    def zip_path(self) -> Path:
        return self.dataset_path / f"{self.name}.tar.gz"

    @property
    def isZipped(self):
    
        return self.zip_path.exists()

    def zip(self):
        """
        Zips the dataset to reduce size and file number
        Parameters
        ----------


        Returns
        -------
        
        """
        if self.isZipped:
            return
        files_to_remove = []
        dirs_to_remove = []
        output_filename = self.zip_path
        directory_path = self.dataset_path


        with tarfile.open(output_filename, "w:gz") as tar:
            # for dirpath, dirs, files in tqdm(os.walk(directory_path)): 

            #     for filename in files:
                    # item_path = Path(dirpath) / filename
            for item in tqdm(os.listdir(directory_path)):
                item_path = Path(directory_path) / item
                if item_path == output_filename:
                    continue
                
                tar.add(item_path, arcname=item)
                
                if item_path.is_dir():
                    dirs_to_remove.append(item_path)
                    
                else:
                    files_to_remove.append(item_path)

        
        for file in files_to_remove:
            os.remove(file)
        for directory in dirs_to_remove:
            shutil.rmtree(directory)

    def unzip(self):
        """
        Unzip the dataset if it is zipped
        Parameters
        ----------


        Returns
        -------
        
        """
        if not self.isZipped:
            raise ValueError(f"Cannot unzip {self.name}. It is not zipped.")
        archive_path = self.zip_path
        output_directory = self.dataset_path
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(output_directory)
        os.remove(archive_path)


    def rename(self, new_name):
        """
        Rename this dataset and change the corresponding attributes 
        Parameters
        ----------
        new_name : the new name to rename the dataset to

        Returns
        -------
        
        """
        global DATASET_PATH
        datasets = get_all_dataset_names()
        if new_name in datasets:
            raise FileExistsError(f"Dataset {new_name} already exists.")
        new_dir = Path(self.path) / new_name
        if new_dir.exists():
            raise FileExistsError(f"Directory {new_dir} already exists.")
        
        if self.isZipped:
            raise ValueError(f"Cannot perform actions because {self.name} is still zipped. Unzip the dataset first.")
        

        new_micrograph_paths = []
        new_segmentations = {}
        new_analysers = {}
        str_old_path = str(self.dataset_path)
        str_new_path = str(new_dir)
        for micrograph in self.micrograph_paths:
            new_micrograph = str(micrograph).replace(str_old_path, str_new_path)
            new_micrograph_paths.append(new_micrograph)
            if micrograph in self.segmentation_paths:
                new_segmentations[new_micrograph] = str(self.segmentation_paths[micrograph]).replace(str_old_path, str_new_path)
            if micrograph in self.analysers:
                new_analysers[new_micrograph] = str(self.analysers[micrograph]).replace(str_old_path, str_new_path)
        

        shutil.move(self.dataset_path, new_dir)
        self.micrograph_paths = new_micrograph_paths
        self.segmentation_paths = new_segmentations
        self.analysers = new_analysers
        
            


        shutil.move(self.pickel_path, DATASET_PATH / new_name)
        self.name = new_name
        self.save()

    def removeMicrographPaths(self, idxs=None, paths=None, all=False):
        """
        Removes the given micrograph from this dataset and removes the segmentation and analysers too.
        Parameters
        ----------
        idxs     : list of indexes to remove
        paths    : list of paths to remove
        all      : bool, whether to remove all paths

        Returns
        -------
        
        """
        def removeIdx(idx):
            micrograph = self.micrograph_paths[idx]
            if micrograph in self.segmentation_paths:
                del self.segmentation_paths[micrograph]
            if micrograph in self.analysers:
                del self.analysers[micrograph]
            if micrograph in self.pixelSizes:
                del self.pixelSizes[micrograph]
            self.micrograph_paths.pop(idx)
        if all:
            self.micrograph_paths = []
            self.segmentation_paths = {}
            self.analysers = {}
            self.save()
            return
        if isinstance(idxs, int):
            removeIdx(idxs)
        elif isinstance(idxs, (tuple, list)):
            for idx in sorted(idxs, reverse=True):
                removeIdx(idx)
                
                
        elif isinstance(paths, (tuple, list)):
            for path in paths:
                try:
                    idx = self.micrograph_paths.index(str(path))
                except:
                    path = Path(path)    
                    idx = self.micrograph_paths.index(path)
                removeIdx(idx)
        elif isinstance(paths, (str, Path)):
            try:

                idx = self.micrograph_paths.index(str(paths))
            except:
                idx = self.micrograph_paths.index(Path(paths))
            removeIdx(idx)
        self.save()
    

    def loadMicrographCsv(self, file):
        df = pd.read_csv(file, header=None, sep=",", index_col=False)
        if len(df.columns) == 1:
            self.addMicrographPaths(df[0])
        else:
            self.addMicrographPaths(df[0])
            for i, row in df.iterrows():
                self.segmentation_paths[row[0]] = row[1]
                if row[0] in self.analysers:
                    self.analysers.pop(row[0])
 
        



    def load_data(self, njobs=1, type_="Analyser", rewrite_dir=False, max_=None, use_csv=False):
        """
        Load the data of the specific type.
        Updates from csv only applies if type_ is Analyser
        Parameters
        ----------
        njobs      : number of parallel processes to load the data
        type_      : type of data to load, can be Analyser, Json, Wrapper, csv or Segmentation
        rewrite_dir: bool, whether to save the analyser again after loading (useful to remove deleted membranes from the csv)
        max_       : maximum number of analyser, wrappers etc to load
        use_csv    : bool, whether to use the csv file to remove deleted membranes
        Returns
        -------
        data       : the data of the provided type
        """

        micrograph_paths = [m for m in self.micrograph_paths if m in self.analysers]
        if use_csv:
            csv = self.csv

            csvs = [csv[csv["Micrograph"] == Path(micrograph).stem]["Index"].to_list() for micrograph in micrograph_paths]
        else:
            csvs = [None for _ in self.micrograph_paths]
        analysers = [self.analysers[m] for m in micrograph_paths]
        if max_ is None or max_ >= len(micrograph_paths):
            
            data = Analyser.load(analysers, njobs=njobs, type_=type_, rewrite_dir=rewrite_dir, dataset_path=self.dataset_path, index=csvs)
            
        else:
            data = Analyser.load(analysers[:max_], njobs=njobs, type_=type_, rewrite_dir=rewrite_dir, dataset_path=self.dataset_path, index=csvs[:max_])
        if type_ == "Json":
            data = {value["Micrograph"]:value["Points"] for value in data}
        if type_=="Analyser" and use_csv:
            self.to_csv(njobs=njobs)
        return data
        
    def __len__(self):
        return len(self.micrograph_paths)

    @property
    def mask_path(self):
        mask_path = self.dataset_path / "masks"
        if not mask_path.exists():
            mask_path.mkdir(parents=True)
        return mask_path

    @property
    def pickel_path(self):
        global DATASET_PATH
        return Path(DATASET_PATH) / self.name
    
    @property
    def dataset_path(self):
        return self.path / self.name


    def copy(self, save_dir:Path, print_func=None):
        """
        Create a copy of this dataset and all its data
        Parameters
        ----------
        save_dir    : the directory where to save the new copy

        Returns
        -------
        inst        : the new dataset object
        """
        if self.isZipped:
            raise ValueError(f"Cannot perform actions because {self.name} is still zipped. Unzip the dataset first.")
        name = self.name
        all_names = get_all_dataset_names()
        name_counter = 1
        while True:
            new_name = f"{name}_{name_counter}"
            if new_name not in all_names and not (Path(save_dir) / new_name).exists():
                break
            name_counter += 1
       
        # wrappers = self.load_data(type_="Wrapper")
        # save_dir = save_dir / new_name
        # save_dir.mkdir(parents=True, exist_ok=True)
        new_dataset = Dataset(new_name, save_dir)
        counter = 1

        for m in self.micrograph_paths:
            new_dataset.addMicrographPaths([m])


            if m in self.analysers:

                w: AnalyserWrapper = Analyser.load(self.analysers[m],type_="Wrapper")
                old_path = w.directory
                new_path = new_dataset.dataset_path / old_path.name
                
                shutil.copytree(old_path, new_path)
                new_dataset.analysers[m] = new_path
                if (new_path / "segmentation.npz").exists():
                    new_dataset.segmentation_paths[m] = new_path / "segmentation.npz"
            elif m in self.segmentation_paths:
                new_seg_path = new_dataset.dataset_path / Path(self.segmentation_paths[m]).name
                shutil.copy(self.segmentation_paths[m], new_seg_path)
                new_dataset.segmentation_paths[m] = new_seg_path
            if print_func is not None and (counter % 10 == 0 or counter == len(self.micrograph_paths)):
                print_func(f"Copying {self.name}: {counter}/{len(self.micrograph_paths)} files\n")
            counter += 1
        
        new_dataset.to_csv()
        new_dataset.save()
        return new_dataset

    @staticmethod
    def load(p):
        """
        Load the dataset by the given path.
        Parameters
        ----------
        p   : Path to the dataset to load

        Returns
        -------
        inst    : the loaded dataset object
        """
        
        global DATASET_PATH
        path = Path(p)

            
        
        try:
            if path.suffix == ".pickle" and path.is_file():
                with open(path, "rb") as f:
                    dataset = CustomUnpickler(f).load()
            elif path.is_dir() and (path / "dataset.pickle").exists():
                with open(path / "dataset.pickle", "rb") as f:
                    dataset = CustomUnpickler(f).load()
            else:
                raise FileNotFoundError(path)
            if not hasattr(dataset, "times"):
                now = datetime.now()
                dataset.times = {"Created":now.strftime("%Y-%m-%d, %H:%M:%S"),
                        "Last changed":now.strftime("%Y-%m-%d, %H:%M:%S"),
                        "Last run":""}
            if not hasattr(dataset, "last_run_kwargs"):
                dataset.last_run_kwargs = {}
            if not hasattr(dataset, "pixelSizes"):
                dataset.pixelSizes = {}
        except FileNotFoundError as e:
            path = DATASET_PATH / p
            if path.suffix == ".pickle" and path.is_file():
                with open(path, "rb") as f:
                    dataset = CustomUnpickler(f).load()
            elif path.is_dir() and (path / "dataset.pickle").exists():
                with open(path / "dataset.pickle", "rb") as f:
                    dataset = CustomUnpickler(f).load()
            else:
                raise FileNotFoundError(path)
            if not hasattr(dataset, "times"):
                now = datetime.now()
                dataset.times = {"Created":now.strftime("%Y-%m-%d, %H:%M:%S"),
                        "Last changed":now.strftime("%Y-%m-%d, %H:%M:%S"),
                        "Last run":""}
            if not hasattr(dataset, "last_run_kwargs"):
                dataset.last_run_kwargs = {}
            if not hasattr(dataset, "pixelSizes"):
                dataset.pixelSizes = {}


        # conversion_date = datetime(2025,5,26,15,45)

        strings = ["Created", "Last changed", "Last run"]
        changed = False
        for string in strings:
            if len(dataset.times[string]) == 0:
                continue
            time_str = dataset.times[string]
            try:
                dt_object = datetime.strptime(time_str, "%m/%d/%Y, %H:%M:%S")
            except ValueError:
                continue
            
            # if dt_object < conversion_date:
            dataset.times[string] = dt_object.strftime("%Y-%m-%d, %H:%M:%S")
            changed = True
        if changed:
            dataset.save()



        return dataset
    

def print_error(error):
    print(error, flush=True)

def run_analysis(input_queue, outputqueue, stopEvent, njobs, q, lock, mask_path, error_queue ):
    """
    Run the analysis, only called by the run method of datasets.
    """
    from cryovia.gui.starting_menu import changeToDebug
    if os.environ["CRYOVIA_MODE"] is not None and int(os.environ["CRYOVIA_MODE"]) == 1:
        changeToDebug()
    test_counter = 0
    try:
        with mp.get_context(MP_START_METHOD).Pool(njobs) as pool:
            while True:
                try:
                    try:
                        with lock:
                            micrograph, segmentation, analyser, rerun, kwargs, counter, dataset_path, shape_classifier,index, ps = input_queue.get_nowait()
                    except Exception as e:
                        return

                    kwargs = deepcopy(kwargs)
                    if stopEvent is not None and stopEvent.is_set():
                        
                        return
                    
                    times = {}
                    now = datetime.now()


                    if analyser is not None and not rerun:
                        if kwargs["general"]["only_run_for_new_data"]:
                            
                            outputqueue.put((None, None, None, None))
                            continue
                        try:
                            analyser = Analyser.load(analyser, dataset_path=dataset_path, index=index)
                        except:
                            wrapper = AnalyserWrapper(analyser, True)
                            analyser = wrapper.analyser
                            analyser.segmentation_path = segmentation
                        
                    else:
                        del kwargs["general"]["only_run_for_new_data"]
                        create_kwargs = {}
                        if "general" in kwargs:
                            create_kwargs = kwargs["general"]
                            create_kwargs["micrograph_pixel_size"] = ps
                        if Path(segmentation).exists():
                            analyser = Analyser(micrograph, dataset_path, segmentation, njobs, name=str(counter), pool=pool, **create_kwargs)
                        elif analyser is not None:
                            analyser = Analyser.load(analyser, dataset_path=dataset_path, index=index)
                            analyser = Analyser(micrograph, dataset_path, analyser.segmentation_path, njobs, name=str(counter), pool=pool, **create_kwargs)
                        else:
                            raise FileNotFoundError(segmentation)
                    if stopEvent is not None and stopEvent.is_set():
                        
                        return
                        
                    times["Loading/Creating"] = (datetime.now() - now).total_seconds()
                    now = datetime.now()
                    if not hasattr(analyser, "micrograph_path_"):
                        analyser.micrograph_path = micrograph
                    if not hasattr(analyser, "segmentation_path_"):
                        wrapper = Analyser.load(micrograph, type_="Wrapper")
                        seg_path = wrapper.directory / "segmentation.npz"
                        if seg_path.exists():
                            analyser.segmentation_path_ = seg_path 
                    
                    if len(analyser.membranes) == 0:
                        path = analyser.save()
                        
                        outputqueue.put((micrograph, path, analyser.segmentation_path, None))
                        continue
                    
                    if kwargs["run"]["maskGrid"]:
                        mask_kwargs= {}
                        if "maskGrid" in kwargs:
                            mask_kwargs = kwargs["maskGrid"]
                        mask = None
                        if "use_existing_mask" in mask_kwargs and mask_kwargs["use_existing_mask"]:
                            current_mask_path = mask_path / (Path(micrograph).stem + "_mask.pickle")
                            if current_mask_path.exists():
                                if current_mask_path.suffix == ".pickle":
                                    self = mask_file.load(current_mask_path)
                                    mask = self.create_mask()
                                    
                                else:
                                    mask,_ = load_file(current_mask_path)

                        
                        
                        else:
                            mask = find_grid_hole_per_file(analyser.micrograph_path, **mask_kwargs)
                        if mask is not None:
                            analyser.applyMask(mask)
                        times["Mask grid"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()
                    
                    if kwargs["run"]["identifyIceContamination"]:
                        
                        analyser.identifyIceContaminations()
                        times["Ice contamination"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()

                    if stopEvent is not None and stopEvent.is_set():
                        return
                    if kwargs["run"]["estimateCurvature"]:
                        curvature_kwargs = {}
                        if "estimateCurvature" in kwargs:
                            curvature_kwargs = kwargs["estimateCurvature"]
                        analyser.estimateCurvature(pool=pool, **curvature_kwargs)
                        times["Curvature Estimation"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()
                    if stopEvent is not None and stopEvent.is_set():
                        return
                    if kwargs["run"]["estimateThickness"]:
                        thickness_kwargs = {}
                        if "estimateThickness" in kwargs:
                            thickness_kwargs = kwargs["estimateThickness"]
                        analyser.estimateThickness(pool=pool, **thickness_kwargs)
                        times["thickness estimation"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()
                        
                        
                        
                    if stopEvent is not None and stopEvent.is_set():
                        return
                    if kwargs["run"]["shapePrediction"]:
                        analyser.predictShapes(shape_classifier)
                        times["Shape prediction"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()

                    analyser.calculateBasicAttributes()
                    if stopEvent is not None and stopEvent.is_set():
                        return
                    if kwargs["run"]["enclosed"]:
                        analyser.findEnclosedVesicles()
                        times["Enclosed calculation"] = (datetime.now() - now).total_seconds()
                        now = datetime.now()
                    if stopEvent is not None and stopEvent.is_set():
                        return
                    path = analyser.save()
                    times["Saving"] = (datetime.now() - now).total_seconds()
                    now = datetime.now()
                    
                    outputqueue.put((micrograph, path, analyser.segmentation_path, times))
                        
                except Exception as e:
                    raise e

                    outputqueue.put((None, traceback.format_exc(), None, None))
                    continue
    except Exception as e:
        error_queue.put(traceback.format_exc())
        raise e
    

if __name__ == "__main__":
    pass
    