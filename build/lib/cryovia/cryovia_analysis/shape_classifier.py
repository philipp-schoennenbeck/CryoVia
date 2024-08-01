from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path
import pickle
import os
from copy import copy, deepcopy
import numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import confusion_matrix
from cryovia.gui.Unpickler import CustomUnpickler
from keras import Sequential
from keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
from keras.models import load_model
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from tensorflow.config import list_logical_devices
from tensorflow import device
from scipy.interpolate.interpolate import interp1d

from cryovia.gui.path_variables import CLASSIFIER_PATH, SHAPE_CURVATURE_PATH


# CLASSIFIER_PATH = Path().home() / ".cryovia" / "Classifiers"
# SHAPE_CURVATURE_PATH = Path().home() / ".cryovia" / "Shape_curvatures"
PROTECTED_SHAPES = set(["sphere", "hourglass", "pear", "prolate", "stomatocyte", "tube", "elongated_pear"])


def create_dir(path):
    """
    Creates the directory at given path
    Parameters
    ----------
    path : Path to directory

    Returns
    -------
    
    """
    
    if path.exists():
        return
    path.mkdir(parents=True)



def get_all_classifier_paths():
    """
    Extracts all the paths of shape classifier files.
    Parameters
    ----------


    Returns
    -------
    classifiers_paths : list of all the paths
    """
    global CLASSIFIER_PATH
    create_dir(CLASSIFIER_PATH)
    classifiers_paths = []
    for file in os.listdir(CLASSIFIER_PATH):
        file = CLASSIFIER_PATH / file
        if file.suffix == ".pickle":
            classifiers_paths.append(file)
    
    return classifiers_paths

def get_classifier_dict():
    """
    Extracts all the paths of shape classifier files and returns a dictionary with name as key and path as vale.
    Parameters
    ----------


    Returns
    -------
    names : dict of classifiers
    
    """
    all_paths = get_all_classifier_paths()
    names = {}
    length_of_suffix = len("_classifier.pickle")
    for path in all_paths:
        name = path.name
        name = name[:-length_of_suffix]
        names[name] = path
    return names

def get_all_classifier_names():
    """
    Extracts all the names of available classifiers
    Parameters
    ----------


    Returns
    -------
    names : set of classifier names
    """
    all_paths = get_all_classifier_paths()
    names = []
    length_of_suffix = len("_classifier.pickle")
    for path in all_paths:
        name = path.name
        name = name[:-length_of_suffix]
        names.append(name)
    return set(names)

def get_all_shape_curvature_paths():
    """
    Extracts all the paths of shape curvature files
    Parameters
    ----------


    Returns
    -------
    shape_curvature_paths : list of paths
    
    """
    global SHAPE_CURVATURE_PATH
    create_dir(SHAPE_CURVATURE_PATH)
    shape_curvature_paths = []
    for file in os.listdir(SHAPE_CURVATURE_PATH):
        file = SHAPE_CURVATURE_PATH / file
        if file.suffix == ".npy":
            shape_curvature_paths.append(file)
    
    return shape_curvature_paths

def get_all_shapes():
    """
    Extracts the names of avilable shape curvatures
    Parameters
    ----------


    Returns
    -------
    names : set of all the available names
    """
    all_shape_curvature_paths = get_all_shape_curvature_paths()
    return set([path.stem for path in all_shape_curvature_paths])



class ShapeClassifier(object):
    """
    Model to classify shapes by curvature
    """
    def __init__(self, name=None,):
        
        self.name = name
        self.changed_ = False
        self.changed_hooks = []
        self.class_set = set()
        self.lastTrainedClasses = ["Default"]
        self.confusion_matrix = None
        self.type_ = "GradientBoostingClassifier"
        self.only_closed_ = True
        self.create_gbc()
        self.save()
        
    @property
    def only_closed(self):
        """
        Whether to only used closed vesicles.
        """
        return self.only_closed_
    
    @only_closed.setter
    def only_closed(self, value):
        if not self.writable:
            return None
        if value != self.only_closed_:
            self.only_closed_ = value
            self.changed = True

    @property
    def type(self):
        """
        Whether the classifier uses a neural network or Gradient Boosting
        """
        return self.type_
    
    @type.setter
    def type(self, value):
        
        self.changeClassifier(value)
            

    
    

    def changeClassifier(self, type_="GradientBoostingClassifier"):
        """
        Change the type of this classifier
        Parameters
        ----------
        type_   : The type to change to. Has to be NeuralNetwork or GradientBoostingClassifier

        Returns
        -------
        type_ : the type of model this classifier uses
        """
        if not self.writable:
            return self.type_
        if self.type_ == type_:
            return self.type_
        assert type_ in ("GradientBoostingClassifier", "NeuralNetwork") 
        self.changed = True

        if type_ == "GradientBoostingClassifier":
            if self.nnWeightsPath.exists():
                os.remove(self.nnWeightsPath)
            self.create_gbc()
        self.type_ = type_
        self.save()
        return self.type_
    
    @property
    def nnWeightsPath(self):
        """
        The path to this classifier weights.
        """
        global CLASSIFIER_PATH
        return CLASSIFIER_PATH / f"{self.name}_weights.h5"
    
    def create_gbc(self):
        """
        Creates the Gradient boosting classifier
        Parameters
        ----------


        Returns
        -------
        
        """
        self.gbc = GradientBoostingClassifier()
        self.changed = True

    @property
    def gbc(self):
        if self.type_ == "GradientBoostingClassifier":
            return self.gbc_
        else:
            if self.gbc_ is None or isinstance(self.gbc_, GradientBoostingClassifier ):

                self.gbc_ = self.loadWeightsAndModel()

                return self.gbc_ 
            else:
                
                return self.gbc_
    
    @gbc.setter
    def gbc(self, value):
        if isinstance(value, GradientBoostingClassifier):
            self.gbc_ = value
        else:
            self.gbc_ = value
            self.saveWeights(value)

    
    def saveWeights(self, model):
        """
        Save the weights of the given model
        Parameters
        ----------
        model : Classifier

        Returns
        -------
        
        """
        model.save(self.nnWeightsPath,save_format="h5")
    
    def loadWeightsAndModel(self):
        """
        Loads the current model and its weights.
        Parameters
        ----------


        Returns
        -------
        
        """
        if self.nnWeightsPath.exists():
            model = createNeuralNetworkClassifier(self.lastTrainedClasses, 100)
            cpus = list_logical_devices('CPU')
            with device(cpus[0]) as d: 
                model.load_weights(self.nnWeightsPath)
            
            return model
        else:
            return createNeuralNetworkClassifier(self.lastTrainedClasses, 100)

    def predict(self, curvature):
        """
        Predict the shapes of the given curvatures
        Parameters
        ----------
        curvature : List of lists of curavture values to predict the shape for

        Returns
        -------
        shapes : list of predicted shapes
        probabilities: list of the probabilities of the predicted shapes
        """
        def reorder(curvatures):
            """
            Reorder the given curvatures to be more uniform
            """
            new_curvatures = np.zeros_like(curvatures)
            for i, curvs in enumerate(curvatures):
                curvs = np.roll(curvs, -np.argmin(curvs))

                peaks, info_dict = find_peaks(curvs,distance=max(1,len(curvs)/3),height=min(1,np.max(curvs)- np.std(curvs)/3))

                if len(peaks) == 0:
                    new_curvatures[i] = curvs
                    
                    continue
                max_curv_idx = peaks[0] 
                dist_between = max_curv_idx
                dist_to_corners = len(curvs)-1 - max_curv_idx
                if dist_between < dist_to_corners:
                    pass
                else:
                    curvs = np.flip(curvs)
                    curvs = np.roll(curvs,1)
                new_curvatures[i] = curvs
            return new_curvatures
        ndim = 2
        if np.array(curvature).ndim == 1:
            ndim = 1
            curvature = np.array([curvature])
        if self.only_closed:
            reordered = reorder(curvature)
        else:
            reordered = curvature

        model = self.gbc
        if isinstance(model, GradientBoostingClassifier):

            shape = self.gbc.predict(reordered)
            probability = np.round(np.max(self.gbc.predict_proba(reordered),-1),3)
            if ndim == 1:
                shape = shape[0]
                probability = probability[0]
        else:
            reordered = np.expand_dims(reordered,-1)
            cpus = list_logical_devices('CPU')
            with device(cpus[0]) as d:
                prediction = model.predict(reordered, batch_size=64, verbose=0)
            class_prediction = np.argmax(prediction,-1)
            # class_dict = {counter:c for counter, c in enumerate(sorted(list(self.classes[0])))}
            shape = [self.lastTrainedClasses[i] for i in class_prediction]
            probability = np.max(prediction,-1) / np.sum(prediction,-1)

            if ndim == 1:
                shape = shape[0]
                probability = probability[0]

        return shape, probability



    def loadData(self, oneHot=False):
        """
        Load all the Data of the used shapes.
        Parameters
        ----------
        oneHot : Bool, whether to return oneHot encoding for neural network

        Returns
        -------
        all_curvatures : the curvature values loaded in
        all_shapes     : the shape classes of the loaded curvature values
        class_dict     : The dictionary of which oneHot encoding corresponds to which class
        """
        global SHAPE_CURVATURE_PATH
        def read_file(shape):
            
            curvatures = []
            file = SHAPE_CURVATURE_PATH / f"{shape}.npy"
            curvatures = np.load(file)
            shapes = [shape for _ in range(len(curvatures))]
            return shapes, curvatures
            
        def reorder(curvatures):
            new_curvatures = np.zeros_like(curvatures)
            for i, curvs in enumerate(curvatures):
                curvs = np.roll(curvs, -np.argmin(curvs))

                peaks, info_dict = find_peaks(curvs,distance=max(1,len(curvs)/3),height=min(1,np.max(curvs)- np.std(curvs)/3))

                if len(peaks) == 0:
                    new_curvatures[i] = curvs
                    
                    continue
                max_curv_idx = peaks[0]
                dist_between = max_curv_idx
                dist_to_corners = len(curvs)-1 - max_curv_idx
                if dist_between < dist_to_corners:
                    pass
                else:
                    curvs = np.flip(curvs)
                    curvs = np.roll(curvs,1)
                new_curvatures[i] = curvs
            return new_curvatures


        all_shapes = []
        all_curvatures = []
        # usable_shapes, _,_ = self.classes
        usable_shapes = sorted(list(self.class_set))
        for shape in usable_shapes:
            
            shape_list, curvature_list = read_file(shape)
            # if shape == "tube":
            #     shape_list = shape_list[250:]
            #     curvature_list = curvature_list[250:]
            if len(curvature_list) > 0:
                all_shapes.extend(shape_list)
                all_curvatures.append(curvature_list)
        
        if oneHot:
            class_dict = {c:counter for counter, c in enumerate(sorted(list(usable_shapes)))}
            oneHot_shapes = []
            for shape in all_shapes:
                oneHot_enc = np.zeros(len(class_dict))
                oneHot_enc[class_dict[shape]] = 1
                oneHot_shapes.append(oneHot_enc)
            all_shapes = oneHot_shapes
        else:
            class_dict = None
        all_curvatures = np.concatenate(all_curvatures)
        if self.only_closed:
            all_curvatures = reorder(all_curvatures)
        else:
            percentages = np.linspace(0.7, 1, 10, endpoint=True)
            starting_points = np.linspace(0,1,21, endpoint=True)

            length = len(all_curvatures[0])
            new_curvatures = []
            new_shapes = []
            

            chosen_idxs = []
            for percentage in percentages:
                for starting_point in starting_points:
                    starting_idx = int(length * starting_point)
                    end_idx = int((starting_point + percentage) * length)
                    new_idxs = np.linspace(starting_idx, end_idx-1, 100,True )
                    if end_idx <= length:
                        pass
                        # current_curvatures = all_curvatures[:,starting_idx:end_idx]
                    else:
                        first_idxs = new_idxs[new_idxs < length]
                        second_idxs = new_idxs[new_idxs >= length] - length
                        
                        new_idxs = np.concatenate((first_idxs, second_idxs))

                    chosen_idxs.append(new_idxs)
            chosen_idxs = np.array(chosen_idxs)
            for counter, (curvature, shape) in enumerate(zip(all_curvatures, all_shapes)):

                f = interp1d(np.linspace(0,100, len(curvature) ), curvature, "quadratic",)

                y = f(chosen_idxs)
                new_curvatures.append(y)
                new_shapes.extend([shape for _ in range(len(chosen_idxs))])
                    #     current_curvatures = all_curvatures[:, starting_idx:]

                    #     current_curvatures = np.concatenate((current_curvatures, all_curvatures[:, :end_idx - length]), axis=-1)
                    # if current_curvatures.shape[-1] != 100:
                    #     new_current_curvatures = []
                    #     for i in current_curvatures:
                    #         f = interp1d(np.linspace(0,100, len(i) ), i, "quadratic",)

                    #         y = f(np.linspace(0,100,100))
                    #         new_current_curvatures.append(y)
                    #     current_curvatures = new_current_curvatures
                    # new_curvatures.append(np.array(current_curvatures))

                    # new_shapes.extend(all_shapes)

            all_curvatures = np.concatenate(new_curvatures)
            all_shapes = new_shapes

                

        
        return all_curvatures, all_shapes, class_dict



    def train(self):
        """
        Train this classifier on the current given shapes.
        Parameters
        ----------


        Returns
        -------
        
        """
        if len(self.class_set) == 0:
            return

        if self.type_ == "GradientBoostingClassifier":
            all_curvatures, all_shapes, _ = self.loadData()
            test_perc = 0.2
            idxs = np.arange(len(all_shapes))
            np.random.shuffle(idxs)

            train_curvatures, test_curvatures, train_shapes, test_shapes = [],[],[],[]
            unique_labels, unique_label_count = np.unique(all_shapes, return_counts=True)
            shape_counter = {}
            for counter, idx in enumerate(idxs):
                if all_shapes[idx] not in shape_counter:
                    shape_counter[all_shapes[idx]] = 0
                if shape_counter[all_shapes[idx]] > unique_label_count[unique_labels == all_shapes[idx]][0] * test_perc:
                # if counter > len(all_shapes) * test_perc:
                    train_curvatures.append(all_curvatures[idx])
                    train_shapes.append(all_shapes[idx])
                else:
                    test_curvatures.append(all_curvatures[idx])
                    test_shapes.append(all_shapes[idx])
                shape_counter[all_shapes[idx]] += 1
            # for counter, idx in enumerate(idxs):
            #     if counter > len(all_shapes) * test_perc:
            #         train_curvatures.append(all_curvatures[idx])
            #         train_shapes.append(all_shapes[idx])
            #     else:
            #         test_curvatures.append(all_curvatures[idx])
            #         test_shapes.append(all_shapes[idx])

            self.gbc = GradientBoostingClassifier().fit(train_curvatures, train_shapes)
            self.confusion_matrix = (confusion_matrix(self.gbc.predict(test_curvatures), test_shapes), self.gbc.classes_)
            self.save()
        else:
            cpus = list_logical_devices('GPU')
            with device(cpus[0]) as d:
                all_curvatures, all_shapes, class_dict = self.loadData(oneHot=True)
                test_perc = 0.2
                idxs = np.arange(len(all_shapes))
                np.random.shuffle(idxs)
                all_curvatures = np.expand_dims(all_curvatures, -1)
                train_curvatures, test_curvatures, train_shapes, test_shapes = [],[],[],[]
                unique_labels, unique_label_count = np.unique(np.argmax(all_shapes,-1), return_counts=True)
                shape_counter = {}
                for counter, idx in enumerate(idxs):
                    if np.argmax(all_shapes[idx]) not in shape_counter:
                        shape_counter[np.argmax(all_shapes[idx])] = 0
                    if shape_counter[np.argmax(all_shapes[idx])] > unique_label_count[unique_labels == np.argmax(all_shapes[idx])][0] * test_perc:
                    # if counter > len(all_shapes) * test_perc:
                        train_curvatures.append(all_curvatures[idx])
                        train_shapes.append(all_shapes[idx])
                    else:
                        test_curvatures.append(all_curvatures[idx])
                        test_shapes.append(all_shapes[idx])
                    shape_counter[np.argmax(all_shapes[idx])] += 1

                train_curvatures = np.array(train_curvatures)
                test_curvatures = np.array(test_curvatures)
                train_shapes = np.array(train_shapes, dtype=np.float32)
                test_shapes = np.array(test_shapes, np.float32)

                model = createNeuralNetworkClassifierWithoutDevice(self.class_set, np.array(train_curvatures).shape[1])

                self.lastTrainedClasses = sorted(list(self.class_set))

                

                callbacks = []    
                
                callbacks.append(EarlyStopping(patience=10))
                callbacks.append(
                        ModelCheckpoint(self.nnWeightsPath, save_best_only=True,
                                        save_weights_only=True, monitor="val_loss", mode="min"))

                # cpus = list_logical_devices('CPU')
                
                model.fit(train_curvatures, train_shapes,validation_data=(test_curvatures, test_shapes), batch_size=1024, epochs=50, verbose=1, callbacks=callbacks)
            
                prediction = model.predict(test_curvatures, batch_size=64)
            prediction_shapes = np.argmax(prediction, -1)
            
            test_shapes = np.argmax(test_shapes, -1)
            class_dict = {value:key for key, value in class_dict.items()} 

            classes = [class_dict[i] for i in range(len(class_dict))]
            self.confusion_matrix = (confusion_matrix(prediction_shapes, test_shapes), classes)

            self.save()
       


    def create_copy(self):
        """
        Create a copy of this classifier
        Parameters
        ----------


        Returns
        -------
        new_copy : The newly created classifier
        """
        def find_new_name():
            """
            Finds a new name which has not been used yet
            """
            names = get_all_classifier_names()
            counter = 1
            while True:
                new_name = f"{self.name}_{counter}"
                if new_name not in names:
                    return new_name
                counter += 1 
        hooks = self.changed_hooks
        self.changed_hooks = []
        new_copy = deepcopy(self)
        self.changed_hooks = hooks
        new_copy.name = find_new_name()
        if self.type_ != "GradientBoostingClassifier":
            new_copy.saveWeights(self.gbc)
        new_copy.save()
        return new_copy

    def save(self, model=None):
        """
        Save this classifier and a given model
        Parameters
        ----------
        model : A neural network model to save the weights for

        Returns
        -------
        
        """
        with open(self.path, "wb") as f:
            self.changed = False
            hooks = self.changed_hooks
            self.changed_hooks = []
            if self.type_ != "GradientBoostingClassifier":
                current_gbc = self.gbc_
                self.gbc_ = None
            pickle.dump(self, f)
            if self.type_ != "GradientBoostingClassifier":
                
                self.gbc_ = current_gbc
            self.changed_hooks = hooks
            if model is not None:
                self.saveWeights(model)
        

    
    @property
    def changed(self):
        return self.changed_
    
    @changed.setter
    def changed(self, value):
        if value != self.changed:
            self.changed_ = value
            for hook in self.changed_hooks:
                hook()
        

    def rename(self, new_name):
        """
        Renames this classifier and changes the corresponding paths
        Parameters
        ----------


        Returns
        -------
        
        """
        if self.name == new_name or new_name in get_all_classifier_names():
            return
        if self.writable:
            old_filepath = self.path
            old_weights = self.nnWeightsPath
            
            self.name = new_name
            if old_weights.exists():

                self.save(self.loadWeightsAndModel())
            os.remove(old_filepath)
            if old_weights.exists():
                os.remove(old_weights)

    def remove(self):
        """
        Remove this classifier if possible.
        Parameters
        ----------


        Returns
        -------
        
        """
        if self.writable:
            # print(f"Removing {self.name} : {self.path}")
            os.remove(self.path)
        else:
            print("Tried removing unwritable default classifier!")

    def __str__(self) -> str:
        if self.changed:
            return self.name + "*"
        return self.name

    @property
    def path(self):
        global CLASSIFIER_PATH
        return CLASSIFIER_PATH / f"{self.name}_classifier.pickle"


    def add_class(self, cls):
        """
        Add a new class to this classifier if possible
        Parameters
        ----------
        cls : the class to add

        Returns
        -------
        
        """
        if not self.writable:
            return
        self.class_set.add(cls)
        # if hasattr(self.gbc, "classes_"):
        #     self.gbc.classes_ = np.append(self.gbc.classes_, cls)
        #     # self.gbc.classes_.append(cls)
        # else:
        #     self.gbc.classes_ = np.array([cls])
        self.changed = True
    
    def remove_class(self, cls):
        """
        Remove of a class from this classifier.
        Parameters
        ----------
        cls : The class to remove

        Returns
        -------
        
        """
        if not self.writable:
            return
        self.class_set.discard(cls)
        # if hasattr(self.gbc, "classes_"):
        #     self.gbc.classes_ = self.gbc.classes_[self.gbc.classes_ != cls]
        #     # np.delete(self.gbc.classes_,)
        #     # self.gbc.classes_.remove(cls)
        # else:
        #     self.gbc.classes_ = np.array([])
        self.changed = True

    @property
    def classes(self):
        # if hasattr(self.gbc, "classes_"):
        #     classes = set(self.gbc.classes_)
        #     available_classes = get_all_shapes()

        #     used_classes = classes.intersection(available_classes)
        #     unused_classes = available_classes.difference(classes)
        #     removed_classes = classes.difference(available_classes)
            
        #     return used_classes, unused_classes, removed_classes
        # else:
        classes = self.class_set
        available_classes = get_all_shapes()

        used_classes = classes.intersection(available_classes)
        unused_classes = available_classes.difference(classes)
        removed_classes = classes.difference(available_classes)
        
        return used_classes, unused_classes, removed_classes
        # return set(), get_all_shapes(), set()

    @property
    def writable(self):
        return self.name != "Default_GBC" and self.name != "Default_NN"

def ShapeClassifierFactory(filepath=None, name=None, to_copy:ShapeClassifier =None, classifier=None):
    """
    Loads or creates a new shape classifier object.
    Parameters
    ----------
    filepath     : The filepath to the classifier to load.
    name         : The name of the new classifier.
    to_copy      : A classifier to copy
    classifier   : A classifier which just gets returned

    Returns
    -------
    inst         : A classifier object
    """
    if all([a is None for a in [filepath, name, to_copy, classifier]]):
        raise ValueError("Something shouldnt be None here.")
    if filepath is not None:
        with open(filepath, "rb") as f:
            inst = CustomUnpickler(f).load()
            if not hasattr(inst, "type_"):
                setattr(inst, "type_", "GradientBoostingClassifier")
            if not hasattr(inst, "gbc_"):
                setattr(inst, "gbc_", GradientBoostingClassifier())
            if not hasattr(inst, "class_set"):
                setattr(inst, "class_set", set())
                if hasattr(inst.gbc, "classes_"):
                    setattr(inst, "class_set", set(inst.gbc.classes_))
            if not hasattr(inst, "only_closed_"):
                setattr(inst, "only_closed_", True)
    

    elif name is not None:
        if name in get_all_classifier_names():
            raise FileExistsError(name)
        inst = ShapeClassifier(name=name)
    elif to_copy is not None:
        inst = to_copy.create_copy()
    elif classifier is not None:
        return classifier
    return inst



def createNeuralNetworkClassifierWithoutDevice(classes=["circle", "not_circle"], num_features=100):
    """
    Creates a new neural network without using "device
    """
    
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu', input_shape=(num_features,1)))
    model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
    model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(len(classes), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def createNeuralNetworkClassifier(classes=["circle", "not_circle"], num_features=100):
    """
    Creates a new neural network
    """
    cpus = list_logical_devices('CPU')
    with device(cpus[0]) as d:
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu', input_shape=(num_features,1)))
        model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
        model.add(Conv1D(filters=64, kernel_size=7,padding="same", activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(len(classes), activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    pass
    