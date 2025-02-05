# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import warnings

warnings.filterwarnings('ignore')

import numpy as np
from matplotlib import pyplot as plt

from PIL import Image, ImageOps
import sparse
import multiprocessing as mp

from scipy.ndimage import label, binary_dilation
from skimage.morphology import skeletonize
# import tensorflow as tf
from keras.utils import Sequence
import mrcfile
# import glob
from pathlib import Path
# import random
from PIL import Image
import shutil
from scipy.spatial.distance import cdist
from skimage.draw import disk
import cv2
import getpass
import math
# from cryovia.cryovia_analysis.skeletonizer import Skeletonizer
#####################################################################################################################################




def shuffle_train_data(X_train, Y_train, loss_weights=None, random_seed=1):
    """
    Shuffles data with seed 1.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train : array(float)
        shuffled array of training images.
    Y_train : array(float)
        Shuffled array of labelled training images.
    """
    np.random.seed(random_seed)
    seed_ind = np.random.permutation(X_train.shape[0])
    X_train = X_train[seed_ind]
    Y_train = Y_train[seed_ind]
    if loss_weights is not None:
        loss_weights = loss_weights[seed_ind]
        return X_train, Y_train, loss_weights

    return X_train, Y_train



def augment_data(X_train, Y_train,loss_weights=None, flip=True):
    """
    Augments the data 8-fold by 90 degree rotations and flipping.

    Parameters
    ----------
    X_train : array(float)
        Array of source images.
    Y_train : float
        Array of label images.
    Returns
    -------
    X_train_aug : array(float)
        Augmented array of training images.
    Y_train_aug : array(float)
        Augmented array of labelled training images.
    """
    X_ = X_train.copy()

    X_train_aug = np.concatenate((X_train, np.rot90(X_, 1, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 2, (1, 2))))
    X_train_aug = np.concatenate((X_train_aug, np.rot90(X_, 3, (1, 2))))

    

    Y_ = Y_train.copy()
    Y_train_aug = np.concatenate((Y_train, np.rot90(Y_, 1, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 2, (1, 2))))
    Y_train_aug = np.concatenate((Y_train_aug, np.rot90(Y_, 3, (1, 2))))
    if flip:
        X_train_aug = np.concatenate((X_train_aug, np.flip(X_train_aug, axis=1)))
        Y_train_aug = np.concatenate((Y_train_aug, np.flip(Y_train_aug, axis=1)))

    if loss_weights is not None:
        loss_weights_ = loss_weights.copy()

        loss_train_aug = np.concatenate((loss_weights, np.rot90(loss_weights_, 1, (1, 2))))
        loss_train_aug = np.concatenate((loss_train_aug, np.rot90(loss_weights_, 2, (1, 2))))
        loss_train_aug = np.concatenate((loss_train_aug, np.rot90(loss_weights_, 3, (1, 2))))
        loss_train_aug = np.concatenate((loss_train_aug, np.flip(loss_train_aug, axis=1)))
        return X_train_aug, Y_train_aug, loss_train_aug


    return X_train_aug, Y_train_aug

def convert_to_oneHot(data, eps=1e-8, classes=3):
    """
    Converts labelled images (`data`) to one-hot encoding.

    Parameters
    ----------
    data : array(int)
        Array of lablelled images.
    Returns
    -------
    data_oneHot : array(int)
        Array of one-hot encoded images.
    """
    data_oneHot = np.zeros((*data.shape, classes), dtype=np.float32)
    for i in range(data.shape[0]):
        data_oneHot[i] = onehot_encoding(data[i].astype(np.int32),n_classes=classes)
        if ( np.abs(np.max(data[i])) <= eps ):
            data_oneHot[i][...,0] *= 0

    return data_oneHot

def onehot_encoding(lbl, n_classes=3, dtype=np.uint32):
    """
    Creates oneHot enconding of the given classes
    Parameters
    ----------
    lbl       : list of classes to encode
    n_classes : number of available classes
    dtype     : dtype of oneHot encoding

    Returns
    -------
    oneHot    : the oneHot encoding
    """
    onehot = np.zeros((*lbl.shape, n_classes), dtype=dtype)
    for i in range(n_classes):
        onehot[lbl == i, ..., i] = 1
    return onehot






def load_npz_files(paths, image_size, step, skeleton=False, dilution=None, is_valid=None, valid_perc=0.2):
    """
    Loads in the segmentation of npz files in patches
    Parameters
    ----------
    paths       : the paths to load in
    image_size  : the size of the patches
    step        : the step size of the patches
    skeleton    : bool, whether to convert the segmentation into a skeleton beforehand
    dilution    : None for no dilution, int for int times dilution
    is_valid    : Redundant, do not use
    valid_perc    : Redundant, do not use

    Returns
    -------
    patches     : the extracted patches
    shapes      : the shape of the patches
    """
    
    all_patches = []
    shapes = []

    if type(paths) != list:
        paths = [paths]
    for path_counter, path in enumerate(paths):
        patches = []

        mrc_data = sparse.load_npz(path).todense()
        
        
        
        shapes.append(mrc_data.shape[:2])
        
            # mrc_data = mrc_data / np.max(np.abs(mrc_data))
            # 
            
        if is_valid is not None:
            size = mrc_data[:,:,0].size
            size = np.sqrt(size*valid_perc).astype(np.int32)

        if skeleton:
            
            new_mrc_data = np.zeros(mrc_data.shape[:2])
            for i in range(mrc_data.shape[-1]):
                skel = skeletonize(mrc_data[...,i])
                if dilution is not None:
                    skel = binary_dilation(skel, np.ones((3,3)), dilution)
                new_mrc_data += skel
            mrc_data = (new_mrc_data > 0) * 1


        starting_points = set()
        for i in range(0,mrc_data.shape[0],step):
            for j in range(0,mrc_data.shape[1],step):
                if i + image_size > mrc_data.shape[0]:
                    i = mrc_data.shape[0] - image_size
                if j + image_size > mrc_data.shape[1]:
                    j = mrc_data.shape[1] - image_size
                if (i,j) in starting_points:
                    continue
                if is_valid is not None:
                    if is_valid:
                        if i < mrc_data.shape[0] - size or j < mrc_data.shape[1] - size:
                            continue
                    else:
                        if i > mrc_data.shape[0] - size and j > mrc_data.shape[1] - size:
                            continue

                
                patches.append(mrc_data[i:i+image_size, j:j+image_size])
        patches = np.array(patches)
        
        all_patches.append(patches )
    all_patches = np.concatenate(all_patches).astype(np.int8)
    return all_patches, shapes




def load_mrc(file):
    
    with mrcfile.open(file, permissive=True) as f:
        data = f.data * 1
        pixel_size = f.voxel_size["x"]
    return data, pixel_size

def load_jpg_png(file):
    img = ImageOps.grayscale(Image.open(file))
    return np.array(img)

def load_npz(file):
    data = sparse.load_npz(file).todense()
    
    if data.ndim == 3:
        lowest_axes = np.argmin(data.shape)
        data = np.moveaxis(data, lowest_axes, [0])
    return data


def load_file(file):
    suffix_function_dict = {
        ".mrc":load_mrc,
        ".png":load_jpg_png,
        ".jpg":load_jpg_png,
        ".jpeg":load_jpg_png,
        ".npz":load_npz,
        ".rec":load_mrc
    }
    suffix = Path(file).suffix.lower()
    if suffix in suffix_function_dict:
        data = suffix_function_dict[suffix](file)
        if not isinstance(data, tuple):
            data = (data, None)
        return data 
    else:
        raise ValueError(f"Could not find a function to open a {suffix} file.")



def solveLayer(layer):
    """
    Extracts all connected components of an image and returns a stack of images, of image for each component
    Parameters
    ----------
    layer   : the layer to solve

    Returns
    -------
    stack   : the stack of the solved layer
    """
    unique_labels = np.unique(layer)
    if len(unique_labels) > 2:
        unique_labels = unique_labels[1:]
        new_stack = []
        for ul in unique_labels:
            new_layer = (layer == ul)*1
            new_stack.append(solveLayer(new_layer))
        return np.concatenate(new_stack)
    labels, num_features = label(layer, np.ones((3,3), dtype=np.int8),)
    if num_features == 1:
        return np.expand_dims(labels,0)
    if num_features == 0:
        return np.expand_dims(labels,0)
    new_stack = []
    for label_counter in range(1,num_features + 1):
        label_image = (labels == label_counter) * 1
        new_stack.append(label_image)
    return np.array(new_stack)


def create_3d_stack(data):
    """
    Solves all layers for the data, by looking for connected components
    Parameters
    ----------
    data    : the data to solve

    Returns
    -------
    new_stack : the solved data stack
    """
    if data.ndim == 2:
        layers = solveLayer(data)
        return layers
    
    new_stack = []
    for layer in data:
        new_stack.append(solveLayer(layer))
    new_stack = np.concatenate(new_stack)
    return new_stack





def fft_rescale_image(image, new_shape):
    """
    Rescale a grayscale image to a new shape using Fourier cropping or padding.

    Parameters:
    - image: 2D numpy array representing the grayscale image.
    - new_shape: Tuple (new_height, new_width) representing the desired output shape.

    Returns:
    - Rescaled image as a 2D numpy array.
    """
    # Perform Fourier Transform
    f_transform = np.fft.fft2(image)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Get the center of the original Fourier transform
    original_shape = image.shape
    center_y, center_x = original_shape[0] // 2, original_shape[1] // 2

    # Determine new cropping/padding sizes
    new_height, new_width = new_shape
    half_new_height, half_new_width = new_height // 2, new_width // 2

    # Create an empty Fourier domain with the new shape
    resized_f_transform = np.zeros(new_shape, dtype=complex)

    # Define bounds for cropping/padding in the Fourier domain
    y1 = max(0, center_y - half_new_height)
    y2 = min(original_shape[0], center_y + half_new_height + new_height % 2)
    x1 = max(0, center_x - half_new_width)
    x2 = min(original_shape[1], center_x + half_new_width + new_width % 2)

    y1_new = max(0, half_new_height - center_y)
    y2_new = y1_new + (y2 - y1)
    x1_new = max(0, half_new_width - center_x)
    x2_new = x1_new + (x2 - x1)

    # Copy the relevant portion of the Fourier transform or pad with zeros
    resized_f_transform[y1_new:y2_new, x1_new:x2_new] = f_transform_shifted[y1:y2, x1:x2]

    # Perform inverse Fourier transform
    resized_f_transform = np.fft.ifftshift(resized_f_transform)
    rescaled_image = np.fft.ifft2(resized_f_transform).real

    # Normalize the image to the original range
    return rescaled_image



# def fft_rescale_image(image, new_size):
#     """
#     Rescales an image using cropping in fourier space 
#     Parameters
#     ----------
#     image       : the image to rescale
#     new_size    : the size to rescale the image to

#     Returns
#     -------
#     rescaled_image : the rescaled image
#     """
    
#     old_size = np.array(image.shape)
    
#     rescale_factor = old_size[0] / new_size[0]
    
#     fft_image = np.fft.fftshift(np.fft.fft2(image))
#     if rescale_factor > 1:
#         difference = (old_size - new_size)
#         to_add_x = 0
#         to_add_y = 0
#         if difference[0] % 2 == 1:
#             to_add_y = -1
#         if difference[1] % 2 == 1:
#             to_add_x = -1

#         difference = (difference/2).astype(np.int32)
#         fft_image = fft_image[difference[0] - to_add_y:-difference[0], difference[1]- to_add_x:-difference[1] ]

#     else:
#         difference = (new_size - old_size)
#         to_add_x = 0
#         to_add_y = 0
#         if difference[0] % 2 == 1:
#             to_add_y = 1
#         if difference[1] % 2 == 1:
#             to_add_x = 1
#         difference = (difference/2).astype(np.int32)
#         pad = [(difference[0], difference[0] + to_add_y),(difference[1], difference[1]+ to_add_x)]
#         fft_image = np.pad(fft_image,pad, mode="constant", constant_values=0)
    
#     rescaled_image = np.real(np.fft.ifft2(np.fft.ifftshift(fft_image)))
#     return rescaled_image


def preprocess(data):
    return data




def createDilatedSkeleton(layer, dilation=None):   
    """
    Creates a dilation of a skeleton of a given segmentation
    Parameters
    ----------
    layer       : a 2d segmentation image
    dilation    : None for no dilation, int for in times dilation
    Returns
    -------
    new_mrc_data: the dilated skeleton
    """
    new_mrc_data = skeletonize(layer)
    if dilation is not None and dilation > 0:
        new_mrc_data = binary_dilation(new_mrc_data, np.ones((3,3)),dilation)

    return new_mrc_data

import sparse
from skimage.transform import resize

def resizeSegmentation(image, shape):
    """
    Resizes a given segmentation
    Parameters
    ----------
    image   : the given segmentation
    shape   : the shape to resize to
    Returns
    -------
    resized_segmentation : the resized segmentation
    """
    
    if isinstance(image, np.ndarray):
        return resize(image, shape,0)
    return resize(image.todense(), shape, 0)
    

def load_segmentation_files(paths, config,stepSize=None, get_shapes=False, seg_pix=None):
    """
    Loads in a list of segmentation files
    Parameters
    ----------
    paths       : the list of paths to load in
    config      : a dictionary with configs for the loading
    stepSize    : the step size for the segmentation patches
    get_shapes  : bool, whether to get also the shapes returned
    seg_pix     : a list of pixel sizes the segmentation should be loaded in
    Returns
    -------
    patches     : the loaded segmentation patches
    shapes      : (optional) the shapes of the patches
    """


    all_patches = []
    shapes = []
    image_size = config.input_shape
    if stepSize is None:
        step = image_size // 2
    else:
        step = stepSize

    if seg_pix is None:
        seg_pix = [None for _ in paths]
    if type(paths) != list:
        paths = [paths]
        
    for path_counter, path in enumerate(paths):
        patches = []
        stack_data,_ = load_file(path)

        stack_data = create_3d_stack(stack_data)


        
        if seg_pix[path_counter] is not None:
            ps = config.pixel_size
            ratio = seg_pix[path_counter] / ps
            new_shape = [int(s*ratio) for s in stack_data.shape[1:]]

            stack_data = np.array([resizeSegmentation(layer, new_shape) for layer in stack_data])
        
        if config.thin_segmentation:
            
            stack_data = np.array([createDilatedSkeleton(layer, config.dilation) for layer in stack_data])
        
        stack_data = (np.sum(stack_data,0) > 0) * 1
        shapes.append(stack_data.shape)

        for i in range(0,stack_data.shape[0],step):
            for j in range(0,stack_data.shape[1],step):
                if i + image_size > stack_data.shape[0]:
                    i = stack_data.shape[0] - image_size
                if j + image_size > stack_data.shape[1]:
                    j = stack_data.shape[1] - image_size
                
                patches.append(stack_data[i:i+image_size, j:j+image_size])
        patches = np.array(patches)
        
        
        all_patches.append(patches )
    all_patches = np.concatenate(all_patches)
    if get_shapes:
        return all_patches, shapes
    return all_patches



def predictMiddle(segmentation, confidence, distance=10, njobs=1, path=None):
    if segmentation.ndim == 2:
        lab, features = label(segmentation)
        segmentation = np.array([(lab == i) * 1 for i in range(1, features)])
    skeletonizer = Skeletonizer(sparse.as_coo( segmentation), 1, njobs) 
    skeleton_points = skeletonizer.find_skeletons() 
    y,x = disk((0,0), distance / 2,)

    for counter, skeleton in enumerate(skeleton_points):
        conf = np.copy(confidence)

        is_circle = cdist(skeleton[:1], skeleton[-1:],metric="euclidean")[0][0] < 1.5

        corrected_points = []
        corrected_skeleton = np.zeros_like(conf, dtype=np.uint8)
        for point in skeleton:
            if np.isnan(conf[point[0], point[1]]):
                continue 
            current_y, current_x = y + point[0], x + point[1]
            idxs = np.where((current_x >= 0) & (current_y >= 0) & (current_x < conf.shape[1]) & (current_y < conf.shape[0]))
            current_x = current_x[idxs]
            current_y = current_y[idxs]
            
            surrouding_confidence = conf[current_y, current_x]
            best_conf = np.nanargmax(surrouding_confidence)
            new_point = (current_y[best_conf], current_x[best_conf])
            corrected_points.append(new_point)
            conf[current_y, current_x] = np.nan
            conf[new_point[0] + y, new_point[1] + x] = np.nan
            corrected_skeleton[new_point[0], new_point[1]] = 1


#     mrc_data = mrcfile.open(path, permissive=True).data * 1
#     mrc_data = fft_rescale_image(mrc_data, segmentation[0].shape)

        



        





def unpatchify(patches,image_shape, config, threshold=True, both=False,):
    """
    Creates whole images from patches
    Parameters
    ----------
    patches     : the patches to unpatchify
    image_shape : the shape of the resulting image
    config      : the config of the segmentation model
    threshold   : bool, whether to return only the argmax values of the segmentation
    both        : bool, whether to return argmax and probabilities
    Returns
    -------
    argmax_prediction: (optional) the argmax prediction of the unpatchified segmentation
    prediction  : of the unpatchified segmentation prediction
    
    """
    
    step = config.input_shape // 2
    

    image = np.zeros((*image_shape,2))
    image_size = patches[0].shape[0]

    starting_points = set()
    counter = 0
    for i in range(0,image_shape[0],step):
        for j in range(0,image_shape[1],step):
            if i + image_size > image_shape[0]:
                i = image_shape[0]  - image_size
            if j + image_size > image_shape[1]:
                j = image_shape[1]  - image_size
            if (i,j) in starting_points:
                continue

            image[i:i+image_size, j:j+image_size] += patches[counter]
            counter += 1
            

    if both:

        prediction = np.argmax(image, -1)
        confidence = image[..., 1] / np.sum(image, -1)
        return prediction, confidence
    if threshold:
        image = np.argmax(image, -1)
    else:

        image = image[..., 1] / np.sum(image, -1)


    return image







def getTrainingData(micrographs, segmentations, config, print_func=None):
    """
    Load in the patches of micrographs and segmentation for training
    Parameters
    ----------
    micrographs     : list of paths of micrographs
    segmentations   : list of paths of segmentations
    config          : config of the segmentation model
    print_func      : a method to print some output
    Returns
    -------
    micrograph_patches : patches of the micrograph images
    segmentation_patches : patches of the segmentations
    
    """

    x_train = load_micrographs_files(micrographs, config)
    y_train = load_segmentation_files(segmentations, config)


    has_membrane_counter = 0
    has_no_membrane_counter = 0
    membrane_idxs = []
    for counter, img in enumerate(y_train):
        if np.max(img) > 0:
            has_membrane_counter += 1
            membrane_idxs.append(counter)
        else:
            has_no_membrane_counter += 1

    
    y_train = y_train[membrane_idxs]

    x_train = x_train[membrane_idxs]
    
    y_train[y_train > 0] = 1

    x_train, y_train = shuffle_train_data(x_train, y_train)


    
    y_train = onehot_encoding(y_train,2).astype(np.int8)

    x_train = x_train[...,np.newaxis]

    if print_func is not None:
        print_func(f"Found {len(x_train)} patches of size {x_train.shape[1]}x{x_train.shape[2]} with membranes for training.")
    x_train, y_train = augment_data(x_train, y_train)

    if print_func is not None:
        print_func(f"After augmention: {len(x_train)} patches of size {x_train.shape[1]}x{x_train.shape[2]}")
    x_train, y_train = shuffle_train_data(x_train, y_train)

    y_train = y_train.astype(np.float32)
    
    

    return x_train, y_train

from cryovia.cryovia_analysis.analyser import Analyser

class customDatasetForPerformance(Sequence):

    def __init__(self, micrographs, segmentations, batch_size, config, path, name="Train", toStop=None,njobs=1, useAll=False, shuffle=True, stepSize=None, flip=True,seg_pixel_sizes=None, image_pixel_sizes=None):
        self.micrographs = micrographs
        self.segmentations = segmentations
        self.batch_size = batch_size
        self.config = config
        self.toStop = toStop

        kwargs = {
            "pixel_size":7,
            "min_size": 0,
            "max_hole_size": 0,
            "micrograph_pixel_size": None,
            "step_size":1
            }
        
        path = Path(path)
        dir_counter = 0
        while True:
            current_dir = path / f"{name}_{dir_counter}"
            if not current_dir.exists():
                current_dir.mkdir(parents=True)
                self.path = current_dir
                break
            dir_counter+= 1
        


        
        self.number_of_files = 0
        self.current_augment_counter = 0
        self.numberOfFilesPerMicrograph = []
        with mp.get_context("spawn").Pool(njobs) as pool:
            for counter, (seg, micro) in enumerate(zip(self.segmentations, self.micrographs)):
                if config.only_use_improved and not config.filled_segmentation:
                    a = Analyser(micro, "Training", seg, pool=pool, name=str(counter),**kwargs)
                    
                    segmentation_patches, micro_patches = a.createImprovedSegmentation(self.config, stepSize=stepSize)
                else:
                    if seg_pixel_sizes is not None and seg in seg_pixel_sizes:
                        seg_pix = seg_pixel_sizes[seg]
                    else:
                        seg_pix = None
                    if image_pixel_sizes is not None and micro in image_pixel_sizes:
                        img_pix = image_pixel_sizes[micro]
                    else:
                        img_pix = None
                    segmentation_patches, shapes = load_segmentation_files([seg], self.config, stepSize=stepSize, seg_pix=[seg_pix], get_shapes=True)
                    micro_patches = load_micrographs_files([micro], self.config,stepSize=stepSize, seg_shapes=shapes, given_pixel_size=img_pix)
                if not useAll:
                    has_membrane_counter = 0
                    has_no_membrane_counter = 0
                    membrane_idxs = []
                    for counter, img in enumerate(segmentation_patches):
                        if np.max(img) > 0:
                            has_membrane_counter += 1
                            membrane_idxs.append(counter)
                        else:
                            has_no_membrane_counter += 1
                    if len(membrane_idxs) == 0:
                        continue
                
                    y_train = segmentation_patches[membrane_idxs]
                    x_train = micro_patches[membrane_idxs]
                else:
                    y_train = segmentation_patches
                    x_train = micro_patches
                if len(x_train) == 0:
                    continue
                x_train = self.normalize(x_train)
                self.numberOfFilesPerMicrograph.append(len(x_train))
                x_train, y_train = augment_data(x_train, y_train, flip=flip)
                for x,y in zip(x_train, y_train):
                    mrcfile.new(self.path / f"{self.number_of_files}.mrc", x.astype(np.float32), overwrite=True)
                    sparse.save_npz(self.path / f"{self.number_of_files}.npz", sparse.as_coo(y.astype(np.uint8)))
                    self.number_of_files += 1
                    if self.toStop is not None:
                        if self.toStop():   
                            break
                if self.toStop is not None and self.toStop():
                    self.clean()
                    break
        
        self.idxs = np.arange(self.number_of_files)
        if shuffle:
            np.random.shuffle(self.idxs)

    def __getitem__(self, index):
        x_train = []
        y_train = []
        index = index * self.batch_size
        for i in range(index, min(index + self.batch_size, len(self.idxs))):
            x_train.append(mrcfile.open(self.path / f"{self.idxs[i]}.mrc", permissive=True).data * 1)
            y_train.append(sparse.load_npz(self.path / f"{self.idxs[i]}.npz").todense())
        
        y_train = np.array(y_train)
        x_train = np.array(x_train)
        y_train = onehot_encoding(y_train,2).astype(np.float32)

        x_train = x_train[...,np.newaxis]

        return x_train, y_train
           
    
    def on_epoch_end(self):
        np.random.shuffle(self.idxs)

    def normalize(self, x):
        means = [np.mean(img) for img in x]
        stds = [np.std(img) for img in x]
        
        x = np.array([(img - mean) / std for img, mean, std in zip(x, means, stds)])
        return x       
                     

    def clean(self):
        if self.path.exists():
            shutil.rmtree(self.path)

    def __len__(self):
        return math.ceil(self.number_of_files  / self.batch_size)




def getTrainingDataForPerformance(micrographs, segmentations, config, print_func=None, validation_start=0, validation_end=0.5,seed=1, batch_size=8, path=".", toStop=None, njobs=1,seg_pixel_sizes=None,image_pixel_sizes=None):
    """
    Load in the patches of micrographs and segmentation for training and return a Dataset. For performance in name only because of legacy reasons.
    Parameters
    ----------
    micrographs     : list of paths of micrographs
    segmentations   : list of paths of segmentations
    config          : config of the segmentation model
    print_func      : a method to print some output
    validation_start: percentage of when the validation index starts
    validation_end  : percentage of when the validation index ends
    seed            : seed for shuffeling
    batch_size      : batch size for model training
    path            : where to store the temp files
    toStop          : event to stop the process
    njobs           : number of parallel processes to create training files
    seg_pixel_sizes : list of pixel sizes for segmentation
    image_pixel_sizes : list of pixel sizes for micrographs
    Returns
    -------
    train_dataset   : tensorflow dataset for loading in train data
    valid_dataset   : tensorflow dataset for loading in validation data
    """


    idxs = np.arange(len(micrographs))
    np.random.seed(seed)
    np.random.shuffle(idxs)
    
    micrographs = [micrographs[idx] for idx in idxs]
    segmentations = [segmentations[idx] for idx in idxs]

    number_of_files = len(micrographs)
    # if seg_pixel_sizes is not None:
    #     seg_pixel_sizes = np.array(seg_pixel_sizes)
    valid_start = int(number_of_files * validation_start)
    valid_end = int(number_of_files * validation_end)
    if np.abs(valid_end - valid_start) == 0:
        valid_end = valid_start + 1
    valid_micrographs = micrographs[valid_start:valid_end ]
    valid_segmentations = segmentations[valid_start:valid_end]
    

    train_micrographs = micrographs[0:valid_start]
    train_micrographs.extend(micrographs[valid_end:])

    train_segmentations = segmentations[0:valid_start]
    train_segmentations.extend(segmentations[valid_end:])

    if seg_pixel_sizes is not None:
        train_seg_pixel_sizes = {i:seg_pixel_sizes[i] for i in train_segmentations}
        valid_seg_pixel_sizes = {i:seg_pixel_sizes[i] for i in valid_segmentations}
        # train_seg_pixel_sizes = seg_pixel_sizes[0:valid_start]
        # train_seg_pixel_sizes.extend(seg_pixel_sizes[valid_end:])
        # valid_seg_pixel_sizes = seg_pixel_sizes[valid_start:valid_end]
        # train_seg_pixel_sizes = seg_pixel_sizes[0:int(number_of_files * validation_start)]
        # train_seg_pixel_sizes.extend(train_seg_pixel_sizes[int(number_of_files * validation_end):])
        # valid_seg_pixel_sizes = seg_pixel_sizes[int(number_of_files * validation_start):int(number_of_files * validation_end)]
    else:
        train_seg_pixel_sizes = None
        valid_seg_pixel_sizes = None




    # train_segmentations = segmentations[0:int(number_of_files * validation_start)]
    # train_segmentations.extend(segmentations[int(number_of_files * validation_end):])

    
    train_dataset = customDatasetForPerformance(train_micrographs, train_segmentations, batch_size, config, path, "Train", toStop=toStop,njobs=njobs, seg_pixel_sizes=train_seg_pixel_sizes, image_pixel_sizes=image_pixel_sizes, useAll=True)


    valid_dataset = customDatasetForPerformance(valid_micrographs, valid_segmentations, batch_size, config, path, "Valid", toStop=toStop, seg_pixel_sizes=valid_seg_pixel_sizes, image_pixel_sizes=image_pixel_sizes, useAll=True)


    return train_dataset, valid_dataset



def loadPredictData(micrographs, config,ps=None):
    """
    Load in micrgraphs data for prediction, loads patches
    Parameters
    ----------
    micrographs     : list of paths of micrographs
    config          : config of the segmentation model
    ps              : pixel size to resize to
    Returns
    -------
    micrograph_patches : patches of the micrograph images
    shapes             : shapes of the patches
    """
    

    x_train, shapes = load_micrographs_files(micrographs, config, per_file=True, get_shapes=True, given_pixel_size=ps)
    x_train = [x_t[...,np.newaxis] for x_t in x_train]
    return x_train, shapes



def save_file(filename:Path, data, pixel_size):
    """
    Saves the given file with the given pixel size
    Parameters
    ----------
    filename    : Path where to save
    data        : Data which to save
    pixel_size  : pixel size to set for mrc files
    Returns
    -------
    micrograph_patches : patches of the micrograph images
    segmentation_patches : patches of the segmentations
    
    """
    data = data.astype(np.uint8)
    filename = Path(filename)
    if filename.suffix == ".mrc":
        with mrcfile.new(filename, overwrite=True) as f:
            f.set_data(data)
            f.voxel_size = pixel_size
    elif filename.suffix == ".npz":
        data = sparse.as_coo(data)
        sparse.save_npz(filename, data)
    elif filename.suffix in [".png", "jpg", "jpeg"]:
        if data.dtype in [np.uint8, np.int8, np.int32]:

            if data.ndim == 3:
                data = np.sum(data, 0)
            
            data = (data > 0) * 255
            data = data.astype(np.uint8)

            
        Image.fromarray(data).save(filename,)
    else:
        raise NotImplementedError(f"save_file not implemented for {filename.suffix}.")


def get_contour_of_filled_segmentation(segmentation):
    """
    Extracts the contour of a segmentation of filled vesciles
    Parameters
    ----------
    segmentation    : segmentation of which to extract contours
    
    Returns
    -------
    contour_iamge   : the segmentation image of the contour
    
    """
    
    contour_image = np.zeros_like(segmentation)
    contours, hierarchy = cv2.findContours(segmentation.astype(np.uint8),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        contour = np.squeeze(contour)
        if contour.ndim == 1:
            continue
        for (x,y) in contour:
            contour_image[y,x] = 1
    return contour_image


def gauss(fx,fy,sig):
    """
    gaussian kernel
    """

    r = np.fft.fftshift(np.sqrt(fx**2 + fy**2))
    
    return np.exp(-2*np.pi**2*(r*sig)**2)

def gaussian_filter(im,sig,apix):
    """
    gaussian filter to high and low pass filtering
    Parameters 
    -----------
        sig (real space) and apix in angstrom
    """
    sig = sig/2/np.pi
    fx,fy = np.meshgrid(np.fft.fftfreq(im.shape[1],apix),\
                        np.fft.fftfreq(im.shape[0],apix))

    im_fft = np.fft.fftshift(np.fft.fft2(im))
    fil = gauss(fx,fy,sig*apix)
    
    im_fft_filtered = im_fft*fil
    newim = np.real(np.fft.ifft2(np.fft.ifftshift(im_fft_filtered)))
    
    return newim


def load_micrographs_files(paths, config, prep=False, per_file=False, get_shapes=False, resize=True, given_pixel_size=None,stepSize=None, seg_shapes=None ):
    """
    Loads in a list of micrograph files
    Parameters
    ----------
    paths       : the list of paths to load in
    config      : a dictionary with configs for the loading
    prep        : whether to do preprocessing, redundant
    per_file    : bool, whether to create nested lists for each file
    stepSize    : the step size for the segmentation patches
    get_shapes  : bool, whether to get also the shapes returned
    resize      : bool, whether to resize the data
    given_pixel_size : pixel_size in case the file does not have one yet
    seg_shapes  : the shapes of the corresponding segmentations
    Returns
    -------
    patches     : the loaded segmentation patches
    shapes      : (optional) the shapes of the patches
    """
    all_patches = []
    image_size = config.input_shape
    
    if stepSize is None:
        step = image_size // 2
    else:
        step = stepSize
    shapes = []
    if type(paths) != list:
        paths = [paths]
    for path_counter, path in enumerate(paths):
        patches = []
        mrc_data, pixel_size = load_file(path)
        if pixel_size is None:
            pixel_size = given_pixel_size
        if pixel_size is None:
            raise ValueError(f"Cannot resize data from {path} because no pixel size was given and it could not be extracted from an mrc file.")

        if config.std_clip > 0:
            mean = np.mean(mrc_data)
            std = np.std(mrc_data)

            mrc_data = np.clip(mrc_data, mean- config.std_clip*std,mean + config.std_clip*std)

        if config.high_pass_filter > 0:
            sig = int(config.high_pass_filter / pixel_size)
            sig += (sig + 1) % 2
            mrc_data = gaussian_filter(mrc_data,0,pixel_size) - gaussian_filter(mrc_data,sig,pixel_size)


        if prep is not None:
            mrc_data = preprocess(mrc_data)
        if resize:
            
            
            
            if not np.isclose(config.pixel_size, pixel_size) or (seg_shapes is not None and not all((s == ss for s,ss in zip(seg_shapes[path_counter], mrc_data.shape)))):
                # shape = [int(s * pixel_size / config.pixel_size) for s in mrc_data.shape[::-1]]
                shape = [int(s * pixel_size / config.pixel_size) for s in mrc_data.shape]
                # mrc_data = cv2.resize(mrc_data, shape,interpolation=cv2.INTER_CUBIC )
  
                mrc_data = fft_rescale_image(mrc_data, shape)
        for i in range(0,mrc_data.shape[0],step):
            for j in range(0,mrc_data.shape[1],step):
                if i + image_size > mrc_data.shape[0]:
                    i = mrc_data.shape[0] - image_size
                if j + image_size > mrc_data.shape[1]:
                    j = mrc_data.shape[1] - image_size
                
                patches.append(mrc_data[i:i+image_size, j:j+image_size])
        patches = np.array(patches)
        
        shapes.append(mrc_data.shape)
        all_patches.append(patches )
    if per_file:
        all_patches = all_patches
    else:
        all_patches = np.concatenate(all_patches)
    if get_shapes:
        return all_patches, shapes
    return all_patches



def patchify(data, config,stepSize=None):
    """
    Creates patches from data
    Parameters
    ----------
    data        : the data to create patches from
    config      : a dictionary with configs for the loading
    stepSize    : the step size for the segmentation patches
    
    Returns
    -------
    patches     : the loaded segmentation patches
    """
   
    image_size = config.input_shape
    if stepSize is None:
        step = image_size // 2
    else:
        step = stepSize
    patches = []
   
    for i in range(0,data.shape[0],step):
        for j in range(0,data.shape[1],step):
            if i + image_size > data.shape[0]:
                i = data.shape[0] - image_size
            if j + image_size > data.shape[1]:
                j = data.shape[1] - image_size
            
            patches.append(data[i:i+image_size, j:j+image_size])
    patches = np.array(patches)
        
      
    return patches





if __name__ == "__main__":
    pass
