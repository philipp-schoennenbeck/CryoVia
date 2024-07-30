
from pathlib import Path
import mrcfile
import numpy as np
import sparse
import pandas as pd
import cv2
import multiprocessing as mp
import pickle
from scipy.fft import fft2, ifft2

import shutil
from scipy.ndimage import label
from scipy.spatial.distance import cdist
from skimage.morphology import skeletonize
from scipy import optimize
from scipy import signal
from scipy.interpolate import splrep, BSpline

from scipy.ndimage import label, binary_dilation


import json
from PIL import Image
import shutil

from cryovia.cryovia_analysis.skeletonizer import Skeletonizer, RidgeDetector
from cryovia.cryovia_analysis.membrane import Membrane
from cryovia.cryovia_analysis.point import Point
import cryovia.cryovia_analysis.custom_utils as utils 

from cryovia.gui.Unpickler import CustomUnpickler

from collections import defaultdict

from matplotlib import pyplot as plt

def mkdir(path:Path):
    if not path.exists() or not path.is_dir():
        path.mkdir(parents=True)


def warning(msg):
    print("Warning: " + msg)


MRC_SUFFIX = set([".mrc", ".rec"])


class curvatureAnalyser:
    def __init__(self, segmentation_image):
        self.segmentation_stack = sparse.as_coo(np.expand_dims(segmentation_image, 0).astype(np.int8))
        self.pixel_size = 1
        self.step_size = 1
        skeletonizer = Skeletonizer(self.segmentation_stack, self.pixel_size, 1)
        self.membranes = []
        
        skeleton_points = skeletonizer.find_skeletons() 
        self.createMembranePoints_from_pts(skeleton_points)



    def createMembranePoints_from_pts(self,all_pts, add=False):
        if not add:
            self.membranes = []
        
        
        for counter, pts in enumerate(all_pts):
            
            membrane = Membrane(counter, 1, self, )
            membrane.is_circle = np.sqrt(np.sum((pts[0] - pts[-1])**2)) < 1.5


            current_points = []
            for current_pt_counter, (y,x) in enumerate(pts):
                p = Point(x,y, current_pt_counter, membrane.membrane_idx)

                current_points.append(p)
            membrane.point_list = current_points
            self.membranes.append(membrane)

    def findNeighbours(self, max_distance=150, estimate_new_vectors=True):

        for membrane in self.membranes:
            membrane:Membrane
            membrane.find_close_points(max_distance=max_distance, estimate_new_vectors=estimate_new_vectors)

    def estimateCurvature(self, favor_positive_curvature=True, max_neighbour_dist=200, gaussian_filter_size=9):
        """Estimate the curvature at each point by fitting a circle to neighbourhood points
        Parameters:
        favor_positive_curvature, bool: To make the curvature of different segments more uniform, multiply curvature of all the 
                                        points of a segment by -1 if more than half of the values are <0 
        max_neighbour_dist, float: the maximum neighbour distance for the fitting of the circle"""
        
       
        def matlab_style_gauss(size=21,sigma=10):
            """
            2D gaussian mask - should give the same result as MATLAB's
            fspecial('gaussian',[shape],[sigma])
            """
            x = np.arange(-(size//2), size//2 + 1)
            h = np.exp( -(x*x) / (2.*sigma*sigma) )
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return x,h


        def smooth_curvatures(gaus_size, curvatures, is_circle=False):
            x,gaus_filter = matlab_style_gauss(gaus_size,gaus_size // 2)
            if is_circle:
                resized_curvatures = np.concatenate((curvatures[-(gaus_size//2):], curvatures, curvatures[:(gaus_size//2)]))
            else:
                before = np.ones(gaus_size//2) * curvatures[0]
                after = np.ones(gaus_size//2) * curvatures[-1]
                resized_curvatures = np.concatenate((before, curvatures, after))
            
            
            smoothed_curvature = np.convolve(resized_curvatures, gaus_filter, "valid")


            return smoothed_curvature

        def calc_R(x,y, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f(c, x, y):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(x, y, *c)
            return Ri - Ri.mean()

        def leastsq_circle(pts):
            # coordinates of the barycenter
            y,x = pts.T
            x_m = np.mean(x)
            y_m = np.mean(y)
            center_estimate = x_m, y_m
            try:
                center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
            except:
                return 0, center_estimate
            xc, yc = center
            Ri       = calc_R(x, y, *center)
            R        = Ri.mean()
            # residu   = np.sum((Ri - R)**2)
            return R, center

        def recalculate_centers(centers, coords, new_curvatures):
            new_centers = []
            for center, coord, curv in zip(centers, coords, new_curvatures):
                center_vector = center - coord
                center_vector = center_vector / np.linalg.norm(center_vector)
                new_center = coord + center_vector * (1/curv)
                new_centers.append(new_center)
            return np.array(new_centers)


        
        if gaussian_filter_size % 2 == 0:
            gaussian_filter_size += 1
        
        self.findNeighbours(max_neighbour_dist)
        all_curvatures = {}

        for membrane in self.membranes:
            membrane:Membrane
            # pos, idxs = path_instance.get_ordered_idxs()
            # _, all_idxs = path_instance.get_ordered_idxs(n=1)
            # contour = np.array([self.skeleton_points[p].coordinates_yx for p in all_idxs])
            results = []
            points = membrane.points()
            all_points = membrane.points(n=1)
            
            neighbourhood_pts = [np.array(p.get_coords_of_neighbourhood(max_neighbour_dist)).T for p in points]

            # results = [nsphere_fit(pts) for pts in neighbourhood_pts]
            # print([len(pts) for pts in neighbourhood_pts])
            
            results = [leastsq_circle(pts) for pts in neighbourhood_pts]
           
            decider = 0
            current = []
            if membrane.is_circle:

                y,x = np.array([p.coordinates_yx for p in all_points]).T

                coords = np.array((x,y),dtype=np.int32).T
                polygonimage = np.zeros(self.segmentation_stack.shape[1:], dtype=np.uint8)
                polygonimage = cv2.fillPoly(polygonimage, [coords], 1)


            for res, p, n_pts in zip(results, points, neighbourhood_pts):                    
                radius, center = res
                p:Point                
                # If the radius is too large, assume it is a flat line instead of a circle
                # if radius is None or radius > 1e9:
                if center is None:
                    p.circle_center_yx = center
                    p.curvature_radius = None
                    p.curvature = 0
                    continue
                center = center[::-1]
                # p.set_circle_center(radius)
                p.circle_center_yx = center
                
                p.curvature_radius = radius

                # Curvature is the invers of the radius of a fitted circle
                p.curvature = 1/(radius*self.pixel_size)
                current.append(p.curvature)
                # print(radius, p.curvature)
                # Find out which side of the membrane the circle center is
                if membrane.is_circle:
                    if not p.in_polygon(polygonimage,save_path=None):
                        p.curvature *= -1
                        decider -= 1
                    else:
                        decider += 1  

                else:

                    if p.is_center_right():
                        p.curvature *= -1
                        decider -= 1
                    else:
                        decider += 1  
                
            curvatures = [p.curvature for p in points]
            centers = [p.circle_center_yx for p in points]
            coords = [p.coordinates_yx for p in points]
            if gaussian_filter_size > 1:
                new_curvatures = smooth_curvatures(gaussian_filter_size, curvatures, membrane.is_circle)
                new_centers = recalculate_centers(centers, coords, new_curvatures )   
            else:
                new_curvatures = curvatures
                new_centers = centers
            
            all_curvatures[membrane.membrane_idx] = curvatures
            for p, new_curv, new_center in zip(points, new_curvatures, new_centers):
                
                p.curvature = new_curv
                p.circle_center_yx = new_center
                p.curvature_radius = 1/new_curv
                

            if favor_positive_curvature and decider < 0:
                for p in points:
                    p.curvature *= -1
            
        return all_curvatures 


class Analyser:
    def __init__(self, micrograph_path, dataset, segmentation_path, njobs=1,name=None,pool=None, create_empty_analyser=False, pixel_size=None, use_only_closed=True,
                 max_hole_size=0, min_size=50,micrograph_pixel_size=None, step_size=13,estimate_middle_plane=True, dark_mode=False  ):
        if create_empty_analyser:
            return
        self.micrograph_path_ = None
        self.segmentation_path_ = None
        self.micrograph_path = Path(micrograph_path)
        self.segmentation_path = segmentation_path
       
        self.only_closed = use_only_closed
        self.max_hole_size = max_hole_size
        self.min_size = min_size
        self.dark_mode = dark_mode
        
        self.found_neighbours = False
        self.njobs = njobs
        # self.project_path = Path(project_path)
        self.dataset_path = Path(dataset)
        self.membranes_ = []
        self.name = name
        self.segmentation_shape_ = None
        self.micrograph_shape_ = None
        self.estimate_middle_plane = estimate_middle_plane
        if micrograph_pixel_size is None:
            self.micrograph_pixel_size = None
        else:
            self.micrograph_pixel_size = float(micrograph_pixel_size)


        self.max_distance = None
        self.pixel_size = pixel_size
        mkdir(self.dataset_path)
        # self.setSegmentationPath()
        self.setPixelSize()
        # self.segmentation(None)

        self.step_size = np.max([1,step_size/self.pixel_size])

        if np.sum(self.loadSegmentation()) == 0:
            seg = self.loadSegmentation()
            if len(seg) ==2:
                seg = sparse.as_coo(np.zeros((0, *seg.shape)))
            self.segmentation_stack = seg
            

            return
        self.createThreeDStack()

        if self.empty:
            return
        self.createMembranes(pool)



    
    @property
    def dataset(self):
        return Path(self.dataset_path).stem

    @property
    def segmentation_path(self):
        return self.segmentation_path_

    @segmentation_path.setter
    def segmentation_path(self, value):
        self.segmentation_path_ = Path(value)

    @property
    def micrograph_path(self):
        return self.micrograph_path_
    
    @micrograph_path.setter
    def micrograph_path(self, value):
        self.micrograph_path_ = Path(value)

    def addMembrane(self, membrane):
        self.membranes_.append(membrane)

    @property
    def membranes(self):
        
        if self.only_closed:
            return [membrane for membrane in self.membranes_ if membrane.is_circle]
        return [m for m in self.membranes_ ]

    @membranes.setter
    def membranes(self, membranes):
        self.membranes_ = membranes

    @property
    def number_of_membranes(self):
        return len(self.membranes_)

    @property
    def empty(self):
        return self.segmentation_stack.sum() == 0
            

    @property
    def segmentation_shape(self):
        if self.segmentation_shape_ is None:
            seg = self.loadSegmentation()
            if seg.ndim == 2:
                self.segmentation_shape_ = seg.shape
            else:
                self.segmentation_shape_ = seg.shape[1:]
        return self.segmentation_shape_


    @property
    def micrograph_shape(self):
        if self.micrograph_shape_ is None:
            self.micrograph_shape_ = self.loadMicrograph().shape
        return self.micrograph_shape_

    def setSegmentationPath(self):
        if self.segmentation_path is None:
            self.segmentation_path = self.dataset_path / (self.micrograph_path)
    
    def setPixelSize(self):
        global MRC_SUFFIX
        if self.micrograph_pixel_size is None:
            if Path(self.micrograph_path).suffix.lower() in MRC_SUFFIX:
                self.micrograph_pixel_size = mrcfile.open(self.micrograph_path, permissive=True, header_only=True).voxel_size["x"]
            else:
                warning("No micrograph pixel size was given and it could not be extracted from the files. Pixel size set to 1")
                self.micrograph_pixel_size = 1
        if self.pixel_size is None:
            
            if self.segmentation_path.exists() and Path(self.segmentation_path).suffix in MRC_SUFFIX :
                self.pixel_size = mrcfile.open(self.segmentation_path, permissive=True, header_only=True).voxel_size["x"]
            else:
                shape = self.loadSegmentation().shape[-1]
                
                microshape = self.loadMicrograph().shape[-1]
                ratio = microshape / shape

                self.pixel_size = self.micrograph_pixel_size * ratio
        


    
    # def segmentation(self, model):
    #     if model is not None and not self.segmentation_path.exists():
    #         #TODO:segmentation
    #         pass
    


    



    def createThreeDStack(self):

        def solveLayer(layer):
            unique_labels = np.unique(layer)
            if len(unique_labels) > 2:
                unique_labels = unique_labels[1:]
                new_stack = []
                
                for ul in unique_labels:
                    new_layer = (layer == ul)*1
                    new_stack.append(solveLayer(new_layer))
                return np.concatenate(new_stack)
            labels, num_features = label(layer, np.ones((3,3)))
            
            if num_features == 1:
                return np.expand_dims(labels,0)
            new_stack = []
            for label_counter in range(1,num_features + 1):
                label_image = (labels == label_counter) * 1
                negative_label_image = np.pad(label_image,1) != 1
                neg_labels, neg_features = label(negative_label_image, np.array([[0,1,0],[1,1,1],[0,1,0]]))
                neg_labels = neg_labels[1:-1,1:-1]
                for i in range(neg_features):
                    if np.sum(neg_labels == (i+1)) <= self.max_hole_size:
                        label_image[neg_labels == (i+1)] = 1


                new_stack.append(label_image)
            return np.array(new_stack)

        data = self.loadSegmentation()
        
        if data.ndim == 2:
            new_stack = solveLayer(data)
            
        else:
        
            new_stack = []
            for layer in data:
                new_stack.append(solveLayer(layer))
            new_stack = np.concatenate(new_stack)
        idxs_to_use = []
        for layer_idx, layer in enumerate(new_stack):
            skeleton = skeletonize(layer)
            idxs_to_use.append(np.sum(skeleton) >= self.min_size)
        new_stack = [layer for to_use, layer in zip(idxs_to_use, new_stack) if to_use]
        new_stack = sparse.as_coo(np.array(new_stack).astype(np.uint8))
        self.segmentation_stack = new_stack

    def findClosestVesicles(self):
        for membrane in self.membranes:
            _ = membrane.min_distance_to_closest_vesicle

    def findNeighbours(self, max_distance=150, estimate_new_vectors=True):
        """
        Finds neighboring points within a specified maximum distance.

        This method identifies neighboring points based on the given maximum
        distance. Optionally, it can also estimate new vectors for these neighbors.

        Args:
            max_distance (float, optional): The maximum distance within which to 
                search for neighbors. Defaults to 150.
            estimate_new_vectors (bool, optional): If True, new vectors will be 
                estimated for the neighbors found. Defaults to True.

        Returns:
 
        """
        if max_distance is None:
            if self.max_distance is None:
                return
            max_distance = self.max_distance
        elif self.max_distance is None:
            self.max_distance = max_distance
        elif max_distance > self.max_distance:
            self.max_distance = max_distance
        
        self.max_distance = max_distance
        self.found_neighbours = True

        for membrane in self.membranes_:
            membrane:Membrane
            membrane.find_close_points(max_distance=max_distance, estimate_new_vectors=estimate_new_vectors)
        
    def loadSegmentation(self):
        def load_mrc(file):
            return mrcfile.open(file, permissive=True).data

        def load_jpg_png(file):
            img = Image.open(file).convert("L")
            return (np.array(img) > 0) * 1

        def load_npz(file):
            data = sparse.load_npz(file).todense()
            if data.ndim == 3:
                data = np.moveaxis(data, np.argmin(data.shape), [0])
            return data
        
        suffix_function_dict = {
            ".mrc":load_mrc,
            ".png":load_jpg_png,
            ".jpg":load_jpg_png,
            ".jpeg":load_jpg_png,
            ".npz":load_npz,
            ".rec":load_mrc
        }
        suffix = Path(self.segmentation_path).suffix.lower()
        if suffix in suffix_function_dict:
            return suffix_function_dict[suffix](self.segmentation_path)
        else:
            raise ValueError(f"Could not find a function to open a {suffix} file.")

    
    def loadMicrograph(self):
        def load_mrc(file):
            return mrcfile.open(file, permissive=True).data * 1

        def load_jpg_png(file):
            img = Image.open(file).convert("L")
            return np.array(img)

        def load_npz(file):
            data = sparse.load_npz(file).todense()
            if data.ndim == 3:
                data = np.moveaxis(data, [-1], [0])
            return data
        
        suffix_function_dict = {
            ".mrc":load_mrc,
            ".png":load_jpg_png,
            ".jpg":load_jpg_png,
            ".jpeg":load_jpg_png,
            ".npz":load_npz,
            ".rec":load_mrc
        }
        suffix = Path(self.micrograph_path).suffix.lower()
        

        if suffix in suffix_function_dict:
            micrograph = suffix_function_dict[suffix](self.micrograph_path).astype(np.float32)
            if hasattr(self, "dark_mode"):
                if not self.dark_mode:
                    micrograph *= -1
            return micrograph
        else:
            raise ValueError(f"Could not find a function to open a {suffix} file.")


    def createMembranes(self, pool=None):
        
        skeletonizer = Skeletonizer(self.segmentation_stack, self.pixel_size, self.njobs)
        
        
        skeleton_points = skeletonizer.find_skeletons() 

        assert self.segmentation_stack.shape[0] == len(skeleton_points)
        
        self.createMembranePoints_from_pts(skeleton_points)


        self.findNeighbours()
        if not self.estimate_middle_plane:

            return
        number_of_membranes = len(skeleton_points)

        tangents = []

        for membrane in self.membranes_:
            tangents.append([point.normal_vector for point in membrane.points(n=1)])

        orig_image = self.getResizedMicrograph()
        ridge_detector = RidgeDetector(self.segmentation_stack, orig_image, self.pixel_size,self.njobs, Path(self.micrograph_path).stem)
        
        ridge_points = ridge_detector.find_skeletons(pool) 
        
        self.createMembranePoints_from_pts(ridge_points, add=True)

        
        new_points = []
        

        window_size = 5
        to_add = int((window_size-1) // 2)
        for_thickness_estimation = []
       
        for idx in range(number_of_membranes): 
            

            distances = cdist(skeleton_points[idx], ridge_points[idx])
            distances_args = np.argmin(distances, 0)
            min_distances = np.min(distances,0)

            signs = []
            tang = tangents[idx]

            # Figure out in which direction the ridge point is from the skeleton points
            for i, (argmin, pt) in enumerate(zip(distances_args, ridge_points[idx])):
                t = tang[argmin]
                p_vector = pt - skeleton_points[idx][argmin]
                if np.all(p_vector == 0):
                    signs.append(1)
                    continue
                p_vector = p_vector / np.linalg.norm(p_vector)
                
                dot = np.dot(t, p_vector)
                signs.append(np.sign(dot))


            min_distances = min_distances * signs

            min_distances = np.concatenate((min_distances[-to_add:], min_distances, min_distances[:to_add]))

            df = pd.Series(min_distances)
            rolling_std = df.rolling(window_size, center=True).std()
            rolling_std = rolling_std[to_add:-to_add]


            if np.all(rolling_std < 1):
                new_points.append(ridge_points[idx])
                for_thickness_estimation.append(True)
                
            else:
                new_points.append(skeleton_points[idx])
                for_thickness_estimation.append(False)
        
        
        self.createMembranePoints_from_pts(new_points)
        for membrane in self.membranes:
            membrane.improved = for_thickness_estimation[membrane.membrane_idx]
        self.max_distance = None
        return 
    
    def createMembranePoints_from_pts(self,all_pts, add=False):
        if not add:
            self.membranes = []
        
        pt_counter = self.numberOfPoints
        for pts in all_pts:
            
            membrane = Membrane(self.number_of_membranes, self.step_size, self, )
            membrane.is_circle = np.sqrt(np.sum((pts[0] - pts[-1])**2)) < 1.5


            current_points = []
            for current_pt_counter, (y,x) in enumerate(pts):
                p = Point(x,y, current_pt_counter, membrane.membrane_idx)

                current_points.append(p)
            membrane.point_list = current_points
            self.addMembrane(membrane)
            # self.membranes.append(membrane)

    @property
    def numberOfPoints(self):
        number = 0
        for membrane in self.membranes:
            number += len(membrane)
        
    

    def draw_skeleton_stack(self):
        stack = np.zeros(self.segmentation_stack.shape[1:])
        for membrane in self.membranes:
            membrane:Membrane
            for point in membrane.point_list:
                point:Point
                y,x = point.coordinates_yx
                stack[y,x] = 1
        return stack



    def estimateCurvatureAdaptive(self):


        from circle_fit import hyperLSQ
        from scipy.stats import linregress
        from scipy.signal import savgol_filter

        def smooth(x, wl, method="interp"):
            return savgol_filter(x, min(len(x), wl),3,mode=method)
        def check_collinear(points, threshold=1e-6, return_distances=False):
            """
            Check if points are collinear.
            
            :param points: List of (x, y) tuples.
            :param threshold: Maximum allowable deviation to consider points collinear.
            :return: True if points are collinear, False otherwise.
            """
            x, y = points.T

            
            # Check if all x-values are the same (vertical line)
            if np.all(x == x[0]):
                return True
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            
            # Calculate the distances from the line y = slope * x + intercept
            distances = np.abs(y - (slope * x + intercept))
            if return_distances:
                return np.all(distances < threshold), distances 
            # Check if all distances are below the threshold
            return np.all(distances < threshold)

        def f(c, x, y):
            """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
            Ri = calc_R(x, y, *c)
            return Ri - Ri.mean()


        def calc_R(x,y, xc, yc):
            """ calculate the distance of each 2D points from the center (xc, yc) """
            return np.sqrt((x-xc)**2 + (y-yc)**2)
        def matlab_style_gauss(size=21,sigma=10):
            """
            2D gaussian mask - should give the same result as MATLAB's
            fspecial('gaussian',[shape],[sigma])
            """
            x = np.arange(-(size//2), size//2 + 1)
            h = np.exp( -(x*x) / (2.*sigma*sigma) )
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return x,h

        def smooth_curvatures(gaus_size, curvatures, is_circle=False):
            x,gaus_filter = matlab_style_gauss(gaus_size,gaus_size // 2)
            if is_circle:
                resized_curvatures = np.concatenate((curvatures[-(gaus_size//2):], curvatures, curvatures[:(gaus_size//2)]))
            else:
                before = np.ones(gaus_size//2) * curvatures[0]
                after = np.ones(gaus_size//2) * curvatures[-1]
                resized_curvatures = np.concatenate((before, curvatures, after))
            
            
            smoothed_curvature = np.convolve(resized_curvatures, gaus_filter, "valid")


            return smoothed_curvature

        def convert_input(coords):
            """
            Converts the input coordinates from a 2D List or 2D np.ndarray to 2 separate 1D np.ndarrays.

            Parameters
            ----------
            coords: 2D List or 2D np.ndarray of shape (n,2). X,Y point coordinates.

            Returns
            -------
            x   : np.ndarray. X point coordinates.
            y   : np.ndarray. Y point coordinates.
            """
            if isinstance(coords, np.ndarray):
                assert coords.ndim == 2, "'coords' must be a (n, 2) array"
                assert coords.shape[1] == 2, "'coords' must be a (n, 2) array"
                x = coords[:, 0]
                y = coords[:, 1]
            elif isinstance(coords, list):
                x = np.array([point[0] for point in coords])
                y = np.array([point[1] for point in coords])
            else:
                raise Exception("Parameter 'coords' is an unsupported type: " + str(type(coords)))
            return x, y


        def sigma(x, y, xc: float, yc: float, r: float) -> float:
            """
            Computes the sigma (RMS error) of a circle fit (xc, yc, r) to a set of 2D points (x, y).
            ----------
            x   : np.ndarray. X point coordinates.
            y   : np.ndarray. Y point coordinates.
            xc  : float. Circle center X coordinate.
            yc  : float. Circle center Y coordinate.
            r   : float. Circle radius.

            Returns
            -------
            sigma : float. Root Mean Square of error (distance) between points (x, y) and circle (xc, yc, r).
            """
            dx = x - xc
            dy = y - yc
            s: float = np.sqrt(np.mean((np.sqrt(dx ** 2 + dy ** 2) - r) ** 2))
            return s

        def test_hyperLSQ(coords, iter_max: int = 99):
            """
            Kenichi Kanatani, Prasanna Rangarajan, "Hyper least squares fitting of circles and ellipses"
            Computational Statistics & Data Analysis, Vol. 55, pages 2197-2208, (2011)

            Parameters
            ----------
            coords: 2D List or 2D np.ndarray of shape (n,2). X,Y point coordinates.
            iter_max    : Optional int. Maximum number of iterations for the iterative fitting algorithm.

            Returns
            -------
            xc  : float. x coordinate of the circle fit
            yc  : float. y coordinate of the circle fit
            r   : float. Radius of the circle fit
            s   : float. Sigma (RMS of error) of the circle fit
            """
            x, y = convert_input(coords)
            n = x.shape[0]

            Xi = x - x.mean()
            Yi = y - y.mean()
            Zi = Xi * Xi + Yi * Yi

            # compute moments
            Mxy = (Xi * Yi).sum() / n
            Mxx = (Xi * Xi).sum() / n
            Myy = (Yi * Yi).sum() / n
            Mxz = (Xi * Zi).sum() / n
            Myz = (Yi * Zi).sum() / n
            Mzz = (Zi * Zi).sum() / n

            # computing the coefficients of characteristic polynomial
            Mz = Mxx + Myy
            Cov_xy = Mxx * Myy - Mxy * Mxy
            Var_z = Mzz - Mz * Mz

            A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
            A1 = Var_z * Mz + 4. * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
            A0 = Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy
            A22 = A2 + A2

            # finding the root of the characteristic polynomial
            Y = A0
            X = 0.
            for i in range(iter_max):
                Dy = A1 + X * (A22 + 16. * (X ** 2))
                xnew = X - Y / Dy
                if xnew == X or not np.isfinite(xnew):
                    break
                ynew = A0 + xnew * (A1 + xnew * (A2 + 4. * xnew * xnew))
                if abs(ynew) >= abs(Y):
                    break
                X, Y = xnew, ynew

            det = X ** 2 - X * Mz + Cov_xy
            if np.isclose(det, 0):   
                return None, None, None, None
            Xcenter = (Mxz * (Myy - X) - Myz * Mxy) / det / 2.
            Ycenter = (Myz * (Mxx - X) - Mxz * Mxy) / det / 2.

            xc: float = Xcenter + x.mean()
            yc: float = Ycenter + y.mean()
            r = np.sqrt(abs(Xcenter ** 2 + Ycenter ** 2 + Mz))
            s = sigma(x, y, xc, yc, r)
            return xc, yc, r, s





        max_distance = int(1500/7)
        min_distance = 5
        gaussian_filter_size = 9
        threshold = 2
        current_min_distances = np.arange(min_distance,max_distance, step=15)

        membrane_curvatures = {}
        for membrane in self.membranes:
           
            coords = membrane.coords
            smooth_int = int(400/self.pixel_size)
            smooth_int = min(smooth_int, len(coords) // 4)
            smooth_int = int(len(coords) * 0.1)
            coords = np.array((smooth(coords.T[0],smooth_int, method="wrap"),smooth(coords.T[1],smooth_int, method="wrap"))).T
            

            old_coords = membrane.coords
            curvature = [p.curvature for p in membrane.points(n=1)]


            for i in range(len(membrane.points(n=1))):
                membrane.point_list[i].coordinates_yx = coords[i]

            
            membrane.find_close_points(max_distance=max_distance, estimate_new_vectors=False)

            neighbourhood_pts = [np.array(p.get_coords_of_neighbourhood(min_distance)).T for p in membrane.points(n=1)]


            
            
            
            current = []
            if membrane.is_circle:

                y,x = np.array([p.coordinates_yx for p in membrane.points(n=1)]).T

                coords = np.array((x,y),dtype=np.int32).T
                polygonimage = np.zeros(self.segmentation_stack.shape[1:], dtype=np.uint8)
                polygonimage = cv2.fillPoly(polygonimage, [coords], 1)

                
            for point_counter, (p, n_pts) in enumerate(zip(membrane.points(n=1), neighbourhood_pts)):          
                
                neighbourhood = p.neighbourhood_points


                neighbour_idxs = np.array([n[0] for n in neighbourhood[0]])
                neighbour_idxs = np.concatenate([neighbour_idxs, [p.idx], [n[0] for n in neighbourhood[1]]])

                neighbour_distances = np.array([n[1] for n in neighbourhood[0]])
                neighbour_distances = np.concatenate([neighbour_distances, [0], [n[1] for n in neighbourhood[1]]])

                max_neighbour_distance = np.max(neighbour_distances)

                neighbour_coords = np.array([n[2] for n in neighbourhood[0]])
                neighbour_coords = np.concatenate([neighbour_coords, [p.coordinates_yx], [n[2] for n in neighbourhood[1]]])
                

                best_radius = None
                best_center = None

                allowed_to_change = True
                
                current_distances = []
                for cmd in current_min_distances:
                    if not allowed_to_change:
                        current_distances.append(True)
                        continue

                    
                    pts = neighbour_coords[neighbour_distances < cmd]

                    method = test_hyperLSQ


                    res = method(pts)

                    if check_collinear(pts, 0.1) or res[0] is None:
                        radius = np.inf
                        center = p.coordinates_yx[::-1]
                        current_distances.append(False)
                        best_radius = radius
                        best_center = center
                        continue
                    

                    xc, yc, r, sig = res

                    center = np.array((yc,xc))
                    radius = r


                    y,x = pts.T
                    distance = np.abs(f(center, x, y))
                    current_distances.append(np.any(distance > threshold))
                    if not current_distances[-1] and allowed_to_change:
                        best_radius = radius
                        best_center = center
                    if current_distances[-1] and allowed_to_change:
                        allowed_to_change = False
                    if cmd > max_neighbour_distance:
                        allowed_to_change = False
                    


                if best_radius is None:                  
                    
                    
                    
                    pts = np.array(p.get_coords_of_neighbourhood(min_distance)).T
                    method = test_hyperLSQ
                    
                    res = method(pts)
                    
                    if res[0] is None:
                        best_radius = np.inf
                        best_center = p.coordinates_yx[::-1]
                    else:
                        xc, yc, r, sig = res
                        
                        best_radius = r
                        best_center = np.array((yc,xc))


                radius = best_radius
                center = best_center

                center = center[::-1]


                p.circle_center_yx = center
                
                # p.curvature_radius = radius

                

                # Curvature is the invers of the radius of a fitted circle
                if radius == np.inf:
                    c = 0
                else:
                    c = 1/(radius*self.pixel_size)

                # Find out which side of the membrane the circle center is
                    if membrane.is_circle:
                        if not p.in_polygon(polygonimage,save_path=None):
                            
                            c *= -1

                p.curvature = c
                current.append(c)
                

            if gaussian_filter_size > 1:
                new_curvatures = smooth_curvatures(gaussian_filter_size, current, membrane.is_circle)
            else:
                new_curvatures = current

            for current_c, p, old_c in zip(new_curvatures, membrane.points(n=1), old_coords):
                p.coordinates_yx = old_c
                p.curvature = current_c


            test = membrane.resize_curvature(200,100)

            membrane_curvatures[membrane.membrane_idx] = test

        return membrane_curvatures
        




    def estimateCurvature(self, favor_positive_curvature=True, max_neighbour_dist=200, gaussian_filter_size=9, pool=None):
        """
        Estimates the curvature of elements in a dataset.

        This method calculates the curvature of points, with an option to favor
        positive curvature values. It uses neighboring points within a specified
        maximum distance and applies a Gaussian filter to smooth the results.

        Args:
            favor_positive_curvature (bool, optional): If True, the method will 
                prioritize positive curvature values. Defaults to True.
            max_neighbour_dist (int, optional): The maximum distance to consider 
                for neighboring points when estimating curvature. Defaults to 200.
            gaussian_filter_size (int, optional): The size of the Gaussian filter 
                used for smoothing the curvature estimates. Defaults to 9.
            pool (multiprocessing.Pool, optional): A pool of worker processes for 
                parallel computation. If None, the computation will be done 
                sequentially. Defaults to None.
        Returns
        all_curavtures (dict) : Dictionary of all the curavture values with membrane indexes as keys
        """
        if os.environ["CRYOVIA_MODE"] == "1":
            return self.estimateCurvatureAdaptive()
       
        def matlab_style_gauss(size=21,sigma=10):
            """
            2D gaussian mask - should give the same result as MATLAB's
            fspecial('gaussian',[shape],[sigma])
            """
            x = np.arange(-(size//2), size//2 + 1)
            h = np.exp( -(x*x) / (2.*sigma*sigma) )
            sumh = h.sum()
            if sumh != 0:
                h /= sumh
            return x,h


        def smooth_curvatures(gaus_size, curvatures, is_circle=False):
            x,gaus_filter = matlab_style_gauss(gaus_size,gaus_size // 2)
            if is_circle:
                resized_curvatures = np.concatenate((curvatures[-(gaus_size//2):], curvatures, curvatures[:(gaus_size//2)]))
            else:
                before = np.ones(gaus_size//2) * curvatures[0]
                after = np.ones(gaus_size//2) * curvatures[-1]
                resized_curvatures = np.concatenate((before, curvatures, after))
            
            
            smoothed_curvature = np.convolve(resized_curvatures, gaus_filter, "valid")


            return smoothed_curvature


        def recalculate_centers(centers, coords, new_curvatures):
            new_centers = []
            for center, coord, curv in zip(centers, coords, new_curvatures):
                center_vector = center - coord
                center_vector = center_vector / np.linalg.norm(center_vector)
                new_center = coord + center_vector * (1/curv)
                new_centers.append(new_center)
            return np.array(new_centers)


        
        if gaussian_filter_size % 2 == 0:
            gaussian_filter_size += 1
        
        self.findNeighbours(max_neighbour_dist)
        all_curvatures = {}

        for membrane in self.membranes:
            membrane:Membrane
            # pos, idxs = path_instance.get_ordered_idxs()
            # _, all_idxs = path_instance.get_ordered_idxs(n=1)
            # contour = np.array([self.skeleton_points[p].coordinates_yx for p in all_idxs])
            results = []
            points = membrane.points()
            all_points = membrane.points(n=1)
            
            neighbourhood_pts = [np.array(p.get_coords_of_neighbourhood(max_neighbour_dist)).T for p in points]

            # results = [nsphere_fit(pts) for pts in neighbourhood_pts]
            # print([len(pts) for pts in neighbourhood_pts])
            if pool is None:
                results = [utils.leastsq_circle(pts) for pts in neighbourhood_pts]
            else:
                
                results = [pool.apply_async(utils.leastsq_circle, args=[pts]) for pts in neighbourhood_pts]
                results = [res.get() for res in results]
            # results = [leastsq_circle(pts) for pts in neighbourhood_pts]

            decider = 0
            current = []
            if membrane.is_circle:

                y,x = np.array([p.coordinates_yx for p in all_points]).T

                coords = np.array((x,y),dtype=np.int32).T
                polygonimage = np.zeros(self.segmentation_stack.shape[1:], dtype=np.uint8)
                polygonimage = cv2.fillPoly(polygonimage, [coords], 1)


            for res, p, n_pts in zip(results, points, neighbourhood_pts):                    
                radius, center = res
                p:Point                
                # If the radius is too large, assume it is a flat line instead of a circle
                # if radius is None or radius > 1e9:
                if center is None:
                    p.circle_center_yx = center
                    p.curvature_radius = None
                    p.curvature = 0
                    continue
                center = center[::-1]
                # p.set_circle_center(radius)
                p.circle_center_yx = center
                
                radius = max(1, radius)
                p.curvature_radius = radius

                # Curvature is the invers of the radius of a fitted circle
                p.curvature = 1/(radius*self.pixel_size)
                current.append(p.curvature)
                # print(radius, p.curvature)
                # Find out which side of the membrane the circle center is
                if membrane.is_circle:
                    if not p.in_polygon(polygonimage,save_path=None):
                        p.curvature *= -1
                        decider -= 1
                    else:
                        decider += 1  

                else:

                    if p.is_center_right():
                        p.curvature *= -1
                        decider -= 1
                    else:
                        decider += 1  
                
            curvatures = [p.curvature for p in points]
            centers = [p.circle_center_yx for p in points]
            coords = [p.coordinates_yx for p in points]
            if gaussian_filter_size > 1:
                new_curvatures = smooth_curvatures(gaussian_filter_size, curvatures, membrane.is_circle)
                new_centers = recalculate_centers(centers, coords, new_curvatures )   
            else:
                new_curvatures = curvatures
                new_centers = centers
            
            all_curvatures[membrane.membrane_idx] = curvatures
            for p, new_curv, new_center in zip(points, new_curvatures, new_centers):
                
                p.curvature = new_curv
                p.circle_center_yx = new_center
                p.curvature_radius = 1/new_curv
                

            if favor_positive_curvature and decider < 0:
                for p in points:
                    p.curvature *= -1
            
            membrane.distribute_attributes_to_neighbours()

        # Share curvature with the neighbours
        # self.distribute_attributes_to_neighbours()
        
        
        return all_curvatures 
    


    
        

    def getResizedMicrograph(self):
        if np.all([i == j for i,j in zip(self.segmentation_shape, self.micrograph_shape)]):
            return self.loadMicrograph()
        return utils.resizeMicrograph(self.loadMicrograph(), self.segmentation_shape)

    def getResizedSegmentation(self, pool=None, new_max_membrane_width=None, no_multiprocessing=False):
        if np.all([i == j for i,j in zip(self.segmentation_shape, self.micrograph_shape)]):
            return self.segmentation_stack
        else:
            if new_max_membrane_width is None:
                if pool is None:
                    if no_multiprocessing:
                        result = [utils.resizeSegmentation(self.segmentation_stack[i, ...], self.micrograph_shape) for i in range(self.segmentation_stack.shape[0])]
                    else:
                        with mp.Pool(self.njobs) as pool:
                            result = [pool.apply_async(utils.resizeSegmentation, [self.segmentation_stack[i, ...], self.micrograph_shape]) for i in range(self.segmentation_stack.shape[0])]
                            result = [res.get() for res in result]
                    resized_label_image = sparse.stack(result)
                else:
                    result = [pool.apply_async(utils.resizeSegmentation, [self.segmentation_stack[i, ...], self.micrograph_shape]) for i in range(self.segmentation_stack.shape[0])]
                    result = [res.get() for res in result]
                    resized_label_image = sparse.stack(result)
            else:

                if pool is None:
                    if no_multiprocessing:
                        result = [utils.resizeSegmentationFromCoords(self.membranes_[i].coords, self.segmentation_stack[i, ...].shape, self.micrograph_shape, new_max_membrane_width, self.pixel_size) for i in range(self.segmentation_stack.shape[0])]

                    else:
                        with mp.Pool(self.njobs) as pool:
                            result = [pool.apply_async(utils.resizeSegmentationFromCoords, [self.membranes_[i].coords, self.segmentation_stack[i, ...].shape, self.micrograph_shape, new_max_membrane_width, self.pixel_size]) for i in range(self.segmentation_stack.shape[0])]
                            result = [res.get() for res in result]
                    resized_label_image = sparse.stack(result)
                else:
                    result = [pool.apply_async(utils.resizeSegmentationFromCoords, [self.membranes_[i].coords, self.segmentation_stack[i, ...].shape, self.micrograph_shape, new_max_membrane_width, self.pixel_size]) for i in range(self.segmentation_stack.shape[0])]
                    result = [res.get() for res in result]
                    resized_label_image = sparse.stack(result)
            return resized_label_image


    def estimateThickness(self, max_neighbour_dist=150, min_thickness=20, max_thickness=70, pool=None, sigma=2, no_multiprocessing=False ):
        """
        Estimates the thickness of points for all membranes.

        This method calculates the thickness of points based the density values around neighboring points within a specified maximum distance. 
        It also applies a Gaussian filter to smooth the results and provides options for multiprocessing.

        Args:
            max_neighbour_dist (int, optional): The maximum distance to consider for neighboring points when estimating thickness. Defaults to 150.
            min_thickness (int, optional): The minimum allowable thickness. Defaults to 20.
            max_thickness (int, optional): The maximum allowable thickness. Defaults to 70.
            pool (multiprocessing.Pool, optional): A pool of worker processes for parallel computation. If None, the computation will be done sequentially. Defaults to None.
            sigma (float, optional): The standard deviation for Gaussian kernel used in smoothing the thickness estimates. Defaults to 2.
            no_multiprocessing (bool, optional): If True, disables the use of multiprocessing even if a pool is provided. Defaults to False.

        Returns:
        """
       
        
        def create_distance_map(pool):

            length_of_filter = max(1, int(10/self.micrograph_pixel_size))
            image_filter = utils.matlab_style_gauss2D(shape=(length_of_filter, length_of_filter), sigma=10/self.micrograph_pixel_size)  
            smoothed_image = signal.convolve(self.loadMicrograph(), image_filter, mode="same")

            resized_segmentation = self.getResizedSegmentation(pool,max_thickness, no_multiprocessing=no_multiprocessing)

            distance_maps = {}
            cropped_images = {}
            distance_maps_min_max = {}
            indices = {}
            # dummy_images = {}

            ratio = self.micrograph_shape[0] / self.segmentation_shape[0]

            interp_ys, interp_xs = {}, {}
            for membrane in self.membranes:
                interp_y, interp_x = membrane.getResizedCoords(ratio).T
                interp_ys[membrane.membrane_idx] = interp_y
                interp_xs[membrane.membrane_idx] = interp_x
               
            
            # croppeds = []
            if no_multiprocessing:
                result = [utils.create_distance_map(membrane, ratio, resized_segmentation[membrane.membrane_idx], self.micrograph_pixel_size, (interp_ys[membrane.membrane_idx],interp_xs[membrane.membrane_idx])) for  membrane in self.membranes]

            else:
                result = [pool.apply_async(utils.create_distance_map, [membrane, ratio, resized_segmentation[membrane.membrane_idx], self.micrograph_pixel_size, (interp_ys[membrane.membrane_idx],interp_xs[membrane.membrane_idx])]) for  membrane in self.membranes]
                result = [res.get() for res in result]
            for res,membrane in zip(result, self.membranes):
                idx = membrane.membrane_idx
                current_distance_map, new_indice_map, (y_min,y_max, x_min,x_max), to_estimate, cropped = res
                distance_maps[idx] = current_distance_map
                indices[idx] = new_indice_map
                cropped_images[idx] = smoothed_image[y_min:y_max, x_min:x_max]
                membrane.to_estimate = to_estimate

                # croppeds.append(cropped)    
            # self.croppeds = croppeds
            
            return distance_maps, cropped_images, indices
        
            
            return distance_maps, cropped_images, distance_maps_min_max, ratio, indices
        
        def getThicknessEstimation(distance_idxs, bins, original_values):
            new_bins = []
            profile = []
            distances = []
            for bin_value in np.unique(distance_idxs):
                new_bins.append(bins[bin_value - 1])
                idxs = np.argwhere(distance_idxs == bin_value)
                values = original_values[idxs]
                distances.append(bins[bin_value - 1])
                profile.append(np.mean(values))
            

            
            
            nr_of_spline_points = int(np.max(distances) - np.min(distances))
            
            # t,c,k = splrep(np.linspace(0, height, len(profile)), profile)
            t,c,k = splrep(distances, profile)
            profile = BSpline(t,c,k)(np.linspace(int(np.min(distances)), int(np.max(distances)), nr_of_spline_points))
            zero_idx = np.where(np.array(distances) == 0)[0]

            zero_idx = nr_of_spline_points / len(distances) * zero_idx
            
           

            result = utils.get_thickness_from_profile(profile, zero_idx, min_thickness, max_thickness)
            return result, zero_idx, profile





        def estimate_thickness( pool, distance_maps, cropped_images, indices):

            if no_multiprocessing:
                result = [utils.estimate_thickness(membrane, distance_maps[membrane.membrane_idx], cropped_images[membrane.membrane_idx], indices[membrane.membrane_idx], max_neighbour_dist, min_thickness, max_thickness, self.micrograph_path.stem)
                        for membrane in self.membranes]
            else:
                result = [pool.apply_async(utils.estimate_thickness, [membrane, distance_maps[membrane.membrane_idx], cropped_images[membrane.membrane_idx], indices[membrane.membrane_idx], max_neighbour_dist, min_thickness, max_thickness, self.micrograph_path.stem])
                        for membrane in self.membranes]
                result = [res.get() for res in result]
            for membrane, res in zip(self.membranes, result):
                
                membrane_attributes, point_attributes = res
                for key, value in membrane_attributes.items():
                    setattr(membrane, key, value)
                for point, point_attribute in zip(membrane.points(), point_attributes):
                    for key, value in point_attribute.items():
                        setattr(point, key, value)

           
                        

        for membrane in self.membranes:
            membrane.thickness = None
            membrane.smoothed_thickness_profile = {"profile":[], "fp":None, "sp":None, "middle_idx":None}
            membrane.thickness_profile = []
        self.findNeighbours(max_neighbour_dist, False)
        for membrane in self.membranes:
            membrane.analyser = None

        if pool is None:
            if no_multiprocessing:
                args = create_distance_map(None)
                estimate_thickness(None, *args)
            else:
                with mp.get_context("spawn").Pool(self.njobs) as pool:
                    args = create_distance_map(pool)
                    # return args
                    estimate_thickness(pool, *args)
        else:
            args = create_distance_map(pool)
            # return args
            estimate_thickness(pool, *args)
        for membrane in self.membranes:
            membrane.analyser = self
            if sigma is not None:
                membrane.distribute_attributes_to_neighbours(["thickness"], sigma)
        return

    def predictShapes(self, predictor:Path):
        """
        Predicts shapes of all vesicles using the specified shape classifier.

        This method applies a shape prediction model provided by the predictor to 
        elements in the dataset, generating predictions for each element.

        Args:
            predictor (Path): A Path object pointing to the shape prediction model 
                to be used for making predictions.

        Returns:

        """
        if isinstance(predictor, (str, Path)):
            with open(predictor, "rb") as file:
                predictor = CustomUnpickler(file).load()

        for membrane in self.membranes:
            membrane:Membrane
            if not membrane.is_circle and predictor.only_closed:
                membrane.shape = "not circular"
                membrane.shape_probability = 1
            else:
                curv = membrane.resize_curvature(200,100)
                
                if curv is None or len(curv) == 0:
                    continue

                shape, proba = predictor.predict(curv)
                
                membrane.shape = shape
                membrane.shape_probability = proba



    def identifyIceContaminations(self, pool=None):
        """
        Calculates a value for the vesicles being ice contamination by comparing the density values inside and outside.

        This method detects ice contaminations in the dataset. It can utilize 
        parallel processing if a multiprocessing pool is provided.

        Args:
            pool (multiprocessing.Pool, optional): A pool of worker processes for 
                parallel computation. If None, the computation will be done 
                sequentially. Defaults to None.

        Returns:

        """

        # if pool is None: 
        #     for membrane in self.membranes:
        #         membrane.isIceContamination = 
        resized_micrograph = self.getResizedMicrograph()
        resized_micrograph -= np.mean(resized_micrograph)
        resized_micrograph /= np.std(resized_micrograph)
        
        for membrane in self.membranes:
            y,x = membrane.coords.T
            pts = np.array(np.array([x,y]).T)

            vesicle = np.zeros(self.segmentation_shape, dtype=np.uint8)

            cv2.drawContours(vesicle, [pts], -1, 1, -1)
            
            
            inner_mean = np.mean(resized_micrograph[vesicle != 0])
            membrane.isIceContamination = inner_mean



    def findEnclosedVesicles(self):
        """
        Identifies vesicles enclosed by other.

        Args:
            None

        Returns:
            
        """
        idxs_to_use = []
        areas = []
        for idx, membrane in enumerate(self.membranes):
            if membrane.is_circle:
                areas.append(membrane.area)
                idxs_to_use.append(idx)
        
        sorted_areas_idxs = np.argsort(areas,)[::-1]
        idxs_to_use = np.array(idxs_to_use)[sorted_areas_idxs]
        for membrane in self.membranes:
            membrane.is_enclosed_in = []
            membrane.encloses = []
        if len(idxs_to_use) <= 1:
            return
        for counter, idx in enumerate(idxs_to_use[:-1]):
            membrane:Membrane = self.membranes[idx]
            
            pts = membrane.coords

            first_polygon = np.zeros(self.segmentation_shape[::-1], dtype=np.uint8)

            cv2.drawContours(first_polygon, [pts], -1, 1, -1)

            # 

            for second_idx in range(idx + 1, len(self.membranes)):
                second_membrane:Membrane = self.membranes[second_idx]
                
                s_pts = second_membrane.coords
                current_polygon = np.zeros(self.segmentation_shape[::-1], dtype=np.uint8)

                cv2.drawContours(current_polygon, [s_pts], -1, 1, -1)
                # print(np.sum(current_polygon), second_idx)
                # 

                nr_of_pts = np.sum(current_polygon)
                
                overlap = np.sum(first_polygon * current_polygon)
                # 
                
                # print(overlap, idx, second_idx)
                if overlap == nr_of_pts and overlap > 0:
                    distances = cdist(pts, s_pts)
                    min_distance = np.min(distances) * self.pixel_size
                    second_membrane.is_enclosed_in.append((idx, min_distance))
                    membrane.encloses.append((second_idx, min_distance))
        


    def __getitem__(self, value):
        for membrane in self.membranes_:
            if membrane.membrane_idx == value:
                return membrane
        raise IndexError(str(value))

    @staticmethod
    def save_all(analysers, njobs=1, pool=None):
        """
        Saves the state of all analyzers in the list.

        This static method saves the state of each analyzer in the provided list of 
        analyzers. It supports parallel processing to speed up the saving process 
        if a multiprocessing pool is provided.

        Args:
            analysers (list): A list of analyzer objects whose state needs to be saved.
            njobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
            pool (multiprocessing.Pool, optional): A pool of worker processes for 
                parallel computation. If None, the computation will be done sequentially. 
                Defaults to None.

        Returns:
            None
        """
        
            
        if njobs <= 1 and pool is None:
            return [analyser.save() for analyser in analysers]
        elif pool is not None:
            results = [pool.apply_async(analyser.save, args=[]) for analyser in analysers]
            return [res.get() for res in results]
        else:
            with mp.get_context("spawn").Pool(njobs) as pool:
                results =  [pool.apply_async(analyser.save, args=[]) for analyser in analysers]
                return [res.get() for res in results]
      

    def save(self, save_path=None, protocol=0,remove_neighbours=True, larch=False):
        """
        Saves the state of this analyser.

        This static method saves the state of each analyzer in the provided list of 
        analyzers. It supports parallel processing to speed up the saving process 
        if a multiprocessing pool is provided.

        Args:
            save_path (Path, optional): Path where to save this analyser
            protocol (int, optional): which pickle protocol to use
            remove_neighbours: redundant
            larch           : redundant


        Returns:
            wrapper_dir (path) : the directory of the wrapper
        """
        
        if save_path is None:
            save_path = self.dataset_path / (self.micrograph_path.stem + ".pickle")
        self.found_neighbours = False
        # for membrane in self.membranes:
        #     membrane :Membrane
        #     membrane.analyser_ = None
        #     for p in membrane.point_list:
                
        #         p.thickness_profile["profile"] = []
        #         p.thickness_profile["unsmoothed"] = []
        
        wrapper = AnalyserWrapper.from_analyser(self,True)
        
        # for membrane in self.membranes:
        #     membrane :Membrane
        #     membrane.analyser_ = self
        return wrapper.directory



    
    def createImprovedSegmentation(self, config,stepSize=None):
        """
        Creates an image stack the segmentation with improved segmentation as 1 and not improved as 2. 

        This method is used extracting patches for specific segmentation model training

        Args:
            config (config): the config of the segmentation model
            stepSize : the stepSize for the patches to extract
        


        Returns:
            segPatches : the extracted patches of the segmentations
            dataPatches: the extracted patches of the micrographs

        """


        from cryovia.gui.segmentation_files.prep_training_data import patchify
        improvedSeg = np.zeros((self.segmentation_shape), np.uint8)
        self.only_closed = False
        for membrane in self.membranes:
            membrane:Membrane
            
            if config.thin_segmentation:
                y,x = membrane.coords.T
                dummy = np.zeros((self.segmentation_shape), np.uint8)
                dummy[y,x] = 1


                for i in range(config.dilation):
                    dummy = binary_dilation(dummy)
                if membrane.improved:
                    improvedSeg[dummy>0] = 1
                else:
                    improvedSeg[dummy>0] = 2
            else:
                if membrane.improved:
                    improvedSeg[self.segmentation_stack[membrane.membrane_idx] > 0] = 1
                else:
                    improvedSeg[self.segmentation_stack[membrane.membrane_idx] > 0] = 2


        segPatches = patchify(improvedSeg, config,stepSize=stepSize)
        idxs = np.sum(segPatches==2,axis=(1,2)) == 0
  
        segPatches = segPatches[idxs]
        dataPatches = patchify(self.getResizedMicrograph(), config,stepSize=stepSize)
        dataPatches = dataPatches[idxs]
        return segPatches, dataPatches

        

   


    def remove_indexes(self, indexes):
        """
        Removes membranes from the analyser at specified indexes.


        Args:
            indexes (list): A list of indexes indicating which membranes to remove 
                from the analyser.

        Returns:
            None
        """
        indexes = set(indexes)
        usable_indexes = set([m.membrane_idx for m in self.membranes_ if m.membrane_idx not in indexes])
        self.remove_all_other_indexes(usable_indexes)

    def remove_all_other_indexes(self, indexes):
        """
        Removes all membranes from the analyser except those at specified indexes.

  

        Args:
            indexes (list): A list of membrane indexes indicating which membranes to retain in 
                the analyser.

        Returns:
            None
        """
        indexes = sorted(indexes)
        try:
            before_shape = self.segmentation_stack.shape
            self.segmentation_stack = self.segmentation_stack[indexes]
        except IndexError as e:
            raise e

        indexes = set(indexes)
        new_membranes = [m for m in self.membranes_ if m.membrane_idx in indexes]

        assert len(self.segmentation_stack) == len(new_membranes)

        for counter, membrane in enumerate(new_membranes):
            membrane.membrane_idx = counter
        self.membranes_ = new_membranes
        for membrane in self.membranes:
            membrane.min_distance_to_closest_vesicle = None
            membrane.distance_to_enclosing_vesicle = None
            membrane.is_enclosed_in = []
            
        self.findClosestVesicles()
        self.findEnclosedVesicles()
        for membrane in self.membranes:
            _ = membrane.distance_to_enclosing_vesicle
        


    def applyMask(self, mask):
        """
        Removes all membranes found not on the mask.
        Args:
            mask (list): A mask indicating where to keep membranes.

        Returns:
            None
        """
        inverse_mask = np.logical_not(mask > 0)
        if any((i != j for i,j in zip(mask.shape, self.segmentation_stack.shape[1:]))):
            inverse_mask = utils.resizeSegmentation(inverse_mask, self.segmentation_shape)
        idxs = []
        for idx, seg in enumerate(self.segmentation_stack):
            if np.sum(seg * inverse_mask) == 0:
                idxs.append(idx)
        self.remove_all_other_indexes(idxs)

    @staticmethod
    def load(path, protocol=4, njobs=1, pool=None, refind_neighbours=False, type_="Analyser", rewrite_dir=False, dataset_path=None, index=None):
        """
        Loads in the specified type_

  

        Args:
            path (Path): Path to load.
            protocol (int, optional): Which pickle protocol to use
            njobs (int, optional): The number of jobs to run in parallel. Defaults to 1.
            pool (multiprocessing.Pool, optional): A pool of worker processes for 
                parallel computation. If None, the computation will be done sequentially. 
                Defaults to None.
            refind_neighbours (bool, optional): redundant
            type_ (string, optional): which type of object to load in. has to be one of Analyser, Json, Wrapper, csv or Segmentation
            rewrite_dir (bool, optional): whether to rewrite the wrapper directory
            dataset_path: (Path, optional): Path of the belonging dataset
            index: (None, list of int, optional): Which indexes to remove

        Returns:
            analyser (List, type_) : single objet of type_ or list of type_
        """
        type_list = ["Analyser", "Json", "Wrapper", "csv", "Segmentation"]
        if type_ not in type_list:
            raise ValueError(f"Parameter type_ ({type_}) is not in {type_list}")
        if isinstance(path, (str, Path)):

            wrapper = AnalyserWrapper.from_path(path)
            if rewrite_dir and not type_ == "Analyser":
                wrapper.rewrite()
            if type_ == "Analyser":
                analyser = wrapper.analyser

                if dataset_path is not None:
                    analyser:Analyser 
                    analyser.dataset_path = dataset_path
                    if index is not None:
                        analyser.remove_all_other_indexes(index)


                analyser.segmentation_path = wrapper.directory / "segmentation.npz"
                if rewrite_dir:
                    wrapper.rewrite()
                return analyser
            elif type_ == "Wrapper":
                return wrapper
            elif type_ == "Json":
                return wrapper.json
            elif type_ == "csv":
                return wrapper.csv
            elif type_ == "Segmentation":
                return wrapper.segmentation
        elif isinstance(path, (list, tuple)):
            if index is None:
                index = [None for _ in range(len(path))]
            if njobs <= 1 and pool is None:
                
                return [Analyser.load(element, protocol, 0, None, refind_neighbours, type_,rewrite_dir, dataset_path, idx) for idx, element in zip(index, path)]
            elif pool is not None:
                results = [pool.apply_async(Analyser.load, args=[element, protocol, 0, None, refind_neighbours, type_,rewrite_dir, dataset_path, idx]) for idx, element in zip(index, path)]
                return [res.get() for res in results]
            else:
                with mp.get_context("spawn").Pool(njobs) as pool:
                    results = [pool.apply_async(Analyser.load, args=[element, protocol, 0, None, refind_neighbours, type_,rewrite_dir, dataset_path, idx]) for idx, element in zip(index, path)]
                    return [res.get() for res in results]
        else:
            raise AttributeError(f"Cannot load Analyser from {type(path)} type.")


    @staticmethod
    def load_generator(path, protocol=4, njobs=1, pool=None,refind_neighbours=True, type_="Analyser", rewrite_dir=False):
        type_list = ["Analyser", "Json", "Wrapper", "csv"]
        if type_ not in type_list:
            raise ValueError(f"Parameter type_ ({type_}) is not in {type_list}")
        if isinstance(path, (str, Path)):
            yield Analyser.load(path, rewrite_dir=rewrite_dir, type_=type_)
            
        elif isinstance(path, (list, tuple, set)):
            
            if njobs <= 1 and pool is None:

                for element in path:
                    yield Analyser.load(element, protocol, 0, None, refind_neighbours)
                
            elif pool is not None:
                results = [pool.apply_async(Analyser.load, args=[element, protocol, 0, None, refind_neighbours,type_,rewrite_dir ]) for element in path]
                for res in results:
                    yield res.get()
                
            else:
                with mp.get_context("spawn").Pool(njobs) as pool:
                    results = [pool.apply_async(Analyser.load, args=[element, protocol, 0, None, refind_neighbours, type_,rewrite_dir]) for element in path]
                    for res in results:
                        yield res.get()
                    
        else:
            raise AttributeError(f"Cannot load Analyser from {type(path)} type.")

    def calculateBasicAttributes(self):
        pass





import os

class AnalyserWrapper:
    def __init__(self, directory, ignore_checking=False, analyser=None):
        self.directory = Path(directory)
        self.analyser_ = analyser
        if not ignore_checking:
            self.check_files()
    
    def check_files(self):
        if not self.valid_dir(self.directory):
            raise FileNotFoundError

            
    @staticmethod
    def from_analyser(analyser, overwrite=False):

        analyser_path = Path(analyser.dataset_path / (analyser.micrograph_path.stem + ".pickle"))
        
        directory = analyser_path.parent / analyser_path.stem
        if directory.exists() and not overwrite:
            raise FileExistsError(directory)
        dirs = [directory / "thumbnails" / "micrograph", directory / "thumbnails" / "segmentation"]
        for d in dirs:
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
        

        if analyser_path.exists():
            os.remove(analyser_path)

        new_analyser_path = directory / "analyser.pickle"

        for membrane in analyser.membranes:
            membrane :Membrane
            membrane.analyser_ = None
            for p in membrane.point_list:
                p.neighbourhood_points = None
                p.thickness_profile = None
                # p.thickness_profile["profile"] = []
                # p.thickness_profile["unsmoothed"] = []
                # if "smoothed" in p.thickness_profile:
                #     p.thickness_profile["smoothed"] = []
        
        
        # shutil.move(analyser.segmentation_path, directory / "segmentation.npz")
        if not isinstance(analyser.segmentation_stack, sparse.COO):
            analyser.segmentation_stack = sparse.as_coo(analyser.segmentation_stack)
        sparse.save_npz(directory / "segmentation.npz", analyser.segmentation_stack)
        analyser.segmentation_path = directory / "segmentation.npz"
        with open(new_analyser_path, "wb") as f:
            pickle.dump(analyser, f)
        


        for membrane in analyser.membranes:
            membrane :Membrane
            membrane.analyser_ = analyser
        AnalyserWrapper.save_membrane(analyser, directory)

        AnalyserWrapper.save_points(analyser, directory)


        
        

        return AnalyserWrapper(directory, analyser=analyser)



    @staticmethod
    def valid_dir(directory):
        directory = Path(directory)
        if not directory.is_dir():
            return False
        files = [directory / "analyser.pickle", directory / "membranes.csv", directory / "points.json", directory / "segmentation.npz"]
        dirs = [directory / "thumbnails" / "micrograph", directory / "thumbnails" / "segmentation"]
        for file in files:
            if not file.exists():
                return False
        for d in dirs:
            if not d.exists():
                return False
        return True


    @staticmethod
    def valid_analyser_path(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        if path.suffix != ".pickle":
            raise ValueError(f"{path.suffix} is not .pickle")
        return AnalyserWrapper.load_analyser(path)


    @staticmethod
    def save_points(analyser, directory):
        points_json = {}
        for m in analyser.membranes:
            curvatures = [float(p.curvature) if p.curvature is not None else None for p in m.point_list ]
            thicknesses = [float(p.thickness) if p.thickness is not None else None for p in m.point_list]
            points_json[m.membrane_idx] = {"Thickness":thicknesses, "Curvature":curvatures}
        with open(directory / "points.json", "w") as f:
            json.dump(points_json, f)

    @staticmethod
    def save_membrane(analyser:Analyser, directory):
        def low_pass_filter(image, cutoff_frequency):
            # Compute the 2D Fourier transform of the image
            image_fft = fft2(image)
            
            # Create a meshgrid of frequency coordinates
            freq_x = np.fft.fftfreq(image.shape[1], 1)
            freq_y = np.fft.fftfreq(image.shape[0], 1)
            freq_meshgrid = np.meshgrid(freq_x, freq_y)
            frequencies = np.sqrt(freq_meshgrid[0]**2 + freq_meshgrid[1]**2)

            
            # Apply the low-pass filter in the Fourier domain
            image_fft_filtered = image_fft * (frequencies <= cutoff_frequency)

            
            # Compute the inverse Fourier transform to obtain the filtered image
            filtered_image = np.real(ifft2(image_fft_filtered))
            
            return filtered_image
        a_attributes = []
        headers = ["Circumference", "Diameter", "Area", "Shape", "Shape probability", "Thickness", "Closed", "Min thickness", "Max thickness", "Mean thickness",
                   "Min curvature", "Max curvature", "Is probably ice", "Is enclosed","Enclosed distance", "Index"]
        seg_image = analyser.segmentation_stack

        if len(analyser.membranes) > 0:
            try:

                micrograph = analyser.getResizedMicrograph()

                # micrograph = low_pass_filter(micrograph, 0.1)
            except Exception as e:
                micrograph = None
            for m in analyser.membranes:
                a_attributes.append([m.length, m.diameter,m.area, m.shape, m.shape_probability, m.thickness, m.is_circle, m.min_thickness, m.max_thickness,m.mean_thickness,
                                    m.min_curvature, m.max_curvature, m.isIceContamination,len(m.is_enclosed_in) > 0,None if len(m.is_enclosed_in) == 0 else np.min([i[1] for i in m.is_enclosed_in]),m.membrane_idx ])
                if micrograph is not None:
                    thumbnail = m.get_thumbnail(micrograph)
                    thumbnail.save(directory / "thumbnails" / "micrograph" / f"{m.membrane_idx}.png")
                seg_thumbnail = m.get_seg_thumbnail(seg_image)
                seg_thumbnail.save(directory / "thumbnails" / "segmentation" / f"{m.membrane_idx}.png")
        df = pd.DataFrame(a_attributes, columns=headers)
        df.to_csv(directory/ "membranes.csv", index=False)


    @staticmethod
    def from_path(analyser_path, rewrite_dir=False):
        analyser_path = Path(analyser_path)

        if rewrite_dir:
            try:
                AnalyserWrapper.valid_analyser_path(analyser_path)
                
            except:
                rewrite_dir = False
        if not rewrite_dir and AnalyserWrapper.valid_dir(analyser_path):
            return AnalyserWrapper(analyser_path)
        if not rewrite_dir and AnalyserWrapper.valid_dir(Path(analyser_path).parent):
            return AnalyserWrapper(Path(analyser_path).parent)
        if not rewrite_dir and AnalyserWrapper.valid_dir(analyser_path.parent / analyser_path.stem):
            return AnalyserWrapper(analyser_path.parent / analyser_path.stem)
        try:
            analyser = AnalyserWrapper.valid_analyser_path(analyser_path)
        except Exception as e:

            raise e
        analyser_path = Path(analyser_path)
        directory = analyser_path.parent / analyser_path.stem
        dirs = [directory / "thumbnails" / "micrograph", directory / "thumbnails" / "segmentation"]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        shutil.move(analyser_path, directory / "analyser.pickle")
        # shutil.move(analyser.segmentation_path, directory / "segmentation.npz")
        if not isinstance(analyser.segmentation_stack, sparse.COO):
            analyser.segmentation_stack = sparse.as_coo(analyser.segmentation_stack)
        sparse.save_npz(directory / "segmentation.npz", analyser.segmentation_stack)
        analyser.segmentation_path = directory / "segmentation.npz"
        with open(directory / "analyser.pickle", "wb") as f:
            pickle.dump(analyser, f)
        AnalyserWrapper.save_membrane(analyser, directory)

        AnalyserWrapper.save_points(analyser, directory)

        return AnalyserWrapper(directory, analyser=analyser)
    

    def rewrite(self):
        analyser = self.analyser

        analyser_path = Path(self.directory)
        directory = analyser_path.parent / analyser_path.stem
        dirs = [directory / "thumbnails" / "micrograph", directory / "thumbnails" / "segmentation"]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        
        AnalyserWrapper.save_membrane(analyser, directory)
        if not isinstance(analyser.segmentation_stack, sparse.COO):
            analyser.segmentation_stack = sparse.as_coo(analyser.segmentation_stack)
        sparse.save_npz(directory / "segmentation.npz", analyser.segmentation_stack)
        AnalyserWrapper.save_points(analyser, directory)
        with open(directory / "analyser.pickle", "wb") as f:
            pickle.dump(analyser, f)


    @staticmethod
    def load_analyser(path, ):
        with open(path, "rb") as f:
            analyser:Analyser = CustomUnpickler(f).load()
        
        for membrane in analyser.membranes:
            membrane.analyser_ = analyser
            for p in membrane.point_list:
                p.thickness_profile = {"profile":[], "fp":None, "sp":None, "middle_idx":None}
                p.neighbourhood_points = [[],[]]
        return analyser

    @property
    def analyser(self):
        if self.analyser_ is None:
            path = self.directory / "analyser.pickle"
            self.analyser_ = self.load_analyser(path)
        return self.analyser_


    @property
    def segmentation(self):
        path = self.directory / "segmentation.npz"
        return sparse.load_npz(path)

    @property
    def json(self):
        points_json = {}
        with open(self.directory / "points.json", "r") as f:
            points_json["Points"] = json.load( f)
        points_json["Micrograph"] = self.name
        return points_json


    @property
    def name(self):
        return self.directory.name

    @property
    def csv(self):
        path = self.directory / "membranes.csv"
        df = pd.read_csv(path,header=0)

        df["Micrograph"] = self.name
        return df

    def remove(self):
        micrograph_path = self.directory / "thumbnails"/ "micrograph"
        segmentation_path = self.directory / "thumbnails"/ "segmentation"
        thumbnail_path = self.directory / "thumbnails"
        membranes_path = self.directory / "membranes.csv"
        points_path = self.directory / "points.json"
        analyser_path =  self.directory / "analyser.pickle"
        whole_segmentation_path = self.directory / "segmentation.npz"
        shutil.rmtree(micrograph_path)
        shutil.rmtree(segmentation_path)
        shutil.rmtree(thumbnail_path)
        os.remove(membranes_path)
        os.remove(points_path)
        os.remove(analyser_path)
        os.remove(whole_segmentation_path)
        if len(os.listdir(self.directory)) == 0:
            shutil.rmtree(self.directory)
        else:
            print(list[os.listdir(self.directory)])
        


    def get_thumbnails(self, idxs=None):
        thumbnails = {}
        segmentations = {}
        if isinstance(idxs, int):
            idxs = [idxs]
        for tn in os.listdir(self.directory / "thumbnails"/ "micrograph"):
            micrograph_file = self.directory / "thumbnails"/ "micrograph" / tn
            segmentation_file = self.directory / "thumbnails"/ "segmentation" / tn
            idx  = micrograph_file.stem
            if idxs is not None:
                if int(idx) not in idxs:
                    continue
            idx = int(idx)
            thumbnails[idx] = Image.open(micrograph_file)
            segmentations[idx] = Image.open(segmentation_file)
        return thumbnails, segmentations