
from pathlib import Path
import numpy as np 
from matplotlib import pyplot as plt
import cv2
from scipy.spatial.distance import cdist
import pandas
from scipy.interpolate import splrep, BSpline
from scipy.interpolate.interpolate import interp1d
from PIL import Image
from cryovia.cryovia_analysis.point import Point
from scipy.ndimage import gaussian_filter
from skimage.draw import disk
import copy
from matplotlib.cm import get_cmap
from scipy.signal import savgol_filter



class Membrane(dict):
    def __init__(self, membrane_idx=None, n=1,g=None, label=None, older_path=None, create_empty=False):
        if create_empty:
            return

        self.analyser_ = g
        self.membrane_idx = membrane_idx
        self.is_circle = False
        self.step_size = g.step_size
        self.shape = None
        self.shape_probability = 0

        self.thickness_profile = []
        self.smoothed_thickness_profile = {"profile":[], "fp":None, "sp":None, "middle_idx":None}
        
        self.diameter_ = None
        self.isIceContamination = 0
        self.improved = False
        self.point_list = []

        self.is_enclosed_in = []
        self.encloses = []

        # self.extrema = {"min_x":0, "min_y":0, "max_x":0, "max_y":0}
        # self.extrema_idx = {"min_x":0, "min_y":0, "max_x":0, "max_y":0}

        self.centroid_ = None
        # self.max_distance_idx = 0
       

        self._average_curvature = None
        self._max_curvature = None
        self._min_curvature = None
        self._average_thickness = None
        self._max_thickness = None
        self._min_thickness = None
        self._thickness = None
        self.to_estimate = True
        self._min_distance_to_closest_vesicle = None
        self._distance_to_enclosing_vesicle = None
        self._area = None
        self._length = None
        # self._diameter = None
    
        self.thumbnail_ = None




    @property
    def analyser(self):
        if isinstance(self.analyser_, (str, Path)):
            raise ValueError(f"analyser_ is still string/Path")
        else:
            return self.analyser_
    
    @analyser.setter
    def analyser(self, value):
        self.analyser_ = value



    def get_first_and_last(self):
        return self[0], self[len(self)-1]


    def first(self):
        return self[0]

    def last(self):
        return self[len(self)-1]


    @property
    def n(self):
        if self.analyser is None:
            return self.step_size
        else:
            self.step_size = self.analyser.step_size
            return self.analyser.step_size


    def idxs(self, n=None):
        if n is None:
            n = self.n
        
        counter = 0
        idxs = []
        while counter <= len(self.point_list):
            try:
                _ = self.point_list[int(counter)]
                idxs.append(int(counter))
                counter += n
            except:
                break
        return idxs

    def points(self, n=None):
        if n is None:
            n = self.n
        
        counter = 0
        points = []
        while counter <= len(self.point_list):
            try:
                points.append(self.point_list[int(counter)])
                counter += n
            except:
                break
        return points

    def get_ordered_idxs(self, start=0, n=None):
        ordered_list = list(self.keys())
        ordered_list.sort()
        
        return_list_keys = [key for key in ordered_list[start:]]
        return_list_keys.extend([key for key in ordered_list[:start]])

        return_list_values = [self[key] for key in return_list_keys]

        if n is None:
            if len(self) / self.n < 25:
                self.n = max(1,int(len(self)/25))
            n = self.n
        else:
            n = max(1,n)
        
        

            
        if n != 1:
            return_list_keys_temp = [return_list_keys[int(counter*n)] for counter in range(int(len(return_list_keys)/n))]
            return_list_values_temp = [return_list_values[int(counter*n)] for counter in range(int(len(return_list_values)/n))]
            # return_list_keys_temp = return_list_keys[::n]
            # return_list_values_temp = return_list_values[::n]

            if return_list_keys_temp[-1] != return_list_keys[-1]:
                return_list_keys_temp.append(return_list_keys[-1])
                return_list_values_temp.append(return_list_values[-1])
            return_list_keys = return_list_keys_temp
            return_list_values = return_list_values_temp
        
        return return_list_keys, return_list_values


    
    def thumbnail_with_skeleton(self):
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)

        image = self.analyser.getResizedMicrograph()[y_min:y_max + 1, x_min:x_max + 1]
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        converted = image.convert("RGB")
        y -= y_min
        x -= x_min
        for y_, x_ in zip(y,x):
            converted.putpixel((x_, y_), (255,0,0))
            

        converted.thumbnail((200,200))
        
        return converted

    def get_thumbnail(self, image):
        
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)

        image = np.copy(image[y_min:y_max + 1, x_min:x_max + 1])
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        image.thumbnail((200,200))
        self.thumbnail_ = image
        return self.thumbnail_
        
    
    def get_seg_thumbnail(self, image):

        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)
        image = np.copy(image[self.membrane_idx][y_min:y_max + 1, x_min:x_max + 1].todense())

        # image -= np.min(image)
        # image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        image.thumbnail((200,200))
        return image




    @property
    def thumbnail(self):
        if self.thumbnail_ is not None:
            return self.thumbnail_
        
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)

        image = self.analyser.getResizedMicrograph()[y_min:y_max + 1, x_min:x_max + 1]
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        image.thumbnail((200,200))
        self.thumbnail_ = image
        return self.thumbnail_
        
    @property
    def seg_thumbnail(self):

        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)
        try:
            image = self.analyser.segmentation_stack[self.membrane_idx][y_min:y_max + 1, x_min:x_max + 1].todense()
        except Exception as e:
            raise e
        # image -= np.min(image)
        # image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        image.thumbnail((200,200))
        return image

    @property
    def croppedImage(self):
        
        
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.segmentation_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.segmentation_shape[1] -1, np.max(x) + 20)

        image = self.analyser.getResizedMicrograph()[y_min:y_max + 1, x_min:x_max + 1]
        image -= np.min(image)
        image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        return image

    @property
    def croppedSegmentation(self):
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.segmentation_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.segmentation_shape[1] -1, np.max(x) + 20)

        image = self.analyser.segmentation_stack[self.membrane_idx][y_min:y_max + 1, x_min:x_max + 1].todense()
        # image -= np.min(image)
        # image /= np.max(image)
        image *= 255
        image = Image.fromarray(np.uint8(image))

        return image

    @property
    def centroid(self):
        if not hasattr(self, "centroid_"):
            self.centroid_ = None
        if self.centroid_ is None:
            order = list(self.items())
            if len(order) == 0:
                self.centroid_ = None
            else:
                coords = np.array([self.analyser.skeleton_points[p].coordinates_yx for _,p in order])      
                self.centroid_ = np.mean(coords, axis=0)
        return self.centroid_
    
    @centroid.setter
    def centroid(self, value):
        self.centroid_ = value

    def new_dict(self, dict):
        super().__init__(dict)



    def find_close_points(self, max_distance=150, estimate_new_vectors=True):
        
        
        if max_distance == 0:
            return

        max_distance = max_distance / self.analyser.pixel_size
        
                

        coords = [point.coordinates_yx for point in self.point_list]
        coords_length = len(coords)
        if self.is_circle:
            coords = np.concatenate((coords, coords))
            
            
            

            res = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=-1)))
            
            if np.max(res) < max_distance * 8:
                current_max_distance = max(1,np.max(res) / 8)
            else:
                current_max_distance = max_distance


            starting_idx = np.argmax(res > current_max_distance)
        
            
            res_total = np.tile(res, len(res)).reshape(len(res), len(res))
            distance = np.abs((res_total.T - res).T)
            distance_ = distance <= current_max_distance

            for idx in range(starting_idx, starting_idx + coords_length):
                dist = distance_[idx]
                non_zero = np.nonzero(dist)[0]
            
                left_idx = non_zero[non_zero < idx]
                right_idx = non_zero[non_zero > idx]
                left = distance[idx][left_idx]
                right = distance[idx][right_idx]
                left_idx = left_idx % coords_length
                right_idx = right_idx % coords_length
                
                idx = idx % coords_length
                neighbours = [[],[]]
                
                for direction, (is_, distances) in enumerate([(left_idx[::-1],left[::-1]), (right_idx, right)]):
                    for i, d in zip(is_, distances):
                        neighbours[direction].append((i ,d*self.analyser.pixel_size, coords[i]))
                self.point_list[idx].neighbourhood_points = neighbours
                

        else:
            res = np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=-1)))
            res = np.concatenate(([0], res))

            res_total = np.tile(res, len(res)).reshape(len(res), len(res))
            distance = np.abs((res_total.T - res).T)
            distance_ = distance <= max_distance
            for counter, dist in enumerate(distance_):
                
                non_zero = np.nonzero(dist)[0]
                
                left_idx = non_zero[non_zero < counter]
                right_idx = non_zero[non_zero > counter]
                left = distance[counter][left_idx]
                right = distance[counter][right_idx]
                left_idx = left_idx % coords_length
                right_idx = right_idx % coords_length
                neighbours = [[],[]]
                idx = counter
                for direction, (is_, distances) in enumerate([(left_idx[::-1],left[::-1]), (right_idx, right)]):
                    for i, d in zip(is_, distances):
                        neighbours[direction].append((i ,d*self.analyser.pixel_size, coords[i]))
                self.point_list[idx].neighbourhood_points = neighbours


        for i in self.point_list:
            i.estimate_tangent_and_normal()



    def turn_horizontal(self):
        # def angle_between(vector_1, vector_2):

        #     unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
        #     unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
        #     dot_product = np.dot(unit_vector_1, unit_vector_2)
        #     angle = np.arccos(dot_product)
        #     return angle * -1
        def unit_vector(vector):
            """ Returns the unit vector of the vector"""
            return vector / np.linalg.norm(vector)

        def angle_between(vector1, vector2):
            """ Returns the angle in radians between given vectors"""
            v1_u = unit_vector(vector1)
            v2_u = unit_vector(vector2)
            minor = np.linalg.det(
                np.stack((v1_u[-2:], v2_u[-2:]))
            )
            if minor == 0:
                return 0

            return np.sign(minor) * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        def rotate(points, origin, angle):
            return (points - origin) * np.exp(complex(0, angle)) + origin

        def turn_horizontal(coords):
            y,x = coords.T
            distances = cdist(coords, coords)
            max_idxs = np.argmax(distances, axis=-1)
            
            max_values = distances[np.arange(0, len(coords)), max_idxs]
            
            max_idx = np.argmax(max_values)
            best_idxs =(max_idx, max_idxs[max_idx])
            pt_1 = coords[best_idxs[0]]
            pt_2 = coords[best_idxs[1]]
            
            longest_vector = pt_1 - pt_2
            
            angle = angle_between(longest_vector, np.array([0,1]))
            complex_coords = np.zeros(len(coords), dtype=np.complex128)
            complex_coords.real = y
            complex_coords.imag = x


            length_vector = np.sqrt(np.sum(longest_vector**2))
            middle = pt_2 + longest_vector/np.linalg.norm(longest_vector) * length_vector*0.5
            
            rotated_coords = rotate(complex_coords, middle[0] +middle[1]*1j, angle)
            new_coords = np.zeros((len(complex_coords),2))
            new_coords[::,0] = rotated_coords.real
            new_coords[::,1] = rotated_coords.imag
            return new_coords
        
        coords = self.coords
        y,x = turn_horizontal(coords).T
        return y,x

    def resize(self, resizeShape, landmarks=None):
        y,x = self.turn_horizontal()
        y -= np.min(y)
        x -= np.min(x)
        range_y = np.max(y) - np.min(y)
        range_x = np.max(x) - np.min(x)
        range_ = max(range_y, range_x)
        modifier = resizeShape / range_
        y *= modifier
        x *= modifier
        if landmarks is not None:
            to_interpolate = np.linspace(0,1,landmarks)
            coords = self.coords
            distance = np.cumsum( np.sqrt(np.sum( np.diff(coords, axis=0)**2, axis=1 )) )
            # rescale distance
            
            distance = np.insert(distance, 0, 0)/distance[-1]

            if self.is_circle:
                period = 1
            else:
                period = None
            x = np.interp(to_interpolate, distance, x, period=period)
            y = np.interp(to_interpolate, distance, y, period=period)
        return y,x


    def resize_curvature(self, resizeShape,landmarks):
       
        curvatures = [p.curvature for p in self.point_list if p.curvature is not None]
        if len(curvatures) == 0:
            return None

        coords = self.coords
        y,x = self.turn_horizontal()


        range_y = np.max(y) - np.min(y)
        range_x = np.max(x) - np.min(x)
        range_ = max(range_y, range_x)



        modifier = resizeShape / range_
        new_curvature = curvatures / modifier

        to_interpolate = np.linspace(0,1,landmarks)
        # y,x = coords.T
        distance = np.cumsum( np.sqrt(np.sum( np.diff(coords, axis=0)**2, axis=1 )) )
        # rescale distance
        
        distance = np.insert(distance, 0, 0)/distance[-1]

        if self.is_circle:
            period = 1
        else:
            period = None
        new_curvature = np.interp(to_interpolate, distance, new_curvature,period=period)

        return new_curvature * self.analyser.pixel_size



    def resize_contour(self, resizeShape,landmarks): 
        coords = self.coords
        y,x = self.turn_horizontal()

        range_y = np.max(y) - np.min(y)
        range_x = np.max(x) - np.min(x)
        range_ = max(range_y, range_x)

        centroid = np.array([(np.mean(y), np.mean(x))])
        distances = cdist(centroid, coords)[0]

        

        modifier = resizeShape / range_
        distances *= modifier
        # new_curvature = curvatures / modifier

        to_interpolate = np.linspace(0,1,landmarks)
        y,x = coords.T
        distance = np.cumsum( np.sqrt(np.sum( np.diff(coords, axis=0)**2, axis=1 )) )
        # rescale distance
        
        distance = np.insert(distance, 0, 0)/distance[-1]


        if self.is_circle:
            period = 1
        else:
            period = None
        distances = np.interp(to_interpolate, distance, distances,period=period)
        return distances
    
    def resize_area(self, resizeShape):
        # coords = self.coords
        # y,x = self.resize(resizeShape)
        # range_y = np.max(y) - np.min(y)
        # range_x = np.max(x) - np.min(x)
        # range_ = max(range_y, range_x)

        # centroid = np.array([(np.mean(y), np.mean(x))])
        # distances = cdist(centroid, coords)[0]
        # modifier = resizeShape / range_

        if self.is_circular:
            coords = np.array(np.array(self.resize(resizeShape)).T).astype(np.int32)
            area = cv2.contourArea(coords) * self.analyser.pixel_size ** 2
            return area

    def resize_length(self, resizeShape):
        coords = np.array(np.array(self.resize(resizeShape)).T).astype(np.int32)
        length = cv2.arcLength(coords, self.is_circular) * self.analyser.pixel_size
        return length

    def distribute_attributes_to_neighbours(self, attr =["curvature"], sigma=None):
        
        """Distribute specific attributes of points to the neighbours by interpolation
        Parameters:
        attr, list: List of attributes to distribute"""

        
        points = self.points()
        idxs = self.idxs()
        all_points = self.point_list

        
        # Get all coordinate of points of the current path
        coords = np.array([p.coordinates_yx for p in all_points])

        # Get distance of the points in order of contour from the start
        distance = np.cumsum( np.sqrt(np.sum( np.diff(coords, axis=0)**2, axis=1 )) )
        # rescale distance
        distance = np.insert(distance, 0, 0)/distance[-1]

        # Get distances only of points used for calculations (stepsize!)
        x = np.array([distance[i] for i in idxs])
        _, unique_idxs = np.unique(x, return_index=True)
        x = x[unique_idxs]
        stepped_coords = np.array([coords[idx] for idx in idxs])
        idxs = np.array(idxs)
        idxs = idxs[unique_idxs]

        # Get attributes
        attr_dict = {a:np.array([p.__getattribute__(a) for p in points]) for a in attr}
        
        for a in attr:

            
            mask = [i is not None for i in attr_dict[a]]
            x_attr = x[mask]
            if len(x_attr) <= 2:
                if hasattr(self, a):
                    setattr(self, a,None)
                    continue
            all_values = np.array(attr_dict[a], dtype=np.float32)
            attr_dict[a] = np.array(attr_dict[a][mask],dtype=np.float32)
            
            
            attr_dist = cdist(stepped_coords, coords)
            to_use_idxs = np.argmin(attr_dist, 0)
            # nones = [True if value is None else False for value in all_values]
            nones = [True if all_values[idx] is None else False for idx in to_use_idxs]

        
            
                                
            if len(attr_dict[a]) == 0:
                continue
            mean_attr = np.mean(attr_dict[a])
            f = interp1d(x_attr, attr_dict[a], "linear",bounds_error=False, fill_value=(attr_dict[a][0],attr_dict[a][-1]))

            y = f(distance)
            y[np.isnan(y)] = mean_attr
            if sigma is not None:
                if self.is_circle:
                    y = gaussian_filter(y, sigma, mode="wrap")
                else:
                    y = gaussian_filter(y, sigma, mode="nearest")
            
            y[nones] = None
            for p,val in zip(all_points, y): 
                p.__setattr__(a,val) 



    def find_max_distance_idx(self):
        
        order = list(self.items())
        coords = np.array([self.analyser.skeleton_points[p].coordinates_yx for _,p in order])
        
        distances = cdist([self.centroid], coords)[0]
        max_distance_idx = order[np.argmax(distances)][0]
        self.max_distance_idx = max_distance_idx
    

    

    @property
    def is_circular(self):
        return self.is_circle
    
    @is_circular.setter
    def is_circular(self, value):
        self.is_circle = value


    


    @property
    def diameter(self):
        
        if self.diameter_ is None:
            coords = self.coords 
            distances = cdist(coords, coords, metric="euclidean")
            self.diameter_ = np.max(distances) * self.analyser.pixel_size
        
        return self.diameter_


    @diameter.setter
    def diameter(self, value):
        self.diameter_ = value

    def get_diameter_points(self):
        order_keys, order_values = self.get_ordered_idxs(n=1)
        coords = np.array([self.analyser.skeleton_points[p].coordinates_yx for p in order_values])
        distances = cdist(coords, coords, metric="euclidean")
        idxs = np.unravel_index(distances.argmax(), distances.shape)
        return (self.analyser.skeleton_points[order_values[idxs[0]]], self.analyser.skeleton_points[order_values[idxs[1]]])



    
    def set_curvatures(self):
        _, idxs = self.get_ordered_idxs(n=1)
        curvatures = [self.analyser.skeleton_points[idx].curvature for idx in idxs if self.analyser.skeleton_points[idx].curvature is not None]
        if len(curvatures) > 0:
            self.min_curvature = np.min(curvatures)
            self.max_curvature = np.max(curvatures)
            self.average_curvature = np.mean(curvatures)

    def set_thicknesses(self):
        _, idxs = self.get_ordered_idxs(n=1)
        thicknesses = [self.analyser.skeleton_points[idx].thickness for idx in idxs if self.analyser.skeleton_points[idx].thickness is not None]
        if len(thicknesses) > 0:
            self.min_thickness = np.min(thicknesses)
            self.max_thickness = np.max(thicknesses)
            self.average_thickness = np.mean(thicknesses)



    @property
    def average_curvature(self):
        if self._average_curvature is None:
            self.set_curvatures()
        return self._average_curvature

    @average_curvature.setter
    def average_curvature(self, value):
        self._average_curvature = value

    @property
    def max_curvature(self):
        curvature = [p.curvature for p in self.points() if p.curvature is not None]
        if len(curvature) == 0:
            return None
        return np.max(curvature)
        

    @property
    def min_curvature(self):
        curvature = [p.curvature for p in self.points() if p.curvature is not None]
        if len(curvature) == 0:
            return None
        return np.min(curvature)
    
 
    @property
    def max_thickness(self):
        if self.thickness is None:
            return None
        thickness = [p.thickness for p in self.points() if p.thickness is not None]
        if len(thickness) == 0:
            return None
        return np.max(thickness)
    
    @property
    def mean_thickness(self):
        if self.thickness is None:
            return None
        thickness = [p.thickness for p in self.points() if p.thickness is not None]
        if len(thickness) == 0:
            return None
        return np.mean(thickness)

    @property
    def min_thickness(self):
        if self.thickness is None:
            return None
        thickness = [p.thickness for p in self.points() if p.thickness is not None]
        if len(thickness) == 0:
            return None
        return np.min(thickness)


    @property
    def thickness(self):
        return self._thickness

    @thickness.setter
    def thickness(self, value):
        self._thickness = value
    
    @property
    def min_distance_to_closest_vesicle(self):
        if self._min_distance_to_closest_vesicle is None:
            distances = []
            self_coords = self.coords
            for membrane in self.analyser.membranes:
           
                
                if membrane.membrane_idx == self.membrane_idx:
                    continue 
                coords = membrane.coords
                dist = cdist(self_coords, coords)
                distances.append(np.min(dist))
            if len(distances) > 0:
                self._min_distance_to_closest_vesicle = np.min(distances) *self.analyser.pixel_size
                

        return self._min_distance_to_closest_vesicle

    @min_distance_to_closest_vesicle.setter
    def min_distance_to_closest_vesicle(self, value):
        self._min_distance_to_closest_vesicle = value
    
    @property
    def distance_to_enclosing_vesicle(self):
        if self._distance_to_enclosing_vesicle is None:
            if len(self.is_enclosed_in) > 0:
                distances = []
                self_coords = self.coords
                for membrane in self.analyser:
                    if membrane.membrane_idx in self.is_enclosed_in:
                        
                        
                        coords = membrane.coords
                        dist = cdist(self.coords, coords)
                        distances.append(np.min(dist))
                if len(distances) > 0:
                    self._distance_to_enclosing_vesicle = np.min(distances) *self.analyser.pixel_spacing
        return self._distance_to_enclosing_vesicle

    @distance_to_enclosing_vesicle.setter
    def distance_to_enclosing_vesicle(self, value):
        self._distance_to_enclosing_vesicle = value
    
    @property
    def area(self):
        if self._area is None:
            if self.is_circular:
                coords = self.coords
                area = cv2.contourArea(coords) * self.analyser.pixel_size ** 2
                self._area = area 
        return self._area

    @area.setter
    def area(self, value):
        self._area = value
    

    @property
    def circularity(self):
        area  = self.area
        
        if area is None:
            return None
        perimeter = self.length
        circularity = 4*np.pi*(area/(perimeter*perimeter))
        return circularity


    @property
    def length(self):
        if self._length is None:
            coords = self.coords
            if self.is_circular:
                coords = np.concatenate((coords, coords[0:1]))
            length = np.sum(np.sqrt(np.sum(np.diff(coords, axis=0)**2,axis=1)))
            self._length = length *self.analyser.pixel_size
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
    
    @property
    def enclosed_by(self):
        return len(self.is_enclosed_in)

    
    def getResizedCoords(self, ratio, pixelSize=None, smoothContour=False):
        def smooth(x, wl, method="interp"):
            return savgol_filter(x, min(len(x), wl),3,mode=method)
        coords = self.coords

        coords = coords * ratio

        y,x = coords.T
        if pixelSize is None:
            pixelSize = self.analyser.pixel_size
        smooth_int = int(400/ pixelSize)
        # smooth_int = min(smooth_int, len(coords) // 4)
        smooth_int = min(smooth_int, int(len(x) * 0.1))
        if smoothContour:
            y = smooth(y, smooth_int, "wrap")
            x = smooth(x, smooth_int, "wrap")



        points = np.arange(len(y)) * ratio

        number_of_points = int(len(y) * ratio * 4)

        to_interp_points = np.linspace(0, int(len(y) * ratio - ratio), number_of_points)

        ty,cy,ky = splrep(points, y,per=self.is_circle)
        y_spline = BSpline(ty, cy, ky,)
        interp_y = y_spline(to_interp_points).astype(np.int32)

        tx,cx,kx = splrep(points, x,per=self.is_circle)
        x_spline = BSpline(tx, cx, kx,)
        interp_x = x_spline(to_interp_points).astype(np.int32)

        coords = np.array(np.array([interp_y, interp_x]).T)
        coords, ind = np.unique(coords, axis=0, return_index=True)
        coords = coords[np.argsort(ind)]
        return coords
    
    @property
    def coords(self):
        return np.array([p.coordinates_yx for p in self.point_list])
    


    def getThicknessMap(self, max_width, get_individual=False):

        
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)

        image = self.analyser.segmentation_stack[self.membrane_idx][y_min:y_max + 1, x_min:x_max + 1].todense()
        image = np.zeros_like(image)
        output_img = np.ones(image.shape) * np.nan

        
        y,x = self.coords.T
        y = np.array(y) - y_min
        x = np.array(x) - x_min
        disk_y, disk_x = disk((0, 0),radius=(max_width/self.analyser.pixel_size)/2, )
        thicknesses = [p.thickness for p in self.point_list]

        for y_,x_ in zip(y, x):
            current_disk_y = disk_y + y_
            current_disk_x = disk_x + x_

            idxs_to_use = np.where((current_disk_y >= 0) & (current_disk_y < image.shape[0]) & (current_disk_x >= 0) & (current_disk_x < image.shape[1]))
            current_disk_y = current_disk_y[idxs_to_use]
            current_disk_x = current_disk_x[idxs_to_use]
            image[current_disk_y, current_disk_x] = 1

        yy,xx = np.nonzero(image)
        
        
        for current_x,current_y in zip(xx,yy):
    
            min_idx = ((x - current_x)**2 + (y - current_y)**2).argmin()
            
            output_img[current_y, current_x] = thicknesses[min_idx]
            

            
        current_cmap = copy.copy(get_cmap(name="autumn").reversed())
        output_img -= np.nanmin(output_img)
        output_img = output_img/np.nanmax(output_img)
        
        current_cmap.set_bad(color='black')
        if get_individual:
            return output_img, current_cmap
        
        output_img = current_cmap(output_img)
        return output_img


    def getCurvatureMap(self, max_width, get_individual=False):

        
        y,x = self.coords.T

        y_min = max(0, np.min(y) - 20)
        y_max = min(self.analyser.micrograph_shape[0] -1, np.max(y) + 20)

        x_min = max(0, np.min(x) - 20)
        x_max = min(self.analyser.micrograph_shape[1] -1, np.max(x) + 20)

        image = self.analyser.segmentation_stack[self.membrane_idx][y_min:y_max + 1, x_min:x_max + 1].todense()
        image = np.zeros_like(image)
        output_img = np.ones(image.shape) * np.nan

        
        y,x = self.coords.T
        y = np.array(y) - y_min
        x = np.array(x) - x_min
        disk_y, disk_x = disk((0, 0),radius=(max_width/self.analyser.pixel_size)/2, )
        curvatures = [p.curvature for p in self.point_list]

        for y_,x_ in zip(y, x):
            current_disk_y = disk_y + y_
            current_disk_x = disk_x + x_

            idxs_to_use = np.where((current_disk_y >= 0) & (current_disk_y < image.shape[0]) & (current_disk_x >= 0) & (current_disk_x < image.shape[1]))
            current_disk_y = current_disk_y[idxs_to_use]
            current_disk_x = current_disk_x[idxs_to_use]
            image[current_disk_y, current_disk_x] = 1

        yy,xx = np.nonzero(image)
        
        
        for current_x,current_y in zip(xx,yy):
    
            min_idx = ((x - current_x)**2 + (y - current_y)**2).argmin()
            
            output_img[current_y, current_x] = curvatures[min_idx]
            

            
        current_cmap = copy.copy(get_cmap(name="seismic"))
        output_img = output_img/np.nanmax(np.abs(output_img)) / 2 + 0.5
        # output_img -= np.nanmin(output_img)
        # output_img = output_img/np.nanmax(output_img)
        
        current_cmap.set_bad(color='black')
        
        if get_individual:
            return output_img, current_cmap
        
        output_img = current_cmap(output_img)
        return output_img