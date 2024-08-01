from os import path
import PIL
from cv2 import circle
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from scipy.signal import correlate
from scipy import signal
import copy




class Point:
    def __init__(self, x,y, idx, membrane_idx, create_empty=False):
        if create_empty:
            return
        
        
        self.coordinates_yx = np.array([y,x])
        self.idx = idx
        self.curvature_radius = None
        self.curvature = None
        self.circle_center_yx = None
        self.membrane_idx = membrane_idx

        # self.neighbourhood_points = [[],[]]
        self.tangent_vector = None
        self.normal_vector = None

        self.neighbourhood_points = [[],[]]
        
        self.thickness = None
        self.thickness_profile = {"profile":[], "fp":None, "sp":None, "middle_idx":None}

        


    def __eq__(self, o) -> bool:
        return self.coordinates_yx[0] == o.coordinates_yx[0] and self.coordinates_yx[1] == o.coordinates_yx[1]

    def __mul__(self, other) -> float:
        return np.sum((np.array(self.coordinates_yx) - np.array(other.coordinates_yx))**2)**0.5

    

    def get_coords_of_neighbourhood(self, max_dist=-1):
        
        y = [i[2][0] for i in self.neighbourhood_points[0] if i[1] < max_dist]
        y.extend([i[2][0] for i in self.neighbourhood_points[1] if i[1] < max_dist])
        x = [i[2][1] for i in self.neighbourhood_points[0] if i[1] < max_dist]
        x.extend([i[2][1] for i in self.neighbourhood_points[1] if i[1] < max_dist])

        x.append(self.coordinates_yx[1])
        y.append(self.coordinates_yx[0])
        x = np.array(x)
        y = np.array(y)

        return y,x

    
    def get_idxs_of_neighbourhood(self, max_dist=-1):
        
        
        idxs = [i[0] for i in self.neighbourhood_points[0] if i[1] < max_dist]
        idxs.extend([i[0] for i in self.neighbourhood_points[1] if i[1] < max_dist])

        idxs.append(self.idx)
        return idxs

    def estimate_tangent_and_normal(self):
        if self.curvature == 0 or self.curvature is None or self.circle_center_yx is None:
        # if True:        

            first_point, second_point = self.get_last_of_neighborhood()


            tangent = first_point[2] -second_point[2]
            self.tangent_vector = tangent/ np.sqrt(np.sum(tangent ** 2))
            
            normal = np.array([tangent[1], -tangent[0]])

            self.normal_vector = normal / np.sqrt(np.sum(normal ** 2))
        else:
            normal = self.coordinates_yx - self.circle_center_yx
            self.normal_vector = normal / np.sqrt(np.sum(normal**2))
            tangent = np.array([normal[1], -normal[0]])
            self.tangent_vector = tangent / np.sqrt(np.sum(tangent ** 2))
        


    
    def is_center_right(self):
        b = self.tangent_vector
        p = np.float64(self.circle_center_yx) - np.float64(self.coordinates_yx)
        if np.all(np.isclose(p,0)):
            return False
        cross_product = np.cross(b,p)
        if cross_product >= 0:
            return True
        return False



    def dot_product(self, p):
        p_vector = p - self.coordinates_yx
        return np.dot(self.tangent_vector, p_vector)




    def in_polygon(self, polygon, save_path=None):
        p = np.float64(self.circle_center_yx) - np.float64(self.coordinates_yx)
        p = p / np.linalg.norm(p)
        pt = (self.coordinates_yx + p*2).astype(np.int32)
        pt = np.clip(pt, (0,0), [i-1 for i in polygon.shape])

        if polygon[pt[0], pt[1]] > 0:
            return True
        return False



    def get_last_of_neighborhood(self):
        if len(self.neighbourhood_points[0]) > 0:
            first = self.neighbourhood_points[0][-1]
        else:
            first = [self.idx, 0, self.coordinates_yx]

        if len(self.neighbourhood_points[1]) > 0:
            second = self.neighbourhood_points[1][-1]
        else:
            second = [self.idx, 0, self.coordinates_yx]

        return first, second



    


    def angle(self, a,b):
        a = a/np.linalg.norm(a)
        b = b/np.linalg.norm(b)
        dot = np.clip(np.dot(a,b),-1,1)

        angle = np.arccos(dot)
        return angle

    


    @property
    def coordinates(self):
        return (self.coordinates_yx[1], self.coordinates_yx[0])

    @property
    def index(self):
        return self.idx

    def __repr__(self) -> str:
        return f"y,x:\t{self.coordinates_yx}\nidx:\t{self.idx}\nlabel:\t{self.membrane_idx}\ncurvature:\t{self.curvature}\nthickness:\t{self.thickness}"