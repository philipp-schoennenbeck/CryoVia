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
            # if np.isclose(np.sum(tangent**2), 0):
            #     print(tangent, normal, first_point, second_point, self.idx, self.membrane_idx, self.neighbourhood_points)
            self.normal_vector = normal / np.sqrt(np.sum(normal ** 2))
        else:
            normal = self.coordinates_yx - self.circle_center_yx
            self.normal_vector = normal / np.sqrt(np.sum(normal**2))
            tangent = np.array([normal[1], -normal[0]])
            self.tangent_vector = tangent / np.sqrt(np.sum(tangent ** 2))
        


    # def estimate_persistence_length(self, pixel_spacing):
    #     """
    #     Calculates the persistence length of the edge
    #     """

    #     # L --> Length of the contour of the last edges in the neighbourhood
    #     # R --> Distance between the last edges in the neighbourhood
    #     first, second = self.get_last_of_neighborhood()
    #     L = first[1] +  second[1]
    #     R = np.sum((np.array(first[2]) - np.array(second[2]))**2)**0.5 * pixel_spacing



    #     #Calculate the flexibility --> persistence_length = 1/flexibility
    #     if 2 * (R / L)**2 - 1 < 0:
    #         flexibility = 1
    #     else:
    #         flexibility = (-np.log(2 * (R / L)**2 - 1 )) / L
    #     if flexibility == 0:
 
    #         self.persistence_length = 0
    #     else:
    #         # Calculate if the persistence length should be negative or positive --> direction of curvature
    #         # This is done by finding the crossing of the line between the last edges and the normal
    #         sign = self.is_point_right_of_tangent()

    #         if sign:
    #             sign = -1    
    #         else:
    #             sign = 1
            
    #         self.persistence_length = 1/flexibility*sign
        
    #     return self.persistence_length
    
    def is_center_right(self):
        b = self.tangent_vector
        p = np.float128(self.circle_center_yx) - np.float128(self.coordinates_yx)
        if np.all(np.isclose(p,0)):
            return False
        cross_product = np.cross(b,p)
        if cross_product >= 0:
            return True
        return False


    # def is_point_right_of_tangent(self, p=None):
    #     first, second = self.get_last_of_neighborhood()
    #     b = first[2] - second[2]
    #     if p is None:
    #         p = self.coordinates_yx - second[2]

    #     cross_product = np.cross(b,p)
    #     if cross_product >= 0:
    #         return True
    #     return False

    def dot_product(self, p):
        p_vector = p - self.coordinates_yx
        return np.dot(self.tangent_vector, p_vector)

    # def set_circle_center(self, radius):
    #     normal = self.normal_vector / np.linalg.norm(self.normal_vector)
    #     circle = self.coordinates_yx + normal * radius


    def in_polygon(self, polygon, save_path=None):
        p = np.float128(self.circle_center_yx) - np.float128(self.coordinates_yx)
        p = p / np.linalg.norm(p)
        pt = (self.coordinates_yx + p*2).astype(np.int32)
        pt = np.clip(pt, (0,0), [i-1 for i in polygon.shape])
        # if save_path is not None:
        #     dummy_image = copy.copy(polygon)
        #     dummy_image[pt[0], pt[1]] = 2
        #     dummy_image[self.coordinates_yx[0],self.coordinates_yx[1]] = 3
        #     plt.imsave(save_path / f"{self.label}_{self.index}_{pt[0]}_{pt[1]}.png", dummy_image)
        if polygon[pt[0], pt[1]] > 0:
            return True
        return False



    # def is_this_point_same_side_as_circle_centre(self, point_coords):
    #     point_vector =  point_coords - self.coordinates_yx
    #     circle_vector = self.circle_center_yx -  self.coordinates_yx
        
    #     cross_product_point = np.cross(self.tangent_vector, point_vector)
    #     cross_product_circle = np.cross(self.tangent_vector, circle_vector)
    #     if cross_product_point * cross_product_circle > 0:
    #         return True

    #     return False

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


    # def get_normal_from_dist_neighbour(self, dist):
    #     first, last = self.get_last_of_neighborhood()
        
    #     if first[1] < dist:
    #         point_one = first[2]
    #     else:
    #         for p in self.neighbourhood_points[0]:
    #             if p[1] < dist:
    #                 point_one = p[2]
    #             else:
    #                 break

    #     if last[1] < dist:
    #         point_two = last[2]
    #     else:
    #         for p in self.neighbourhood_points[1]:
    #             if p[1] < dist:
    #                 point_two = p[2]
    #             else:
    #                 break
        
    #     normal_vector = point_two - point_one
    #     normal_vector = normal_vector/np.linalg.norm(normal_vector) 

    #     normal_vector = np.array([normal_vector[1], -normal_vector[0]])
    #     if abs(self.angle(normal_vector, self.normal_vector)) > np.pi /2:
    #         normal_vector *= -1
    #     return normal_vector 

    


    def angle(self, a,b):
        a = a/np.linalg.norm(a)
        b = b/np.linalg.norm(b)
        dot = np.clip(np.dot(a,b),-1,1)

        angle = np.arccos(dot)
        return angle



            
            
            # ax.plot(np.arange(len(self.thickness_profile))* pixel_spacing, self.thickness_profile,  c="b")

            # if self.fp_sp is not None:
            #         ax.vlines([self.fp_sp[0] *pixel_spacing, self.fp_sp[1]*pixel_spacing],  np.min(self.thickness_profile), np.max(self.thickness_profile), colors="b")
            # if self.label is not None:
            #     if self.arg_cor is None:
            #         arg_cor = 0
            #     else:
            #         arg_cor = self.arg_cor
            #     if path_profile is not None:

            #         pass
            #         modifier_plus = arg_cor + len(self.thickness_profile)/2 - len(path_profile)/2 

            #         ax.plot((np.arange(len(path_profile)))* pixel_spacing, path_profile, c="r")

            #         if path_first:
            #             ax.vlines([(path_first)*pixel_spacing, (path_second )*pixel_spacing], np.min(self.thickness_profile), np.max(self.thickness_profile), colors="r" )
        
        return fig, ax

    


    @property
    def coordinates(self):
        return (self.coordinates_yx[1], self.coordinates_yx[0])

    @property
    def index(self):
        return self.idx

    def __repr__(self) -> str:
        return f"y,x:\t{self.coordinates_yx}\nidx:\t{self.idx}\nlabel:\t{self.membrane_idx}\ncurvature:\t{self.curvature}\nthickness:\t{self.thickness}"