
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import mrcfile
from skimage.morphology import skeletonize, binary_dilation
import cryovia.cryovia_analysis.sknw as sknw
from scipy.ndimage import label
from skimage.measure import label as skilabel 
import networkx as nx
import itertools as it
from scipy.spatial.distance import cdist
from skimage.draw import line_nd
from cv2 import line
from skimage.draw import disk,circle_perimeter
import copy
from scipy import optimize
import multiprocessing as mp
from scipy.sparse.csgraph import dijkstra
import sparse
import getpass




class connection_value:
    order = {"same":0, "lineness":1, "center_dist":2}
    def __init__(self, connection, value, node_idx, connection_type, direction, degree) -> None:
        self.connection = connection
        self.value = value
        self.node_idx = node_idx
        self.type = connection_type
        self.edge_direction = direction
        # self.degree = degree
        self.degree = 1

    def __eq__(self, other) -> bool:
        if self.type == "same" and other.type == "same":
            return True
        else:
            if self.degree != other.degree:
                return False
            return self.type == other.type and self.value == other.value


    def __gt__(self, other):
        if self.type == "same" and other.type == "same":
            return False
        if self.degree == other.degree:
            if self.order[self.type] > self.order[other.type]:
                return True
            if self.order[self.type] < self.order[other.type]:
                return False
            return self.value > other.value
        return self.degree > other.degree 
    
    def __repr__(self) -> str:
        return f"{self.type} {self.value} {self.node_idx} {self.connection} {self.degree}"
    

def calc_tangent(nr_of_neighbours, pts):
    # nr_of_neighbours = max(min(len(self.pts), self.neighbours // 4),1)
    # nr_of_neighbours = max(neighbours // 4,1)
    # nr_of_neighbours = 5

    neighbourhood_pts = (pts[:nr_of_neighbours], pts[-nr_of_neighbours:][::-1])

    # vector_1 = neighbourhood_pts[0][0] - neighbourhood_pts[0][-1]
    # vector_1 = vector_1 / np.linalg.norm(vector_1)

    # vector_2 = neighbourhood_pts[1][0] - neighbourhood_pts[1][-1]
    # vector_2 = vector_2 / np.linalg.norm(vector_2)
    
    vector_1 = fit_line(neighbourhood_pts[0],start=None,get_vector=True, verbose=False)
    vector_2 = fit_line(neighbourhood_pts[1],start=None,get_vector=True, verbose=False)

    vector_1_approx = neighbourhood_pts[0][-1 ] - pts[0] 
    vector_1_approx = vector_1_approx / np.linalg.norm(vector_1_approx)
    vector_2_approx = neighbourhood_pts[1][-1 ] - pts[-1] 
    vector_2_approx = vector_2_approx / np.linalg.norm(vector_2_approx)

    vector_1_angle = np.arccos(np.clip(np.dot(vector_1, vector_1_approx),-1,1)) 
    vector_2_angle = np.arccos(np.clip(np.dot(vector_2, vector_2_approx),-1,1))
    if vector_1_angle > np.pi*0.5 and vector_1_angle < np.pi * 1.5:
        vector_1 = vector_1 * -1 
    if vector_2_angle > np.pi*0.5 and vector_2_angle < np.pi * 1.5:
        vector_2 = vector_2 * -1 

    return vector_1, vector_2
    # self.tangents = (vector_1, vector_2)

def fit_line(pts, start, get_vector=False, verbose=False):
    def perp_dist(x,y, a,b,c):
        return np.abs(a*x+b*y+c) / np.sqrt(a**2 + b**2)

    def rotate(origin, points, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        
        rotated_points=[]
        for point in points: 
            
            px, py = point

            qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
            qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
            rotated_points.append([qx,qy])
        return np.array(rotated_points)
    rotated_points = rotate(np.mean(pts,0), pts, np.pi * 0.5)
    
    if np.std(pts[...,0]) == 0:
        if get_vector:
            return np.array([0,1])
        else:
            return 0
    if np.std(pts[...,1]) == 0:
        if get_vector:
            return np.array([1,0])
        else:
            return 0

    try:
        (a,b), res, rank, sing_v, rcond = np.polyfit(pts[:,1], pts[:,0],1,full=True)
    except Exception as e:
        raise e
    perp_error = np.sum(perp_dist(pts[:,1],pts[:,0], a,-1, b))
    own_error = np.sum((np.abs((a*pts[:,1] + b) - pts[:,0])))
    




    (a_,b_), res_, rank, sing_v, rcond = np.polyfit(rotated_points[:,1], rotated_points[:,0],1,full=True)
    perp_error_ = np.sum(perp_dist(rotated_points[:,1],rotated_points[:,0], a_,-1, b_))
    own_error_ = np.sum((np.abs((a_*rotated_points[:,1] + b_) - rotated_points[:,0])))

    if get_vector:
        if perp_error_ < perp_error:
            p_1 = np.array([0,b_])
            p_2 = np.array([1,a_+b_])
        else:
            p_1 = np.array([0,b])
            p_2 = np.array([1,a+b])
        vector = p_2-p_1
        vector = vector / np.linalg.norm(vector)
        if perp_error_ < perp_error:
            vector = np.array([vector[1], -vector[0]])
        return vector[::-1]
        
            
    else:
        return min(perp_error / len(pts),perp_error_ / len(pts))

def angle_is_appropriate(angle, max_angle=np.pi*1.3, min_angle=np.pi*0.7):
    return angle > min_angle and angle < max_angle

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y, weights=None):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    if weights is not None:
        Ri *= weights
    return Ri - Ri.mean()

def leastsq_circle(pts, weights=None):
    # coordinates of the barycenter
    x,y = pts.T
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    try:
        center, ier = optimize.leastsq(f, center_estimate, args=(x,y, weights))
    except:
        return 0, center_estimate
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    # residu   = np.sum((Ri - R)**2)
    return R, center



def draw_edge(network, edge, y=None, x=None):
    if y is None and x is None:
        edge_data = network.get_edge_data(*edge)
        y,x = edge_data["pts"].T
    
    y_min = max(0, np.min(y) - 20)
    y_max = min(network.image.shape[0]-1, np.max(y) + 20)
    x_min = max(0, np.min(x) - 20)
    x_max = min(network.image.shape[1]-1, np.max(x) + 20)

    y = y - y_min
    x = x - x_min
    new_image = np.zeros((y_max - y_min, x_max - x_min))
    new_image[y,x] = 1
    plt.figure(figsize=(15,15))
    plt.title(f"{edge}, {y_min},{y_max},{x_min},{x_max}")

    plt.imshow(new_image, cmap="gray")
    plt.show()
    

class edge_dict(dict):
    def __getitem__(self, key):
        try:
            item = super().__getitem__(key)
        except:
            item = super().__getitem__((key[1], key[0], key[2]))
        return item

class curvature_network(nx.MultiGraph):
    def __init__(self, incoming_graph_data, image, closest_coord_map,fuse_distance=10, min_length=10, line_width=3, idx=None, verbose=False, mask=None):
        super().__init__(incoming_graph_data)
        self.fuse_distance = fuse_distance
        self.idx = idx
        self.closest_coord_map = closest_coord_map
        self.image = image
        self.min_length = min_length
        self.junctions = {}
        self.custom_edges = edge_dict()
        self.line_width = line_width
        self.border_threshold = 10
        self.used_edge_keys = {}
        self.uses = {}
        self.not_allowed = {}
        if mask is not None:
            self.remove_edges_on_mask(mask)
        self.remove_small_outer_edges(verbose=verbose)
        if fuse_distance > 0:
            self.fuse_nodes()
        
    def remove_edges_on_mask(self,mask):
        edges_to_remove = []
        for u,v,k,data in self.edges(data=True, keys=True):
            pts = data["pts"]
            if np.sum(mask[pts[...,0], pts[...,1]] == 0) > 0.5 * len(pts):
                edges_to_remove.append((u,v,k))

        self.remove_edges_from(edges_to_remove)
    
    def border_edge(self,pts,u,v, border_threshold=None):
        def at_border(pt, border_threshold=None):
            if border_threshold is None:
                border_threshold = self.border_threshold
            if (pt[0] - border_threshold) <= 0 or (pt[0] + border_threshold) >= self.image.shape[0]-1:
                return True
            if (pt[1] - border_threshold) <= 0 or (pt[1] + border_threshold) >= self.image.shape[1]-1:
                return True
            return False
        a = pts[0]
        b = pts[-1]
        return at_border(a,border_threshold) and at_border(b,border_threshold)

        

    def create_instance_map(self, mask=None, only_closed=False, only_longest=False, verbose=False, label_image=None, dilation=False, non_closed_min_size=0, accept_border_edges=False):
        instances = []
        edges = []
        skeletons = []
        closed = []
        all_pts = []

        self.create_custom_edges()
        for e in self.custom_edges.values():
            e.calc_uses()

        combined_image = np.zeros(self.image.shape)
        edge_lengths = []
        for (u,v,k,data) in self.edges(keys=True, data=True):
            pts = data["pts"]
            combined_image[pts[...,0],pts[...,1]] += 1
            edge_lengths.append(((u,v,k), len(pts)))
        
        sorted_lengths = sorted(edge_lengths, key=lambda x: x[1])
        skip_edges = set()
        for ((u,v,k), length) in sorted_lengths:
            data = self.get_edge_data(u,v,k)
            pts = data["pts"]
            overused = np.sum(combined_image[pts[...,0], pts[...,1]] > 1)
            if overused > length * 0.7:
                skip_edges.add((u,v,k))
                skip_edges.add((v,u,k))
            combined_image[pts[...,0], pts[...,1]] -= 1
    



        
        for (u,v,k,data) in self.edges(keys=True, data=True):
            if (u,v,k) in skip_edges:
                continue
            if (u,v,k) in self.custom_edges:
                if self.custom_edges[(u,v,k)].uses > 1:
                    continue
                if self.custom_edges[(v,u,k)].uses > 1:
                    continue
            if only_closed:
                if u != v and not (accept_border_edges and self.border_edge(data["pts"], u,v)):
                    continue
            l = np.zeros((self.image.shape[0]+2, self.image.shape[1]+2), dtype=np.uint8)
            pts = data["pts"]
            is_closed = u == v
            d = cdist([pts[-1]], [pts[0]])[0][0]
            if u == v or cdist([pts[-1]], [pts[0]])[0][0] < 10:
                is_closed = True
                extra_pts = np.array(line_nd(pts[-1], pts[0])).T
                if len(extra_pts) > 0:

                    pts = np.concatenate((pts, extra_pts))
            
        


            if u!= v and len(pts) < non_closed_min_size:
                continue
            # for pt in pts:
            #     if tuple(pt) in self.closest_coord_map:
            #         for cp in self.closest_coord_map[tuple(pt)]:
            #             l[cp[0], cp[1]] = 1
            
            l[pts[...,0]+1, pts[...,1]+1] = 1

            skeletons.append(l[1:-1,1:-1])

            if dilation:
                lab = skilabel(l, background=-1, connectivity=2)
                num_features = len(np.unique(lab))
                for i in range(int(self.line_width/2)):
                    current_l = binary_dilation(l)
                    current_label  = skilabel(current_l, background=-1, connectivity=2)
                    current_num_features = len(np.unique(current_label))
                    if current_num_features == num_features:
                        l = current_l
                    else:
                        break
            else:
                
                lab,num_features = skilabel(l==0, connectivity=1,return_num=True)
                # is_border = self.border_edge(data["pts"], u,v)
                # is_border_closer = self.border_edge(data["pts"], u,v,0)
                allowed_features = 1
                if is_closed:
                    allowed_features = 2
                

                if num_features > allowed_features:
                    un, nu = np.unique(lab, return_counts=True)
                    argsorted_uniques = np.argsort(nu)
                    for b in range(num_features-allowed_features):
                        l[lab==un[argsorted_uniques[b]]]= 1
                    lab,num_features = skilabel(l==0, connectivity=1,return_num=True)

                # num_features = len(np.unique(lab))
                decrease = 0
                current_y, current_x = np.nonzero(l)
                
                while True:
                    current_l = np.zeros(l.shape, dtype=np.uint8)
                    radius = self.line_width/2 - decrease
                    if radius <= 1:
                        break
                    disk_y, disk_x = disk((0, 0),radius=radius, )
                    
                    for y,x in zip(current_y, current_x):
                        current_disk_y = disk_y + y
                        current_disk_x = disk_x + x

                        idxs_to_use = np.where((current_disk_y >= 0) & (current_disk_y < l.shape[0]) & (current_disk_x >= 0) & (current_disk_x < l.shape[1]))
                        current_disk_y = current_disk_y[idxs_to_use]
                        current_disk_x = current_disk_x[idxs_to_use]
                        current_l[current_disk_y, current_disk_x] = 1
                    current_l[[0,-1],:] = 0
                    current_l[:,[0,-1]] = 0
                    current_label,current_num_features  = skilabel(current_l == 0, connectivity=1,return_num=True)
                    # current_num_features = len(np.unique(current_label)) 

                    if current_num_features == num_features:
                        l = (current_l == 1) * 1 
                        break
                    
                    else:
                        decrease += 1
                    

            if mask is not None:
                if np.sum(l * mask) > 0:
                    continue
        
            instances.append(l[1:-1, 1:-1])
            edges.append(self.custom_edges[(u,v,k)]) #(u,v,k,data)
            closed.append(is_closed)
            all_pts.append(pts)

        if only_longest and len(instances) > 0:

            max_idx = np.argmax(np.sum(instances, (1,2)))
            instances = [instances[max_idx]]
            edges = [edges[max_idx]]
        return instances, edges, skeletons, closed, all_pts
        
        
    
    def create_custom_edges(self):
        self.custom_edges = edge_dict()
        for e in self.edges(keys=True):

            self.custom_edges[e] = edge(self, self.image, self.closest_coord_map, e, 50, 1, 10)
            self.add_new_edge_key(*e)


    def solve_junctions(self):

        def get_uses(e): 
            if e in self.uses:
                return self.uses[e]

            elif (e[1], e[0], e[2]) in self.uses:
                
                return self.uses[(e[1], e[0], e[2])]
            else:
                raise KeyError(e)
        def reduce_use(e):
            if e in self.uses:
                # if self.uses[edge] > 1:
                self.uses[e] -= 1
                return
            elif (e[1], e[0], e[2]) in self.uses:
                
                # if self.uses[(edge[1], edge[0], edge[2])] > 1:
                self.uses[(e[1], e[0], e[2])] -= 1
                return
            else:
                raise KeyError(e)
        def remove_edge(e, ignore=False):
            # self.remove_edge(*edge)
            # return
            if get_uses(e) > 0 and not ignore:
                return
            if e in self.custom_edges:
                
                self.remove_edge(*e)
            else:
                self.remove_edge(e[1], e[0], e[2])
            return
            if edge in self.uses:
                if self.uses[edge] > 1:
                    self.uses[edge] -= 1
                    return
                self.remove_edge(*edge)
            elif (edge[1], edge[0], edge[2]) in self.uses:
                
                if self.uses[(edge[1], edge[0], edge[2])] > 1:
                    self.uses[(edge[1], edge[0], edge[2])] -= 1
                    return
                self.remove_edge(*edge)
            else:
                raise KeyError(edge)


        def nodes_allowed(nodes_used, allowed_combination, fc,sc, node_idx):
            if fc[0] not in nodes_used and fc[1] not in nodes_used and sc[0] not in nodes_used and sc[1] not in nodes_used:
                return True
            if sc[0] not in nodes_used and sc[1] not in nodes_used and (fc, node_idx) in allowed_combination:
                return True
            if fc[0] not in nodes_used and fc[1] not in nodes_used and (sc, node_idx) in allowed_combination:
                return True
            if (fc, node_idx) in allowed_combination and (sc, node_idx) in allowed_combination:
                return True
            return False
        counter = 0

        while any([degree[1] > 1 for degree in self.degree()]):
            self.create_custom_edges()

            junctions = []
            
            for (node, degree) in self.degree():
                
                if degree >= 2:
                    if degree == 2:
                        if len([e for e in self.edges(node, keys=True)]) == 1:
                            continue
                        

                    j = junction(self, node, distance=10)
                    junctions.append(j)
            
            for e in self.custom_edges.values():
                e.calc_uses()


            connection_values = []
            junction_dict = {}

            for j in junctions:
                j:junction
                junction_dict[j.node_idx] = j
                connections = j.solve_edge_connections(verbose=False, counter=counter)
                # junction_connections[j.node_idx] = connections
                connection_values.extend(connections)

            sorted_values = sorted(connection_values)

            edges_to_remove = set()
            nodes_used = set()
            to_break = True
            degree = None

            double_connections = []
            for c_counter, connection in enumerate(sorted_values):
                fc, sc = connection.connection

                if fc == sc or connection.type == "same":
                    continue
                all_nodes = set([fc[0], fc[1], sc[0], sc[1]])
                
                if len(all_nodes) == 2:
                    all_nodes.remove(connection.node_idx)
                    other_node = list(all_nodes)
                    assert len(other_node) == 1
                    other_node = other_node[0]

                    for oc in sorted_values[c_counter + 1:]:
                        # if fc[:2] == oc.connection[0][:2] or (fc[1], fc[0]) == oc.connection[0][:2]:
                        #     if sc[:2] == oc.connection[1][:2] or (sc[1], sc[0]) == oc.connection[1][:2]:
                        #         double_connections.append((connection, oc))
                        # elif fc[:2] == oc.connection[1][:2] or (fc[1], fc[0]) == oc.connection[1][:2]:
                        #     if sc[:2] == oc.connection[0][:2] or (sc[1], sc[0]) == oc.connection[0][:2]:
                        #         double_connections.append((connection, oc))
                        if fc == oc.connection[0] or (fc[1], fc[0], fc[2]) == oc.connection[0]:
                            if sc == oc.connection[1] or (sc[1], sc[0], sc[2]) == oc.connection[1]:
                                double_connections.append((connection, oc))
                        elif fc == oc.connection[1] or (fc[1], fc[0], fc[2]) == oc.connection[1]:
                            if sc == oc.connection[0] or (sc[1], sc[0], sc[2]) == oc.connection[0]:
                                double_connections.append((connection, oc))
            
            ignore_uses = set()

            for double_connection in double_connections:
                fc, sc = double_connection
                if fc.connection[0] in edges_to_remove or fc.connection[1] in edges_to_remove or (fc.connection[0][1],fc.connection[0][0],fc.connection[0][2]) in edges_to_remove or (fc.connection[1][1],fc.connection[1][0],fc.connection[1][2]) in edges_to_remove:
                    continue
                # if self.is_allowed(*fc, *sc):
                u,v,k = self.combine_edges(fc.connection, fc.edge_direction, fc.node_idx, junction_dict[fc.node_idx].junction_pts)

                self.custom_edges[(u,v,k)] = edge(self, self.image, self.closest_coord_map, (u,v,k), 50, 1, 10)
                self.uses[(u,v,k)] = 0
                # self.combine_edges(((u,v,k), (u,v,k)), None, u, junction_dict[u].junction_pts)
                nodes_used.add(fc.connection[0][0])
                nodes_used.add(fc.connection[0][1])
                if len(self.custom_edges[fc.connection[0]].pts) > len(self.custom_edges[fc.connection[1]].pts):
                    ignore_uses.add(fc.connection[0])
                    
                else:
                    ignore_uses.add(fc.connection[1])
                edges_to_remove.add(fc.connection[0])
                edges_to_remove.add(fc.connection[1])
                edges_to_remove.add((u,v,k))
                reduce_use(fc.connection[0])
                reduce_use(fc.connection[1])
                to_break = False
                


            still_allow_edge_node_combination = set()
            for connection in sorted_values:
                connection:connection_value
                fc = connection.connection[0]
                sc = connection.connection[1]
                
                if connection.type  == "same" :
                    u,v,k = self.combine_edges(connection.connection, connection.edge_direction, connection.node_idx, junction_dict[connection.node_idx].junction_pts)
                    
                    if (fc[1],fc[0], fc[2]) not in edges_to_remove:
                        edges_to_remove.add(fc)
                    if (sc[1],sc[0], sc[2]) not in edges_to_remove:
                        edges_to_remove.add(sc)
                    reduce_use(fc)
                    to_break = False
                elif get_uses(fc) > 0 and get_uses(sc) > 0 and sc not in edges_to_remove and (fc[1],fc[0], fc[2]) not in edges_to_remove and sc not in edges_to_remove and (sc[1],sc[0], sc[2]) not in edges_to_remove:
                    if nodes_allowed(nodes_used, still_allow_edge_node_combination, fc, sc, connection.node_idx):
                    
                        if self.is_allowed(*fc, *sc):
                            if degree is None or connection.degree <= degree:

                                u,v,k = self.combine_edges(connection.connection, connection.edge_direction, connection.node_idx, junction_dict[connection.node_idx].junction_pts)

                                if (fc[1],fc[0], fc[2]) not in edges_to_remove:
                                    edges_to_remove.add(fc)
                                if (sc[1],sc[0], sc[2]) not in edges_to_remove:
                                    edges_to_remove.add(sc)
                                
                                reduce_use(fc)
                                reduce_use(sc)
                                nodes_used.add(fc[0])
                                nodes_used.add(fc[1])
                                nodes_used.add(sc[0])
                                nodes_used.add(sc[1])
                                if get_uses(fc) > 0:
                                    still_allow_edge_node_combination.add((fc, connection.node_idx))
                                    still_allow_edge_node_combination.add(((fc[1], fc[0], fc[2]), connection.node_idx))
                                elif (fc, connection.node_idx) in still_allow_edge_node_combination:
                                    still_allow_edge_node_combination.remove((fc, connection.node_idx))
                                    still_allow_edge_node_combination.remove(((fc[1], fc[0], fc[2]), connection.node_idx))
                                
                                if get_uses(sc) > 0:
                                    still_allow_edge_node_combination.add((sc, connection.node_idx))
                                    still_allow_edge_node_combination.add(((sc[1], sc[0], sc[2]), connection.node_idx))
                                elif (sc, connection.node_idx) in still_allow_edge_node_combination:
                                    still_allow_edge_node_combination.remove((sc, connection.node_idx))
                                    still_allow_edge_node_combination.remove(((sc[1], sc[0], sc[2]), connection.node_idx))
                                to_break = False
                                degree = connection.degree
                # else:
                #     u,v,k = self.combine_edges(connection.connection, connection.edge_direction, connection.node_idx, junction_dict[connection.node_idx].junction_pts)
                #     edges_to_remove.add(connection.connection[0])
                #     edges_to_remove.add(connection.connection[1])
                    
            
            
            for e in edges_to_remove:
                remove_edge(e, e in ignore_uses)
            # self.remove_edges_from(edges_to_remove)
            nodes_to_remove = []
            for n, d in self.degree():
                if d == 0:
                    nodes_to_remove.append(n)
            # for n in nodes_to_remove:
            #     self.remove_node(n)
            if to_break:
                break

            
            #     if sth_solved:

            #         break


            # else:
            #     break

            counter += 1
            self.create_custom_edges()



    def solve_junctions_orig(self):

        counter = 0

        while any([degree[1] > 2 for degree in self.degree()]):
            
            self.create_custom_edges()

            junctions = []
            
            for (node, degree) in self.degree():
                
                if degree > 2:

                    j = junction(self, node, distance=10)
                    junctions.append(j)
            
            for e in self.custom_edges.values():
                e.calc_uses()
            for j in junctions:
                j:junction
                sth_solved = j.solve_edge_connections(verbose=False, counter=counter)
                if sth_solved:

                    break


            else:
                break

            counter += 1
            


    def remove_small_outer_edges(self, verbose=False):
        while True:
            self.create_custom_edges()
            edges = self.edges(data=True, keys=True)
            edges = [(u,v,k,data) for u,v,k,data in edges if self.degree(u) == 1 or self.degree(v) == 1 or (u==v and data["weight"] < 10)]

            if len(edges) == 0:
                return
            lengths = [data["weight"] for u,v,k,data in edges]
            argsorted = np.argsort(lengths)
            if lengths[argsorted[0]] < self.min_length:
                u,v,k, data = edges[argsorted[0]]
                if self.degree(v) == 1:
                    node_to_keep = u
                    node_to_delete = v
                elif self.degree(u) == 1:
                    node_to_keep = v
                    node_to_delete = u
                else:
                    if self.degree(u) == 2:
                        node_to_delete = u
                        node_to_keep = None
                    else:
                        node_to_delete = None
                        node_to_keep = u
                self.remove_edge(u,v,k)
                if node_to_delete is not None:
                    self.remove_node(node_to_delete)
                if node_to_keep is None:
                    continue
                if self.degree(node_to_keep) == 2:
                    edges = [e for e in self.edges(node_to_keep, keys=True)]
                    if len(edges) == 1:
                        if  edges[0][0] == edges[0][1]:
                            edges = [edges[0], edges[0]]
                        else:
                            assert len(edges) == 2, f"{len(edges)}, {edges}"


                    assert len(edges) <= 2, f"{len(edges)}, {edges}"
                    
                    directions = [self.custom_edges[e].get_closest_idx(self.nodes(data=True)[node_to_keep]["pts"]) for e in edges]
                    self.combine_edges(edges, directions, node_to_keep)
                    self.remove_edges_from(edges)
                    self.remove_node(node_to_keep)               

            else:
                break
        while True:
            self.create_custom_edges()
            for node, degree in self.degree():
                if degree == 2:
                    edges = [e for e in self.edges(node, keys=True)]

                    if len(edges) == 1:
                        continue
                        if  edges[0][0] == edges[0][1]:
                            edges = [edges[0], edges[0]]
                        else:
                            assert len(edges) == 2, f"{len(edges)}, {edges}"


                    assert len(edges) <= 2, f"{len(edges)}, {edges}"
                    
                    directions = [self.custom_edges[e].get_closest_idx(self.nodes(data=True)[node]["pts"]) for e in edges]
                    self.combine_edges(edges, directions, node)
                    self.remove_edges_from(edges)
                    self.remove_node(node)         
                    break
            else:
                break


    def calc_lengths(self, pts):
        return np.sum(np.hypot(*np.diff(pts, axis=0).T))

    def combine_nodes(self, nodes_idx):
        
       
    
        connected_nodes = []
        for idx_1, idx_2 in it.combinations(nodes_idx, 2):
            if idx_2 not in self[idx_1]:
                continue
            current_edges = self[idx_1][idx_2]
            edges_idx = list(sorted(current_edges.keys()))
            edge_to_use = edges_idx[np.argmin([current_edges[idx]["weight"] for idx in edges_idx])]
            if current_edges[edge_to_use]["weight"] < self.fuse_distance:
                for connected_set in connected_nodes:
                    if idx_1 in connected_set or idx_2 in connected_set:
                        connected_set.update([idx_1, idx_2])
                        break
                else:
                    connected_nodes.append(set([idx_1, idx_2]))
        if len(connected_nodes) == 0:
            return []
        while True:
            for set_1, set_2 in it.combinations(connected_nodes,2):
                set_1:set
                if len(set_1.intersection(set_2)) > 0:
                    set_1.update(set_2)
                    connected_nodes.remove(set_2)
                    break
            else:
                break
        all_seen_nodes = set()
        for counter, nodes_idx in enumerate(connected_nodes):
            real_node_idxs = set()
            edges = {}
       
            connections = {}
            for idx_1, idx_2 in it.combinations(nodes_idx, 2):
                try:
                    if idx_2 not in self[idx_1]:
                        continue
                except Exception as e:
                    raise e
                current_edges = self[idx_1][idx_2]
                edges_idx = list(sorted(current_edges.keys()))
                edge_to_use = edges_idx[np.argmin([current_edges[idx]["weight"] for idx in edges_idx])]
                if current_edges[edge_to_use]["weight"] < self.fuse_distance:
                    
                    edges[(idx_1, idx_2, edge_to_use)] = self[idx_1][idx_2][edge_to_use]
                    real_node_idxs.update([idx_1, idx_2])
                    connections[(idx_1, idx_2)]  = current_edges[edge_to_use]
            
            

            before_idxs = nodes_idx
            if len(real_node_idxs) == 0:
                continue
            nodes_idx = real_node_idxs
            try:
                nodes = [self.nodes[idx] for idx in nodes_idx]
            except Exception as e:


                raise e
            original_node_pts = {idx:node for idx, node in zip(nodes_idx, nodes)}
            new_node_attr = {}
            new_node_pts = [node["pts"] for node in nodes]
            new_node_pts.extend([edge["pts"] for edge in edges.values()])
            new_node_attr["pts"] = np.concatenate(new_node_pts)
            new_node_attr["o"] = np.mean(new_node_attr["pts"], 0)
            new_node_attr["connections"] = connections
            new_node_attr["orig_pts"] = original_node_pts
            
            idx = np.max([i for i in self.nodes.keys()]) + 1
            self.add_node(idx, **new_node_attr)

            for node_idx in nodes_idx:
                current_edges = [i for i in self.edges(node_idx,keys=True)]
                for edge in current_edges:
                    if edge in edges:
                        self.remove_edge(*edge)
                    else:
                        u,v,k = edge

                        if u in nodes_idx and v in nodes_idx:
                            self.add_edge(idx, idx,**self.get_edge_data(u,v,k))

                            self.remove_edge(u,v,k)
                        elif u in nodes_idx:

                            self.add_edge(v, idx,**self.get_edge_data(u,v,k))
                            self.remove_edge(u,v,k)
                        elif v in nodes_idx:
                            self.add_edge(u, idx,**self.get_edge_data(u,v,k))
                            self.remove_edge(u,v,k)
                self.remove_node(node_idx)
            all_seen_nodes.update(nodes_idx)
        return all_seen_nodes


    def combine_edges(self, edge_idxs, direction_idxs, node_idx, junction_pts=np.empty((0,2), dtype=np.int32, )):
        def find_best_connection(coord_1, coord_2, connections:dict, verbose=False):
            def dist(a,b):
                return np.sqrt(np.sum((a-b)**2))
            keys = set()
            for key in connections.keys():
                keys.update(key[:2])
            keys = list(keys)
            connection_matrix = np.zeros((len(keys),len(keys)))
            for key, value in connections.items():
                connection_matrix[keys.index(key[0]),keys.index(key[1])] = value["weight"]
                connection_matrix[keys.index(key[1]),keys.index(key[0])] = value["weight"]
            
            best_idx = []

            for coord in [coord_1, coord_2]:
                
                best_idx.append(keys[np.argmin([np.min(cdist([coord], self.nodes(data=True)[node_idx]["orig_pts"][key]["pts"])) for key in keys])])
            best_idx = [keys.index(idx) for idx in best_idx]
            _, pred, _ = dijkstra(connection_matrix, False, indices=[best_idx[-1]], return_predecessors=True, min_only=True)
           

            
            
            idxs = [best_idx[0]]

            if best_idx[0] == best_idx[1]:
                pts = np.array(line_nd(coord_1, [int(o) for o in self.nodes(data=True)[node_idx]["orig_pts"][keys[best_idx[0]]]["o"]], )).T
                line_pts = np.array(line_nd([int(o) for o in self.nodes(data=True)[node_idx]["orig_pts"][keys[best_idx[0]]]["o"]],coord_2 )).T
                pts = np.concatenate((pts, line_pts))
            else:
                assert pred[best_idx[0]] >= 0, f"{pred}, {best_idx[0]}, {connections}, {node_idx}"
                while True:
                    next_idx = pred[idxs[-1]]
                    idxs.append(next_idx)
                    if next_idx == best_idx[-1]:
                        break
                pts = np.array(line_nd(coord_1, [int(o) for o in self.nodes(data=True)[node_idx]["orig_pts"][keys[idxs[0]]]["o"]], )).T
                if len(pts) == 0:
                    pts = np.array([coord_1])
                if len(idxs) > 1:
                    for idx_1, idx_2 in zip(idxs[:-1], idxs[1:]):
                        if (keys[idx_1], keys[idx_2]) in connections:
                            current_pts = connections[(keys[idx_1], keys[idx_2])]["pts"]
                        else:
                            current_pts = connections[(keys[idx_2], keys[idx_1])]["pts"]
                        asdf = self.nodes(data=True)[node_idx]["orig_pts"][keys[idxs[0]]]["pts"]
                        assert len(pts) > 0, f"{coord_1}, {asdf}, "
                        assert len(current_pts) > 0
                        if dist(pts[-1], current_pts[0]) > dist(pts[-1], current_pts[-1]):
                            current_pts = current_pts[::-1]
                        line_pts = np.array(line_nd(pts[-1], current_pts[0])).T
                        
                        pts = np.concatenate((pts, line_pts, current_pts))
                line_pts = np.array(line_nd(pts[-1], coord_2)).T
                pts = np.concatenate((pts, line_pts))
            return pts       


        if edge_idxs[0] == edge_idxs[1]:
            edge_data = self.custom_edges[edge_idxs[0]].pts
            node_attr = self.nodes(data=True)[node_idx]
            new_node_idx = np.max([i for i in self.nodes.keys()]) + 1
            if "connections" in node_attr:                
                extra_line = find_best_connection(edge_data[0], edge_data[-1], node_attr["connections"])
            else:
                extra_line = np.array(line_nd(edge_data[-1], edge_data[0], endpoint=False)).T
            if self.line_width > 0:
                extra_line_pts = line(np.zeros_like(self.image, dtype=np.int8), edge_data[-1][::-1], edge_data[0][::-1],1,self.line_width)
                extra_line_pts = np.array(np.nonzero(extra_line_pts)).T
            else:
                extra_line_pts = []
            # 
            if len(extra_line) > 0:
                extra_line = extra_line[1:]
            edge_data = np.concatenate((edge_data, extra_line))

            
            
            attr = {"pts":edge_data[:1], "o":edge_data[0], "junction_pts":junction_pts}
            edge_data = edge_data[1:]
            
            self.add_node(new_node_idx, **attr)
            key = self.add_edge(new_node_idx,new_node_idx, pts=edge_data, weight=self.calc_lengths(edge_data), extra_pts=extra_line_pts, junction_pts=junction_pts)



            # self.remove_edge(*edge_idxs[0])
            return new_node_idx, new_node_idx, key

        edge_pts_1 = self.custom_edges[edge_idxs[0]].pts
        edge_pts_2 = self.custom_edges[edge_idxs[1]].pts
        
        if direction_idxs[0] == 0:
            edge_pts_1 = edge_pts_1[::-1]
        if direction_idxs[1] == 1:
            edge_pts_2 = edge_pts_2[::-1]
        node_attr = self.nodes(data=True)[node_idx]
        if "connections" in node_attr:

            extra_line = find_best_connection(edge_pts_1[-1], edge_pts_2[0], node_attr["connections"])

        else:


            extra_line = np.array(line_nd(edge_pts_1[-1], edge_pts_2[0], endpoint=False)).T
        if self.line_width > 0:

            extra_line_pts = line(np.zeros_like(self.image, dtype=np.int8), edge_pts_1[-1][::-1], edge_pts_2[0][::-1],1,self.line_width)
            extra_line_pts = np.array(np.nonzero(extra_line_pts)).T
        else:
            extra_line_pts = []
        if len(extra_line) > 0:
            extra_line = extra_line[1:]
        

 
        new_edge_data = np.concatenate((edge_pts_1, extra_line, edge_pts_2))
        u,v = self.other_node(edge_idxs[0], node_idx), self.other_node(edge_idxs[1], node_idx)
        

        edge_data = self.edges(keys=True, data=True)
        
        edge_data_1 = self.get_edge_data(*self.custom_edges[edge_idxs[0]].edge_idx)
        edge_data_2 = self.get_edge_data(*self.custom_edges[edge_idxs[1]].edge_idx)

        if "junction_pts" in edge_data_1:
            junction_pts = np.concatenate((junction_pts, edge_data_1["junction_pts"]))
        if "junction_pts" in edge_data_2:
            junction_pts = np.concatenate((junction_pts, edge_data_2["junction_pts"]))
        
        new_nodes = set([node_idx])
        self.add_not_allowed(*edge_idxs[0], *edge_idxs[1])
        self.add_not_allowed(*edge_idxs[1], *edge_idxs[0])
        if  "nodes" in edge_data_2 and u in edge_data_2["nodes"]:
            
            distances = cdist([edge_pts_1[0]], edge_pts_2)
            best_idx = np.argmin(distances, 1)[0]
            new_edge_data = np.concatenate((edge_pts_1, extra_line, edge_pts_2[:best_idx]))
            other_pts = edge_pts_2[best_idx:]
            
            new_node_idx = np.max([i for i in self.nodes.keys()]) + 1
            node_attr = {"pts":new_edge_data[:1], "o":new_edge_data[0], "junction_pts":junction_pts}
            edge_data = new_edge_data[1:]
            
            self.add_node(new_node_idx, **node_attr)
            key = self.add_edge(new_node_idx,new_node_idx, pts=edge_data, weight=self.calc_lengths(edge_data), junction_pts=junction_pts)
            
            new_edge_data = other_pts
            extra_line_pts = []
            return new_node_idx, new_node_idx, key
        

            
        elif "nodes" in edge_data_1 and v in edge_data_1["nodes"]:
            
           
            distances = cdist([edge_pts_2[-1]], edge_pts_1)
            best_idx = np.argmin(distances, 1)[0]
            new_edge_data = np.concatenate((edge_pts_1[best_idx:], extra_line, edge_pts_2))[::-1]
            other_pts = edge_pts_1[:best_idx]
           

            new_node_idx = np.max([i for i in self.nodes.keys()]) + 1
            node_attr = {"pts":new_edge_data[:1], "o":new_edge_data[0], "junction_pts":junction_pts}
            edge_data = new_edge_data[1:]
            
            self.add_node(new_node_idx, **node_attr)
            key = self.add_edge(new_node_idx,new_node_idx, pts=edge_data, weight=self.calc_lengths(edge_data), junction_pts=junction_pts)
            return new_node_idx, new_node_idx, key

        
        nodes_1 = set()
        nodes_2 = set()
        if "nodes" in edge_data_1:
            nodes_1 = edge_data_1["nodes"]
        if "nodes" in edge_data_2:
            nodes_2 = edge_data_2["nodes"]
       
        new_nodes = nodes_1.union(nodes_2).union(new_nodes)

            # new_nodes = edge_data_1["nodes"].union(edge_data_2["nodes"])
        # else:
        #     new_nodes = set([u,v, node_idx])
        attr = {"pts":new_edge_data, "weight":self.calc_lengths(new_edge_data)}

        key = self.get_new_edge_key(u,v)
        # key = self.new_edge_key(u,v)
        self.add_edge(u,v,key=key, pts=attr["pts"], weight=attr["weight"],nodes=new_nodes, junction_pts=junction_pts, extra_pts=extra_line_pts)
        self.add_new_edge_key(u,v,key)
        
        self.add_not_allowed(*edge_idxs[1], u,v,key)
        self.add_not_allowed(*edge_idxs[0], u,v,key)
        self.add_not_allowed(u,v,key, *edge_idxs[0])
        self.add_not_allowed(u,v,key, *edge_idxs[1])


        return u,v,key

        # self.remove_edges_from(edge_idxs)


    def add_not_allowed(self, u,v,key, x,y,key_2):
        if (u,v,key) in self.not_allowed:
            t = (u,v,key)
            self.not_allowed[t].add((x,y,key_2))
        elif (v,u,key) in self.not_allowed:
            t = (v,u,key)
            self.not_allowed[t].add((x,y,key_2))
        else:
            self.not_allowed[(u,v,key)] = set([(x,y,key_2), (u,v,key)])
            t = (u,v,key)
        
        
        if (x,y,key_2) in self.not_allowed:
            r = (x,y,key_2)
            self.not_allowed[r].add((u,v,key))
        elif (y,x,key_2) in self.not_allowed:
            r = (y,x,key_2)
            self.not_allowed[r].add((u,v,key))
        else:
            self.not_allowed[(y,x,key_2)] = set([(x,y,key_2), (u,v,key)])
            r = (y,x,key_2)
        
        self.not_allowed[t].update(self.not_allowed[r])
        self.not_allowed[r].update(self.not_allowed[t])
        

    def is_allowed(self, u,v,key, x,y,key_2):
        if (u,v,key) in self.not_allowed:
            t = (u,v,key)
            if (x,y,key_2) in self.not_allowed[t] or (y,x,key_2) in self.not_allowed[t]:
                return False
        elif (v,u,key) in self.not_allowed:
            t = (v,u,key)
            if (x,y,key_2) in self.not_allowed[t] or (y,x,key_2) in self.not_allowed[t]:
                return False
        else:
            first_bool = True
        
        
        if (x,y,key_2) in self.not_allowed:
            t = (x,y,key_2)
            if (u,v,key) in self.not_allowed[t] or (v,u,key) in self.not_allowed[t]:
                return False
        elif (y,x,key_2) in self.not_allowed:
            t = (y,x,key_2)
            if (u,v,key) in self.not_allowed[t] or (v,u,key) in self.not_allowed[t]:
                return False
        else:
            second_bool = True
       
        return True

    def add_new_edge_key(self, u,v, key):
        if (u,v) in self.used_edge_keys:
            self.used_edge_keys[(u,v)].add(key)
        elif (v,u) in self.used_edge_keys:
            self.used_edge_keys[(v,u)].add(key)
        else:
            self.used_edge_keys[(u,v)] = set([key])

    def get_new_edge_key(self, u,v):
        if (u,v) in self.used_edge_keys:
            t = (u,v)
        elif (v,u) in self.used_edge_keys:
            t = (v,u)
        else:
            return 0
        new_key = np.max(list(self.used_edge_keys[t])) + 1 
        return new_key

    def other_node(self, e, node):
        if e[0] == node:
            return e[1]
        if e[1] == node:
            return e[0]
        raise ValueError(f"Node:{e} is not part of edge: {e}.")


    def fuse_nodes(self):
        
        nodes = self.nodes(data=True)
        if len(nodes) == 1:
            return
        pts = {node[0]:node[1]["pts"] for node in nodes if self.degree(node[0])>1}
        if len(pts) <= 1:
            return
        fusing_idxs = {}
        for idx_1, idx_2 in it.combinations(pts.keys(),2):
            pts_1 = pts[idx_1]
            pts_2 = pts[idx_2]
            distance = np.min(cdist(pts_1, pts_2))
            if distance < self.fuse_distance:
                if idx_1 not in fusing_idxs:
                    fusing_idxs[idx_1] = set([idx_1, idx_2])
                if idx_2 not in fusing_idxs:
                    fusing_idxs[idx_2] = set([idx_1, idx_2])
                fusing_idxs[idx_1].update(fusing_idxs[idx_2])
                fusing_idxs[idx_2].update(fusing_idxs[idx_1])
                for t in fusing_idxs[idx_1]:
                    if t == idx_1:
                        continue
                    fusing_idxs[t].update(fusing_idxs[idx_1])
        
        seen_nodes = set()


        for key, value in fusing_idxs.items():
            if key not in seen_nodes:

                used_nodes = self.combine_nodes(value)

                if len(used_nodes) > 0:
                    seen_nodes.update(value)
            

    def get_all_pts(self):
        
        pts = [value[3]["pts"] for value in self.edges(data=True, keys=True)]
        pts.extend([value[1]["pts"] for value in self.nodes(True)])
        return np.concatenate(pts)


class edge:
    def __init__(self, network:curvature_network, image,closest_coord_map:dict, edge_idx:tuple, neighbours:int, distance:int, min_length=5, ):
        self.network = network
        self.edge_idx = edge_idx
        data = network.get_edge_data(*edge_idx)
        if "nodes" not in data:
            data["nodes"] = set()
        self.original_pts = network.get_edge_data(*edge_idx)["pts"]

        
        data = network.get_edge_data(*edge_idx)
        if "not_allowed" not in data:
            data["not_allowed"] = set()
        self.curvatures = {0:None, 1:None}
        
        self.distance = distance
        self.min_length = min_length
        self.closest_coord_map = closest_coord_map
        self.junctions = {}
        self.uses = 1
        self.create_pts()
        self.neighbours = max(neighbours, len(self.pts) * 0.2)
        self.neighbours = int(min(self.neighbours, len(self.pts)))
        self.calc_curvature()
        self.calc_tangent()
        # self.calc_lineness()




    def min_distance(self, other):
        self_pts = np.array([self.pts[0],self.pts[-1]])
        other_pts = np.array([other.pts[0], other.pts[-1]])
        dist = cdist(self_pts, other_pts)
        idx = np.unravel_index(np.argmin(dist), (2,2))
        return dist.flatten(), [(0,0),(0,1),(1,0),(1,1)]

    def calc_uses(self):
        # if self.edge_idx in self.network.uses:
        #     self.uses = self.network.uses[self.edge_idx]
        #     return
        # elif (self.edge_idx[1], self.edge_idx[0], self.edge_idx[2]) in self.network.uses:
        #     self.uses = self.network.uses[(self.edge_idx[1], self.edge_idx[0], self.edge_idx[2])]
        #     return

        if len(self.junctions.keys()) > 1:


            if len(self.junctions.keys()) == 2 and self.network.get_edge_data(*self.edge_idx)["weight"] < 150:
                self.uses = np.sum([value for value in self.junctions.values()]) -1 
        self.network.uses[self.edge_idx] = self.uses

    def not_allowed(self, other):
        return other.edge_idx in self.network.get_edge_data(*self.edge_idx)["not_allowed"]
    
    def add_to_not_allowed(self, other):
        self.network.get_edge_data(*self.edge_idx)["not_allowed"].add(other.edge_idx)

    def create_pts(self):
        cropsize = min((len(self.original_pts) - self.min_length) // 2, self.distance)
        if cropsize <= 0:
            self.pts = np.copy(self.original_pts)
        else:
            self.pts = self.original_pts[cropsize:-cropsize]
    

    def calc_tangent(self, verbose=False):
        # nr_of_neighbours = max(min(len(self.pts), self.neighbours // 4),1)
        nr_of_neighbours = max(self.neighbours // 4,2)
        nr_of_neighbours = min(10, len(self.pts))
        # nr_of_neighbours = 5

        neighbourhood_pts = (self.original_pts[:nr_of_neighbours], self.original_pts[-nr_of_neighbours:][::-1])

        # vector_1 = neighbourhood_pts[0][0] - neighbourhood_pts[0][-1]
        # vector_1 = vector_1 / np.linalg.norm(vector_1)

        # vector_2 = neighbourhood_pts[1][0] - neighbourhood_pts[1][-1]
        # vector_2 = vector_2 / np.linalg.norm(vector_2)
        
        vector_1 = self.fit_line(neighbourhood_pts[0],start=None,get_vector=True, verbose=verbose)
        vector_2 = self.fit_line(neighbourhood_pts[1],start=None,get_vector=True, verbose=verbose)

        vector_1_approx = neighbourhood_pts[0][-1 ] -self.original_pts[0] 
        vector_1_approx = vector_1_approx / np.linalg.norm(vector_1_approx)
        vector_2_approx = neighbourhood_pts[1][-1 ] - self.original_pts[-1] 
        vector_2_approx = vector_2_approx / np.linalg.norm(vector_2_approx)

        vector_1_angle = np.arccos(np.clip(np.dot(vector_1, vector_1_approx),-1,1)) 
        vector_2_angle = np.arccos(np.clip(np.dot(vector_2, vector_2_approx),-1,1))
        if vector_1_angle > np.pi*0.5 and vector_1_angle < np.pi * 1.5:
            vector_1 = vector_1 * -1 
        if vector_2_angle > np.pi*0.5 and vector_2_angle < np.pi * 1.5:
            vector_2 = vector_2 * -1 


        self.tangents = (vector_1, vector_2)

    def angle(self, other, directions, verbose=False):
        self.calc_tangent(verbose=verbose)
        other.calc_tangent(verbose=verbose)
        angle = np.arccos(np.clip(np.dot(self.tangents[directions[0]],other.tangents[directions[1]]),-1,1))

        return angle


    def fit_line(self, pts, start, get_vector=False, verbose=False):
        def perp_dist(x,y, a,b,c):
            return np.abs(a*x+b*y+c) / np.sqrt(a**2 + b**2)

        def rotate(origin, points, angle):
            """
            Rotate a point counterclockwise by a given angle around a given origin.

            The angle should be given in radians.
            """
            ox, oy = origin
            
            rotated_points=[]
            for point in points: 
                
                px, py = point

                qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
                qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
                rotated_points.append([qx,qy])
            return np.array(rotated_points)
        rotated_points = rotate(np.mean(pts,0), pts, np.pi * 0.5)
        
        if np.std(pts[...,0]) == 0:
            if get_vector:
                return np.array([0,1])
            else:
                return 0
        if np.std(pts[...,1]) == 0:
            if get_vector:
                return np.array([1,0])
            else:
                return 0

        try:
            (a,b), res, rank, sing_v, rcond = np.polyfit(pts[:,1], pts[:,0],1,full=True)
        except Exception as e:
            raise e
        perp_error = np.sum(perp_dist(pts[:,1],pts[:,0], a,-1, b))
        own_error = np.sum((np.abs((a*pts[:,1] + b) - pts[:,0])))
        




        (a_,b_), res_, rank, sing_v, rcond = np.polyfit(rotated_points[:,1], rotated_points[:,0],1,full=True)
        perp_error_ = np.sum(perp_dist(rotated_points[:,1],rotated_points[:,0], a_,-1, b_))
        own_error_ = np.sum((np.abs((a_*rotated_points[:,1] + b_) - rotated_points[:,0])))

        if get_vector:
            if perp_error_ < perp_error:
                p_1 = np.array([0,b_])
                p_2 = np.array([1,a_+b_])
            else:
                p_1 = np.array([0,b])
                p_2 = np.array([1,a+b])
            vector = p_2-p_1
            vector = vector / np.linalg.norm(vector)
            if perp_error_ < perp_error:
                vector = np.array([vector[1], -vector[0]])
            return vector[::-1]
            
                
        else:
            return min(perp_error / len(pts),perp_error_ / len(pts))

    def calc_lineness(self):


        nr_of_neighbours = max(1, self.neighbours//2)

        neighbourhood_pts = (self.pts[:nr_of_neighbours], self.pts[-nr_of_neighbours:])


    def calc_curvature(self, ):
        nr_of_neighbours = self.neighbours
        # nr_of_neighbours = min(len(self.pts), nr_of_neighbours)


        neighbourhood_pts = (self.pts[:nr_of_neighbours], self.pts[-nr_of_neighbours:])
        # neighbourhood_pts = [np.concatenate([self.closest_coord_map[tuple(pt)] for pt in pts]) for pts in neighbourhood_pts]
        results = [leastsq_circle(pts) for pts in neighbourhood_pts]
        self.curvatures = {i:{"curvature":j[0], "center":j[1]} for i,j in enumerate(results)}
        

    def curvature_of_pts(self, pts, weights=None):
        result = leastsq_circle(pts, weights)
        return {"curvature":result[0], "center":result[1]}


    def get_combined_curvature_center(self, other, direction_idxs):
        pts = []
        neighbours = (self.neighbours, other.neighbours)
        # neighbours = min(max(1, min(len(self.pts), self.neighbours)),max(1, min(len(other.pts), other.neighbours)))
        weights= []
        for e, direction, neighbour in zip([self, other], direction_idxs, neighbours):
            e:edge
            current_pts = e.pts
            assert direction in [0,1]
            if direction == 0:
                current_pts = current_pts[:neighbour]
            else:
                current_pts = current_pts[-neighbour:]
            pts.append(current_pts)
            weights.append(np.ones(len(current_pts)) / len(current_pts))
        pts = np.concatenate(pts)
        weights = np.concatenate(weights)
        result = self.curvature_of_pts(pts, weights)
        return result
    

    def get_curvature_fitting(self, other, direction_idxs):
        
        curvature_center_self = self.curvatures[direction_idxs[0]]["center"]
        radius_self = self.curvatures[direction_idxs[0]]["curvature"]

        if direction_idxs[1] == 0:
            current_pts = other.pts[:other.neighbours]
        else:
            current_pts = other.pts[-other.neighbours:]
        distance = cdist(np.array([curvature_center_self]), current_pts)

        average_distance = np.mean(np.abs(distance - radius_self))



        return average_distance
    

    def get_combined_lineness(self, other, direction_idxs):
        pts = []
        # neighbours = max(1, min(len(self.pts), self.neighbours//2))
        neighbours = (self.neighbours, other.neighbours)
        for e, direction, neighbour in zip([self, other], direction_idxs, neighbours):
            e:edge
            current_pts = e.pts
            assert direction in [0,1]
            if direction == 0:
                current_pts = current_pts[:neighbour]
            else:
                current_pts = current_pts[-neighbour:]
            pts.append(current_pts)
        pts = np.concatenate(pts)
        
        if self != other:
            result = self.fit_line(pts, None)
            return result
        else:
            return np.Inf

    def __eq__(self, __o: object) -> bool:
        return self.edge_idx == __o.edge_idx
    
    def find_combined_curvature_distance(self, other, direction_idxs, normalize=False):
        other:edge
        curvature_center_self = self.curvatures[direction_idxs[0]]["center"]
        curvature_center_other = other.curvatures[direction_idxs[1]]["center"]
        radius_self = self.curvatures[direction_idxs[0]]["curvature"]
        radius_other = other.curvatures[direction_idxs[0]]["curvature"]
        # combined_radius = (radius_self+radius_other) / 2
        combined_curvature_center = self.get_combined_curvature_center(other, direction_idxs)
        combined_radius = combined_curvature_center["curvature"]
        combined_curvature_center = combined_curvature_center["center"]

        centers = [curvature_center_self, curvature_center_other, combined_curvature_center]
        distances = cdist(centers, centers)

        if normalize:
            radii = []
            for radius_1 in [radius_self, radius_other, combined_radius]:
                
                radii.append([])
                for radius_2 in [radius_self, radius_other, combined_radius]:
                    radii[-1].append((radius_1+ radius_2)/2)

            distances /= radii
        for i in range(3):
            distances[i,i] = np.Inf
        return np.min(distances)

    def get_closest_idx(self, junction_pts):
        pts = [self.original_pts[0], self.original_pts[-1]]
        distances = cdist(pts, junction_pts)
        min_dist = np.min(distances, 1)
        return np.argmin(min_dist)

class junction:
    def __init__(self, network:curvature_network, node_idx:int, distance=10):
        self.network = network
        self.node_idx = node_idx
        self.edges = {}
        self.distance=distance
        self.pts = self.network.nodes(data=True)[self.node_idx]["pts"]
        self.junction_pts = []
        for pt in self.pts:
            if self.network.closest_coord_map is not None:
                if tuple(pt) in self.network.closest_coord_map:
                    self.junction_pts.append(self.network.closest_coord_map[tuple(pt)])
        self.junction_pts = np.concatenate(self.junction_pts)

        self.create_edges()
        # self.solve_edge_connections()
    
    def create_edges(self):
        for e in self.network.edges(self.node_idx, keys=True):
            self.edges[e] = self.network.custom_edges[e]
            
        for e in self.edges.keys():
            idx = self.edges[e].get_closest_idx(self.junction_pts)
            if idx not in self.edges[e].junctions:
                self.edges[e].junctions[idx] = 0
            self.edges[e].junctions[idx] += len(self.edges.keys())-1
        
            # self.edges[e].junctions.append(self)




    def not_allowed(self, idx_1, idx_2):
        return self.edges[idx_1].not_allowed(self.edges[idx_2])

    def add_to_not_allowed(self, idx_1, idx_2):
        self.edges[idx_1].add_to_not_allowed(self.edges[idx_2])
        self.edges[idx_2].add_to_not_allowed(self.edges[idx_1])

    def solve_edge_connections(self, verbose=False, counter=None, max_angle=np.pi*1.3, min_angle=np.pi*0.7):
        def angle_is_appropriate(angle):
            return angle > min_angle and angle < max_angle
        connections = []
        c_id = []
        c_val = []
        to_delete = []
        for edge_idx, e in self.edges.items():
            u,v,k = e.edge_idx
            if u == v:
                if angle_is_appropriate(e.angle(e, (0, -1), verbose=verbose)):

                    connections.append(((u,v,k), (u,v,k)))
                    c_id.append("same")
                    c_val.append(0)
                to_delete.append(edge_idx)
                
        
        edge_direction = {}
        for edge_idx, e in self.edges.items():
            edge_direction[edge_idx] = e.get_closest_idx(self.pts)
        
        group_distance = 3
        for edge_idx in to_delete:
            del self.edges[edge_idx]


        current_idxs = list(self.edges.keys())
        already_combined = set()
        uses = {idx:self.edges[idx].uses for idx in current_idxs}

        while len(current_idxs) not in [0,1]:
            changed=False


            lineness = np.array([[self.edges[idx].get_combined_lineness(self.edges[other_idx], (edge_direction[idx],edge_direction[other_idx])) for other_idx in current_idxs] for idx in current_idxs])
            flattened_lineness = lineness.flatten()
            sorted_idxs = np.argsort(flattened_lineness)
            line_idxs = set()
            angles = {}
            not_allowed_per_angle = set()
            for connection in connections:
                not_allowed_per_angle.add((connection[0],connection[1]))
                not_allowed_per_angle.add((connection[1],connection[0]))
            for idx_1, idx_2 in it.combinations(current_idxs,2):
                angle = self.edges[idx_1].angle(self.edges[idx_2], (edge_direction[idx_1], edge_direction[idx_2]), verbose=verbose)
                angles[(idx_1, idx_2)] = angle
                if not angle_is_appropriate(angle):
                    not_allowed_per_angle.add((idx_1, idx_2))
                    not_allowed_per_angle.add((idx_2, idx_1))
                    # self.add_to_not_allowed(idx_1, idx_2)
                else:
                    pass



            for idx in sorted_idxs:
                if flattened_lineness[idx] < 1:

                    idx_2d = np.unravel_index(idx, lineness.shape)
                    if idx_2d[0] in line_idxs or idx_2d[1] in line_idxs:
                        continue
                    if (current_idxs[idx_2d[0]], current_idxs[idx_2d[1]]) in not_allowed_per_angle:
                        continue
                    if (current_idxs[idx_2d[0]], current_idxs[idx_2d[1]]) in connections or (current_idxs[idx_2d[1]], current_idxs[idx_2d[0]]) in connections:
                        continue
                    changed = True
                    connections.append((current_idxs[idx_2d[0]], current_idxs[idx_2d[1]]))
                    c_id.append("lineness")
                    c_val.append(flattened_lineness[idx])
                    line_idxs.update(idx_2d)
                else:
                    break

            current_idxs = [current_idxs[g] for g in np.arange(len(current_idxs)) if g not in line_idxs]
            if len(current_idxs) <= 1:
                break



            center_distances = np.array([[self.edges[idx].get_curvature_fitting(self.edges[other_idx], (edge_direction[idx],edge_direction[other_idx])) for other_idx in current_idxs] for idx in current_idxs])

            for i in range(len(current_idxs)):
                for j in range(i+1,len(current_idxs)):
                    center_distances[i][j] = (center_distances[i][j] + center_distances[j][i]) / 2

            center_distances_flattened = center_distances.flatten()
            argsorted_center_distances = np.argsort(center_distances_flattened)
            dont_use_anymore = set()


            for idx in argsorted_center_distances:
                argmin_idx = np.unravel_index(idx, center_distances.shape)
                if (current_idxs[argmin_idx[0]], current_idxs[argmin_idx[1]]) in not_allowed_per_angle:
                    continue
                if argmin_idx[0] == argmin_idx[1]:
                    continue
                if current_idxs[argmin_idx[0]] in dont_use_anymore or current_idxs[argmin_idx[1]] in dont_use_anymore:
                    continue
                if (current_idxs[argmin_idx[0]], current_idxs[argmin_idx[1]]) in connections or (current_idxs[argmin_idx[1]], current_idxs[argmin_idx[0]]) in connections:
                    continue
                changed = True
                connections.append((current_idxs[argmin_idx[0]], current_idxs[argmin_idx[1]]))
                c_id.append("center_dist")
                c_val.append(center_distances_flattened[idx])
                # idx = current_idxs.pop()
                # dont_use_anymore.add(current_idxs[argmin_idx[0]])
                # # idx = current_idxs.pop(min(argmin_idx))
                # dont_use_anymore.add(current_idxs[argmin_idx[1]])
                
            break
            current_idxs = [idx for idx in current_idxs if idx not in dont_use_anymore]
            all_idxs = list(self.edges.keys())
            new_idxs = []
            for idx in all_idxs:
                if idx in current_idxs:
                    new_idxs.append(idx)
                else:
                    if uses[idx] > 1:
                        uses[idx] -= 1
                        new_idxs.append(idx)

            current_idxs = new_idxs
            if not changed:
                break

        
        connection_values = []
        for i in range(len(connections)):
            c = connection_value(connections[i], c_val[i], self.node_idx, c_id[i], (edge_direction[connections[i][0]],edge_direction[connections[i][1]], ),self.network.degree(self.node_idx))
            connection_values.append(c)

        return connection_values
        

            

def close_small_holes(image, min_size=20):
    negative_image = image == 0
    nl, _ = label(negative_image, )
    us, ss = np.unique(nl, return_counts=True)
    for u,s in zip(us, ss):
        if s < min_size:
            image[nl == u] = 1
    return image


import traceback
from datetime import datetime


def find_intersection_point(intersection):
    # Perform bitwise AND to find the intersection points
    # intersection = image1 + image2 

    # Find the coordinates of the intersection points
    y_coords, x_coords = np.where(intersection > 1)
    
    if len(x_coords) == 0:
        return None  # No intersection points found

    # Take the average of the coordinates to get the intersection point (approximation)
    x_intersect = int(np.mean(x_coords))
    y_intersect = int(np.mean(y_coords))

    return np.array([y_intersect, x_intersect])



def get_edges(current_closed, current_skeletons, verbose=False, line_width=1):
    edges = []
    instances = []
    closed_counter = 0
    for counter, closed in enumerate(current_closed):
        if not closed:
            
            sk = sknw.build_sknw(skeletonize(current_skeletons[counter]),ring=True, multi=True, full=False)
            cnx = curvature_network(sk, current_skeletons[counter],None, 10, line_width=line_width, idx=counter, verbose=verbose)
            cnx.create_custom_edges()
            # edges.extend(list(cnx.custom_edges.values()))
            
            cnx_instances, es, skeletons, closed, pts = cnx.create_instance_map(None, False, False, False, False, )
            edges.extend(es)
            instances.extend(cnx_instances)
            closed_counter += 1
            
    return edges, instances


def get_edge_combinations(edges, max_dist,other_edges=None,shape=None):
    edge_combinations = []
    other_edges_are_different = True
    if other_edges is None:
        other_edges_are_different = False
        other_edges = edges
    for e_1 in range(len(edges)):
        for e_2 in range(len(other_edges)):

            if e_1 == e_2 and not other_edges_are_different:
                continue
            if edges[e_1] is None or other_edges[e_2] is None:
                continue
            edge_1 = edges[e_1]
            edge_2 = other_edges[e_2]


            img = np.zeros(shape)
            img[edge_1.pts[...,0],edge_1.pts[...,1]] += 1
            img[edge_2.pts[...,0],edge_2.pts[...,1]] += 1
            if np.sum(img > 1) > 0:
                continue
            dists, direction_idxs = edge_1.min_distance(edge_2)
            
            for dist, direction_idx in zip(dists, direction_idxs): 
                if dist < max_dist:
                    edge_combinations.append((e_1, e_2, dist, direction_idx))
    for e, edge in enumerate(other_edges):
        if edge is None:
            continue
        dist = cdist([edge.pts[0]],[edge.pts[-1]])[0][0]

        if dist < max_dist:
            edge_combinations.append((e,e, dist, (0,1)))
    return edge_combinations

def solve_for_layer(testing_image, k, mask, only_closed, data, non_closed_min_size, max_membrane_thickness=None, max_nodes=30):
    testing_skeleton = skeletonize(testing_image)
    sknw_graph = sknw.build_sknw(testing_skeleton,ring=True, multi=True, full=False)
    nodes = sknw_graph.number_of_nodes()
    edges = sknw_graph.number_of_edges()

    if nodes > max_nodes:
        return None
    pts = np.array(np.nonzero(testing_skeleton)).T
    label_pts = np.array(np.nonzero(testing_image)).T
    # try:
    if len(label_pts) > 10000:
        best_skeleton_coord = []
        all_distances = []
        for i in range(len(label_pts) // 10000 + 1 ):
            distances = cdist(pts, label_pts[i*10000:(i+1)*10000])
            best_skeleton_coord.append(np.argmin(distances,0))
            all_distances.append(np.min(distances,0))
        distances = np.concatenate(all_distances)
        best_skeleton_coord = np.concatenate(best_skeleton_coord)
    else:
        distances = cdist(pts, label_pts)
        best_skeleton_coord = np.argmin(distances, 0)
        distances = np.min(distances, 0)
    closest_coord_map = {}
    distances_map = {}
    for idx, argmin in enumerate(best_skeleton_coord):
        
        if tuple(pts[argmin]) not in closest_coord_map:
            closest_coord_map[tuple(pts[argmin])] = []
            distances_map[tuple(pts[argmin])] = []
        closest_coord_map[tuple(pts[argmin])].append(label_pts[idx])
        distances_map[tuple(pts[argmin])].append(distances[idx])
    mean_max_distance = np.mean([np.max(value) for key,value in distances_map.items()])
    mean_max_distance = max(3, round(mean_max_distance)* 2 +1)
    # except:
    mean_max_distance = 7

    
    if max_membrane_thickness is not None:
        mean_max_distance = max_membrane_thickness

    
    closest_coord_map = {key:np.array(value) for key, value in closest_coord_map.items()}
    cnx = curvature_network(sknw_graph, testing_image, closest_coord_map, 10, line_width=mean_max_distance, idx=k, mask=mask)
    try:
        cnx.solve_junctions()
    except Exception as e:
        traceback.print_exc()
    result = cnx.create_instance_map(None, only_closed, label_image=data, non_closed_min_size=non_closed_min_size)
    return result


def get_points(edges, e_1, e_2, direction_idx, i):
    if i == 0:
        edge_1 = edges[e_1]
        edge_2 = edges[e_2]
        edge_1_direction = direction_idx[0]
        edge_2_direction = direction_idx[1]
    else:
        edge_1 = edges[e_2]
        edge_2 = edges[e_1]
        edge_1_direction = direction_idx[1]
        edge_2_direction = direction_idx[0]
    center = edge_1.curvatures[edge_1_direction]["center"]
    radius = edge_1.curvatures[edge_1_direction]["curvature"]
    if edge_1_direction == 0:
        starting_point = edge_1.pts[0]
    else:
        starting_point = edge_1.pts[-1]
    if edge_2_direction == 0:
        end_point = edge_2.pts[0]
    else:
        end_point = edge_2.pts[-1]
    return edge_1, edge_2, edge_1_direction, edge_2_direction, starting_point, end_point, center, radius



def solve_direction_from_to(from_idx, to_idx, edge_pts):
    a = abs(from_idx - to_idx)
    b = len(edge_pts) - a
    if a < b:
        if from_idx > to_idx:
            pts = edge_pts[to_idx:from_idx+1][::-1]
            
        else:
            pts = edge_pts[from_idx:to_idx+1]
    else:
        if from_idx > to_idx:
            pts = np.concatenate([edge_pts[from_idx:], edge_pts[:to_idx+1]])
        else:
            pts = np.concatenate([edge_pts[to_idx:], edge_pts[:from_idx+1]])[::-1]
    return pts

def solve_circle_pts(center, radius, shape, edge_1_weight, edge_2_weight, starting_point, end_point, tangent, verbose=False ):
    if radius > 2**20:
        d = cdist([starting_point], [end_point])[0][0]
        y,x = line_nd((starting_point[0], starting_point[1]), (int(starting_point[0] - tangent[0]*d),int(starting_point[1] - tangent[1]* d)),endpoint=True)
        idxs = np.where((y>=0) & (y < shape[0]-2) & (x >= 0) & (x < shape[1]-2) )
        y = y[idxs]
        x = x[idxs]

    else:
        y,x = circle_perimeter(int(center[0]), int(center[1]), int(radius), shape=[s -2 for s in shape])
    circle_pts = np.array(np.array([y,x]).T)
    
    img = np.zeros([s -2 for s in shape], dtype=np.uint8)
    img[circle_pts[...,0],circle_pts[...,1]] = 1
    
    img = skeletonize(img)
    sk = sknw.build_sknw(img, ring=True, multi=True, full=False)
    # sk = curvature_network(sk, None, None, 0, 10,0)
    sk_edges = sk.edges(data=True, keys=True) 
    c = 0
    if len(sk_edges) == 0:
        return None
        
    for (u,v,k,data) in sk_edges:
        c += 1
        edge_pts = data["pts"]          
        distances = cdist(np.array([starting_point, end_point]), edge_pts)
        idxs = np.argmin(distances, 1)
        from_idx = idxs[0]
        to_idx = idxs[1]
        # TODO: Better decide direction
        
        pts = solve_direction_from_to(from_idx, to_idx, edge_pts)

        current_intersection_image = np.zeros(shape, dtype=np.uint8)
        for pt in pts:
            current_intersection_image[pt[0]:pt[0]+3, pt[1]:pt[1]+3] = 1
        

        
        path = np.sqrt(np.sum(np.diff(pts, axis=0)**2,axis=1))
        unconnected =  np.any(path > 1.5)
        new_weight = np.sum(path)
        too_long = new_weight > edge_1_weight and new_weight > edge_2_weight

    return current_intersection_image, pts, too_long, unconnected


def get_angles(combining_pts, edges, e_1, e_2, direction_idx):
    if len(combining_pts[0]) > 1 and len(combining_pts[1]) > 1:
        tangents_1 = calc_tangent(max(2, len(combining_pts[0])//2), combining_pts[0],)
        tangents_2 = calc_tangent(max(2, len(combining_pts[1])//2), combining_pts[1],)
        inbetween_angle = np.arccos(np.clip(np.dot(tangents_1[1], tangents_2[1]),-1,1))
        angle_method = 0
        angle_1 = np.arccos(np.clip(np.dot(tangents_1[0], edges[e_1].tangents[direction_idx[0]]),-1,1))
        angle_2 = np.arccos(np.clip(np.dot(tangents_2[0], edges[e_2].tangents[direction_idx[1]]),-1,1))
    elif len(combining_pts[0]) > 1:
        tangents_1 = calc_tangent(max(2, len(combining_pts[0])//2), combining_pts[0],)
        tangents_2 = edges[e_2].tangents[direction_idx[1]]
        inbetween_angle = np.arccos(np.clip(np.dot(tangents_1[1], tangents_2),-1,1))
        angle_method = 1
        angle_1 = np.arccos(np.clip(np.dot(tangents_1[0], edges[e_1].tangents[direction_idx[0]]),-1,1))

        if direction_idx[1] == 0:
            line_vector = combining_pts[0][len(combining_pts[0]) // 2] - edges[e_2].pts[0] 
        else:
            line_vector = combining_pts[0][len(combining_pts[0]) // 2] - edges[e_2].pts[-1] 
        line_vector = line_vector / np.linalg.norm(line_vector)
        angle_2 = np.arccos(np.clip(np.dot(line_vector, edges[e_2].tangents[direction_idx[1]]),-1,1))
    elif len(combining_pts[1]) > 1:
        tangents_1 = edges[e_1].tangents[direction_idx[0]]
        tangents_2 = calc_tangent(max(2, len(combining_pts[1])//2), combining_pts[1],)
        inbetween_angle = np.arccos(np.clip(np.dot(tangents_1, tangents_2[1]),-1,1))
        angle_method = 2
        if direction_idx[0] == 0:
            line_vector = combining_pts[1][len(combining_pts[1]) // 2] - edges[e_1].pts[0] 
        else:
            line_vector = combining_pts[1][len(combining_pts[1]) // 2] - edges[e_1].pts[-1] 
        angle_1 = np.arccos(np.clip(np.dot(line_vector, edges[e_1].tangents[direction_idx[0]]),-1,1))
        angle_2 = np.arccos(np.clip(np.dot(tangents_2[0], edges[e_2].tangents[direction_idx[1]]),-1,1))
    else:
        inbetween_angle = None
        angle_method = 3
        angle_1 = None
        angle_2 = None
    return angle_1, angle_2, inbetween_angle, angle_method


def create_new_image(edges, e_1, e_2, combining_pts, direction_idx, shape):
    new_pts = edges[e_1].pts

    if direction_idx[0] == 0:
        new_pts = new_pts[::-1]
    if len(combining_pts[0]) > 2:
        if cdist([new_pts[-1]], [combining_pts[0][0]])[0][0] > 1.5:
            line_pts = np.array(np.array(line_nd(new_pts[-1],combining_pts[0][0],endpoint=False)).T)
            if len(line_pts) > 1:
                new_pts = np.concatenate((new_pts, line_pts))
                
        new_pts = np.concatenate((new_pts, combining_pts[0]))
    if len(combining_pts[1]) > 2:
        if cdist([new_pts[-1]], [combining_pts[1][-1]])[0][0] > 1.5:
            line_pts = np.array(np.array(line_nd(new_pts[-1],combining_pts[1][-1],endpoint=False)).T)
            if len(line_pts) > 1:
                new_pts = np.concatenate((new_pts, line_pts))
                
        new_pts = np.concatenate((new_pts, combining_pts[1][::-1]))
    
        

    
    next_pts = edges[e_2].pts
    if direction_idx[1] == 1:
        next_pts = next_pts[::-1]
    if cdist([new_pts[-1]], [next_pts[0]])[0][0] > 1.5:
        line_pts = np.array(np.array(line_nd(new_pts[-1],next_pts[0],endpoint=False)).T)
        if len(line_pts) > 1:
            new_pts = np.concatenate((new_pts, line_pts))
    
    closed = e_1 == e_2
    if e_1 != e_2:
        new_pts = np.concatenate((new_pts, next_pts))
        if cdist([new_pts[0]],[new_pts[-1]] )[0][0] < 10:
            
            line_pts = np.array(np.array(line_nd(new_pts[-1],new_pts[0],endpoint=False)).T)
            new_pts = np.concatenate((new_pts, line_pts))
            closed = True
    img = np.zeros(shape)
    img[new_pts[...,0], new_pts[...,1]] = 1
    return img, closed


def solve_open_membranes(edge_combination, shape, edges,verbose=False):
    e_1, e_2, dist, direction_idx = edge_combination

    intersection_image = np.zeros(shape, dtype=np.uint8)
    intersection_image = np.pad(intersection_image, 1)
    direction_pts = []
    too_long = False
    unconnected = False
    
    for i in range(2):
        edge_1, edge_2, edge_1_direction, edge_2_direction, starting_point, end_point, center, radius = get_points(edges, e_1, e_2, direction_idx, i)
        
        edge_1_weight = edge_1.network.get_edge_data(*edge_1.edge_idx)["weight"]
        edge_2_weight = edge_2.network.get_edge_data(*edge_2.edge_idx)["weight"]
        
        tangent = edge_1.tangents[edge_1_direction]
        res = solve_circle_pts(center, radius, intersection_image.shape, edge_1_weight, edge_2_weight,  starting_point, end_point, tangent, verbose=e_1 == 30 and e_2 == 50 and direction_idx == (1,1))
        ################
        if res is None:
            return False, None, None, 0
        current_circle_image, directional_pts, current_too_long, current_unconnected = res
        direction_pts.append(directional_pts)
        intersection_image += current_circle_image
        too_long = too_long or current_too_long
        unconnected = unconnected or current_unconnected
        
    intersection = find_intersection_point(intersection_image[1:-1,1:-1])
    
    if intersection is not None and not too_long and not unconnected:
        
        combining_pts = []
        for i in range(2):
            d = cdist(direction_pts[i], [intersection])
            closest_idx = np.argmin(d)
            combining_pts.append(direction_pts[i][:closest_idx+1])

        conc_combining_pts = np.concatenate(combining_pts)
        ds = [cdist(c, np.concatenate((edge_1.pts, edge_2.pts))) for c in combining_pts]
        average_ds = np.array([np.mean(np.min(d,1)) for d in ds])

        # Look for too small circle pts
        if np.any(average_ds > 1.5):
            angle_1, angle_2, inbetween_angle, angle_method = get_angles(combining_pts, edges, e_1, e_2, direction_idx)
            


            if all([a is not None for a in [angle_1, angle_2, inbetween_angle]]) and all([angle_is_appropriate(a) for a in [angle_1, angle_2, inbetween_angle]]):
                

                
                img_combined = np.zeros(shape)
                pts = edges[e_1].pts
                img_combined[pts[...,0],pts[...,1]] = 1
                pts = edges[e_2].pts
                img_combined[pts[...,0],pts[...,1]] = 1
                img_combined[conc_combining_pts[...,0],conc_combining_pts[...,1]] = 5
                


                img_combined, closed = create_new_image(edges, e_1, e_2, combining_pts, direction_idx, shape)

            

       
                return True, img_combined, closed, (angle_1, angle_2, inbetween_angle)
    return False, None, None, 0




def solve_skeleton_per_job(l, mask, only_closed, non_closed_min_size=0, name=None, max_membrane_thickness=None, connect_parts=True, max_nodes=30):
    def combination_sort_function(combination):
        edge_combination, image, closed_combined, angles = combination
        if closed_combined:
            return (0, np.sum(np.abs([angle - np.pi for angle in angles])) )
        else:
            return (1, np.sum(np.abs([angle - np.pi for angle in angles])) )
    # if mask is not None:
    #     mask = (mask == 0)*1
    #     mask = binary_dilation(mask, np.ones((3,3)))
    current_instances = []
    current_edges = []
    current_labels = []
    current_skeletons = []
    current_closed = []
    all_pts = []
    if isinstance(l, (str, Path)):
        data = mrcfile.open(l, permissive=True).data * 1
    else:
        data = l
    
    data = close_small_holes(data)


    l_, ls_ = label(data, np.ones((3,3)))
    shape = data.shape

    for k in np.unique(l_):
        if k == 0:
            continue
        testing_image = l_ == k
        result = solve_for_layer(testing_image, k, mask, only_closed, data, non_closed_min_size, max_membrane_thickness)
        
        if result is None:
            continue
        cnx_instances, edges, skeletons, closed, pts = result
        if len(cnx_instances) > 0:
            current_instances.append(cnx_instances)
            # current_edges.extend(edges)
            current_labels.append(k)
            current_skeletons.extend(skeletons)
            current_closed.extend(closed)
            all_pts.extend(pts)
    
    if len(current_instances) > 0:
        current_instances = np.concatenate(current_instances)
    if connect_parts:
        mean_max_distance = 7
        if max_membrane_thickness is not None:
            mean_max_distance = max_membrane_thickness
        current_instances = np.array([ci for ci, closed in zip(current_instances, current_closed) if closed])
        edges, new_instances = get_edges(current_closed, current_skeletons, line_width=mean_max_distance)
        
        edge_combinations = get_edge_combinations(edges, max_dist=100, shape=shape)
        solved = set()
        edge_combination_solved = set()
        combined_combinations = []
        for edge_combination in edge_combinations:
            e_1, e_2, dist, edge_direction = edge_combination
            if e_1 in solved or e_2 in solved or (e_1, e_2) in edge_combination_solved or (e_2, e_1) in edge_combination_solved:
                continue
            combined, image, closed_combined, angles = solve_open_membranes(edge_combination, shape, edges)
            
            if combined:
                combined_combinations.append((edge_combination, image, closed_combined, angles))

        combination_counter = 0
        while True:

            combination_counter += 1
            # path.mkdir(exist_ok=True)

            combined_combinations = sorted(combined_combinations, key=lambda x: combination_sort_function(x))
            to_remove = set()
            found_sth = False
            for counter, combination in enumerate(combined_combinations):
                edge_combination, image, closed_combined, angles = combination
                e_1, e_2, dist, edge_direction = edge_combination
                if e_1 in solved or e_2 in solved or (e_1, e_2) in edge_combination_solved or (e_2, e_1) in edge_combination_solved:
                    to_remove.add(counter)
                    continue
                if found_sth:
                    continue
                to_remove.add(counter)
                solved.add(e_1)
                solved.add(e_2)
                edge_combination_solved.add((e_1, e_2)) 
                edge_combination_solved.add((e_2, e_1))
                new_edges, new_inst = get_edges([False], [image], verbose=edge_combination[0] == 40 and edge_combination[1] == 41, line_width=mean_max_distance)
                new_instances.append(new_inst[0])
                current_skeletons.append(image)

                if not closed_combined:
                    
                    new_edge_combinations = get_edge_combinations(edges, 100, [new_edges[0]], shape=shape)
                    

                    edges.append(new_edges[0])
                    

                    for new_edge_combination in new_edge_combinations:
                        new_e_1, new_e_2, new_dist, new_edge_direction = new_edge_combination
                        
                        if new_e_1 == new_e_2:
                            new_edge_combination = (len(edges)-1, len(edges)-1, new_dist, new_edge_direction)
                            
                        else:
                            new_edge_combination = (new_e_1, len(edges)-1, new_dist, new_edge_direction)
                            
                        edge_combinations.append(new_edge_combination)
                        edge_combination_solved.add((len(edges)-1, e_1))
                        edge_combination_solved.add((len(edges)-1, e_2))

                        combined, image, closed_combined, angles = solve_open_membranes(edge_combinations[-1], shape, edges,verbose=combination_counter == 5)

                        if combined:
                            combined_combinations.append((new_edge_combination, image, closed_combined, angles))
                    found_sth = True
                else:
                    edges.append(None)
            else:
                if not found_sth:
                    break

            combined_combinations = [comb for counter, comb in enumerate(combined_combinations) if counter not in to_remove]
            

        new_instances = [ni for counter, ni in enumerate(new_instances) if counter not in solved]

        if len(current_instances) > 0 and len(new_instances) > 0:
            current_instances = np.concatenate((current_instances, new_instances))
        elif len(current_instances) > 0:
            pass
        else:
            current_instances = np.array(new_instances)


    if len(current_instances) == 0:
        return np.zeros((0, *data.shape)), np.zeros((0, *data.shape))
    return current_instances, None
    return current_instances, np.array(current_skeletons)

def solve_skeletons(label_files_or_images, masks=None, only_closed=False, njobs=1, non_closed_min_size=0,max_nodes=30):

    if masks is None:
        masks = [None for _ in label_files_or_images]
    if njobs > 1:
        with mp.Pool(njobs) as pool:
            result = [pool.apply_async(solve_skeleton_per_job, [l, mask, only_closed,non_closed_min_size, None, None, True, max_nodes]) for l, mask in zip(label_files_or_images, masks)]
            results = [res.get() for res in result]
            instances = []
            skeletons = []
            for (ins, skel) in results:
                instances.append(ins)
                skeletons.append(skel)
             
    else:
        instances = []
        skeletons = []
        for (ins, skel) in [solve_skeleton_per_job(l, mask, only_closed, non_closed_min_size, None, None, True, max_nodes) for l, mask in zip(label_files_or_images, masks)]:
            instances.append(ins)
            skeletons.append(skel)

            
    return instances, skeletons

