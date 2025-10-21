
import numpy as np
from matplotlib import pyplot as plt
 
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="networkx backend defined more than once: nx-loopback")
    import networkx as nx
import warnings
from skimage.draw import line



import copy
import multiprocessing as mp

COUNTER = 0
CURRENT_LAB = 0



def find_path_for_nodes(nodes, G, max_depth):
    return_paths = []
    node_1, node_2  = nodes
    if node_1 == node_2:
        return return_paths
    paths = nx.all_simple_edge_paths(G, node_1, node_2,max_depth)
    for path in paths:
        path_node_counter = {}
        for n in path:
            u,v = n[:2]
            if u not in path_node_counter:
                path_node_counter[u] = 0
            if v not in path_node_counter:
                path_node_counter[v] = 0
            path_node_counter[u] += 1
            path_node_counter[v] += 1
        
        nr_of_single_nodes = len([i for i in path_node_counter.values() if i==1])


        if nr_of_single_nodes != 2:
            continue
        if any([i > 2 for i in path_node_counter.values()]):
            continue
        new_path = []
        for edge in path:
            
            edge = (min(edge[:2]), max(edge[:2]), edge[2])
            new_path.append(edge)
        new_path = Network_path(G, new_path, False, name=G.name +"_"+ str(len(return_paths)))
        return_paths.append(new_path)
        
    return return_paths



def get_shared_nodes(edge_1, edge_2):
    return [node for node in edge_1[:2] if node in edge_2[:2]]



def next_to_point( pts_list, pt, first=True):
    """Return a list of points so that the first point is next to a specified point (if first==True, otherwise reverse the list)"""
    
    if (pt == pts_list[0]).all(1).any():
        if not first:
            pts_list = pts_list[::-1]
    elif (pt == pts_list[-1]).all(1).any():
        if first:
            pts_list = pts_list[::-1]
    else:
        warnings.warn(f"{pts_list} is not close to {pt}")
    return pts_list




def find_cycles(start_node, max_depth,G):
        current_cycles = []
        for cycle in find_next_node(start_node, start_node, [], set(), max_depth, G):
                
            if cycle is not None:
                current_cycles.append(cycle)
        return current_cycles

def find_next_node(current_node, start_node, nodes_before, edges_before, max_depth, G):
    
    nodes_before = copy.copy(nodes_before)
    
    nodes_before.append(current_node)
    
    if len(nodes_before) > max_depth:
        yield None
        return
    current_edges = G.edges(current_node,keys=True)

    for edge in current_edges:
        
        edge = (min(edge[:2]),max(edge[:2]), edge[2])
        
        
        if edge in edges_before:
            yield None
        other_node = get_other_node(edge,current_node) 
        
        if other_node == start_node:
            
            
            yield [edge]
        elif other_node in nodes_before:

            yield None
        else:
            current_edges_before = copy.copy(edges_before)
            current_edges_before.add(edge)
            for result in find_next_node(other_node, start_node, nodes_before, current_edges_before, max_depth, G):
                if result is not None:

                    to_yield = [edge]
                    to_yield.extend(result) 
                    yield to_yield
            yield None



def find_all_cycles_multi(G:nx.MultiGraph, max_depth=15):
    def search_for_next_edge(cycle_set, node):
        for element in cycle_set:
            if element[0] == node or element[1] == node:
                return element

    


    start_nodes = [i[0] for i in G.degree if i[1] > 1]
    cycles = []
    filtered_cycles = []
    

    if G.pool is not None and len(G.nodes) > 5:
        
        nx_copy = nx.MultiGraph(G)
        for attr in ["pts"]:
            nx.set_edge_attributes(nx_copy,  None, attr)
        result = [G.pool.apply_async(find_cycles, args=[node, max_depth, nx_copy]) for node in start_nodes]

        results = [res.get() for res in result]
        for res in results:
            cycles.extend(res)
    else:
        for node_counter, node in enumerate(start_nodes):
            for cycle in find_next_node(node, node, [], set(), max_depth, G):
                
                if cycle is not None:
                    cycles.append(cycle)

    cleaned_cycles = set()

    for cycle in cycles:
        cleaned_cycle = set()
        for edge in cycle:
            new_edge = (min(edge[:2]),max(edge[:2]), edge[2])
            cleaned_cycle.add(new_edge)
        cleaned_cycles.add(frozenset(cleaned_cycle))
    
    for cycle in cleaned_cycles:
        if len(cycle) == 1:
            edge = next(iter(cycle))
            if edge[0] == edge[1]:
                filtered_cycles.append([edge])
        else:
            cycle = set(cycle)
            current_edge = cycle.pop()
            filtered_cycle = [current_edge]
            current_node = current_edge[0]
            while len(cycle) > 0:
                
                
                next_edge = search_for_next_edge(cycle, current_node)
                current_node = get_other_node( next_edge,current_node)
                filtered_cycle.append(next_edge)
                current_edge = next_edge
                cycle.remove(next_edge)
            filtered_cycles.append(filtered_cycle)

    return filtered_cycles




def get_other_node(edge, node):
    """Get the node of and edge which is not a specified node"""
    if edge[0] == node:
        return edge[1]
    return edge[0]


def turn_edge_away_from_pt(edge, pt):
    def dist(p1, p2):
        return np.sum(np.sqrt(np.sum((p1-p2)**2)))
    distances = []
    distances.append(dist(edge[0], pt))
    distances.append(dist(edge[-1], pt))
    min_dist = np.argmin(distances)
    if min_dist == 0 :
        return edge
    
    return edge[::-1]

def turn_edge_to_edge(edge_1, edge_2):
    def dist(p1, p2):
        return np.sum(np.sqrt(np.sum((p1-p2)**2)))
    
    distances = []
    distances.append(dist(edge_1[0], edge_2[0]))
    distances.append(dist(edge_1[-1], edge_2[0]))
    distances.append(dist(edge_1[0], edge_2[-1]))
    distances.append(dist(edge_1[-1], edge_2[-1]))
    min_dist = np.argmin(distances)
    if min_dist == 0 or min_dist == 2:
        return edge_1[::-1]
    
    return edge_1




class Network_path:
    def __init__(self, sknw_graph:nx.MultiGraph, path:list, is_cycle=False, belongs_to_cycle=None, name=None):
        
        self.edge_path = None
        self.node_path = None
        self.is_cycle = is_cycle
        self.name = name
        
        if belongs_to_cycle is None:

            self.belongs_to_cycle = self.is_cycle
        else:
            self.belongs_to_cycle = belongs_to_cycle
        self.sknw_graph = sknw_graph
        
        self.length_ = None


        self.create_edge_path(path)
        self.create_node_path()
        self.current_index = 0
        self.get_edge_path = True
        self.pts_ = None
        

    def create_edge_path(self, edges):
        path = []
        for edge in edges:
            u,v,k = edge
            path.append((min(u,v), max(u,v), k))
        self.edge_path = path

    
    def create_node_path(self):
        current_edge = self.edge_path[0]
        node_path = []
        if len(self.edge_path) > 1:
            for edge_counter, next_edge in enumerate(self.edge_path[1:]):
                if edge_counter == 0:
                    nodes = get_shared_nodes(current_edge, next_edge)
                    node_path.extend([node for node in current_edge[:2] if node not in nodes])
                # elif edge_counter == len(self.edge_path[1:]) - 1:
                #     nodes = get_shared_nodes(current_edge, next_edge)
                #     node_path.extend([node for node in next_edge[:2] if node not in nodes])
                node_path.extend(get_shared_nodes(current_edge, next_edge))
                current_edge = next_edge
            node_path.append(get_other_node(current_edge, node_path[-1]))
            
        else:
            node_path = current_edge[:2]
        self.node_path = node_path



    @property
    def length(self):
        if self.length_ is None:
            pts = self.get_points()
            if self.is_cycle and (pts[0][0] != pts[-1][0] or pts[0][1] != pts[-1][1]):
                pts = np.concatenate((pts, pts[0:1]))
            length = np.sum(np.sqrt(np.sum(np.diff(pts, axis=0)**2,axis=1)))
            self.length_ = length
        return self.length_

    
    
    @property
    def stats(self):
        return {"length":self.length, "ellipse_fitting":self.ellipse_fitting}

    def __repr__(self) -> str:
        return str(self.edge_path)
    
    def __getitem__(self, key) -> tuple:
        return self.edge_path[key]

    def __len__(self) -> int:
        return len(self.edge_path)

    def get_points(self):
        if self.pts_ is None:
        
        # path = self.remove_everything_but_path(path)
            if len(self.edge_path) == 1:
                pts = self.sknw_graph.get_edge_data(*self.edge_path[0])["pts"]

            else:
                first_edge = self.sknw_graph.get_edge_data(*self.edge_path[0])["pts"]
                second_edge = self.sknw_graph.get_edge_data(*self.edge_path[1])["pts"]

                pts = turn_edge_to_edge(first_edge, second_edge)

                for edge_counter, edge in enumerate(self.edge_path[1:]):
                    edge = self.sknw_graph.get_edge_data(*edge)["pts"]
                    next_pts = turn_edge_away_from_pt(edge, pts[-1])

                    line_pts = np.array(line(pts[-1][0],pts[-1][1], next_pts[0][0], next_pts[0][1])).T
                    if len(line_pts) > 2:
                        pts = np.concatenate((pts, line_pts[1:-1]))
                    pts = np.concatenate((pts, next_pts))
            if self.is_cycle:
                line_pts = np.array(line(pts[-1][0],pts[-1][1], pts[0][0],pts[0][1])).T
                if len(line_pts) > 2:
                    pts = np.concatenate((pts, line_pts[1:-1]))

            pts = pts.astype(np.int32)
            dif = np.sum(np.abs(np.diff(pts, axis=0)), 1)
            idx_to_keep = dif > 0
            idx_to_keep = np.concatenate(([True], idx_to_keep))
            # for idx, pt in enumerate(pts[1:]):
            #     if pts[idx][0] == pt[0] and pts[idx][1] == pt[1]:
            #         continue
            #     else:
            #         idx_to_keep.append(idx+1)

            pts = pts[idx_to_keep]
            if self.is_cycle:
                if pts[0][0] == pts[-1][0] and pts[0][1] == pts[-1][1]:
                    pts = pts[:-1]


            self.pts_ = pts
        return self.pts_
    


    def __iter__(self):
        self.current_index = 0
        self.get_edge_path = True
        return self

    def __next__(self):
        if self.edge_path is None:
            raise StopIteration
        if self.get_edge_path:
            if self.current_index < len(self.edge_path):
                
                x = self.edge_path[self.current_index]
                self.current_index += 1
                return x
        else:
            if self.current_index < len(self.node_path):
                x = self.node_path[self.current_index]
                self.current_index += 1
                return x
        raise StopIteration
    
    @property
    def nodes(self):
        self.current_index = 0
        self.get_edge_path = False
        return self




class Subgraph(nx.MultiGraph):
    """A class to clean up segmentation which has multiple membranes fused together """
    def __init__(self, incoming_graph_data, njobs=1, pixel_size=1,shape=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.pixel_size = pixel_size
        self.paths = []
        self.cycles = []
        self.too_short_cycles = []
        self.pool = None
        self.shape = shape


    def remove_degree_two_nodes(self):
        """Remove nodes which have degree of 2 and combine the two edges of the node"""
        # Find nodes to remove
        node_to_remove = [i[0] for i in self.degree if i[1] == 2]

        # Combine edges
        for node in node_to_remove:
            self.combine_edges(node)

    def combine_edges(self, node):
        """Combine the two edges of a node two one edge and remove the node"""
        
        nodes_data = self.nodes(data=True)[node]
        
        if "pts" not in nodes_data:
            warnings.warn(f"pts not in node_data {node}")
        edges = list(self.edges(node, data=True,keys=True))

        if len(edges) == 1:
            if edges[0][0] == edges[0][1]:
                pt_1 = edges[0][-1]["pts"][0]
                pt_2 = edges[0][-1]["pts"][-1]
                new_pts = np.array(line(*pt_1, *pt_2)).T
                nx.set_node_attributes(self, {node:new_pts}, "pts")
            return
        
        # Get the points from the two edges so that the last point of the first edge is next to the node and the first point of the second edge is next to the node
        first_pts_list = next_to_point(edges[0][-1]["pts"], nodes_data["pts"], False)
        
        second_pts_list = next_to_point(edges[1][-1]["pts"], nodes_data["pts"], True)
        

        # New node points will be the line between the two edges
        yy,xx = line(*first_pts_list[-1], *second_pts_list[0])

        

        # If the two end_points are next to each other use them as a list, otherwise add the points between them aswell
        if len(yy) > 2:
            extra_pts = [(y,x) for y,x in zip(yy[1:-1],xx[1:-1])]
            new_pts = np.concatenate((first_pts_list, extra_pts, second_pts_list))
        else:
            new_pts = np.concatenate((first_pts_list, second_pts_list))
        
        
        dif = np.sum(np.abs(np.diff(new_pts, axis=0)), 1)
        idx_to_keep = dif > 0
        idx_to_keep = np.concatenate(([True], idx_to_keep))
        new_pts = new_pts[idx_to_keep]
        # Get the nodes which are not the current node from the two adjacent edges
        other_nodes = [get_other_node(edge, node) for edge in edges]

        # Add an edge between the two other nodes and remove the current node
        new_weight = np.sum(np.sqrt(np.sum(np.diff(new_pts[1:-1], axis=0)**2,axis=1)))
        

        key = self.add_edge(*other_nodes, pts=new_pts, weight=new_weight, cycle_counter=0, path_counter=0, mean_thickness=0, multi_edge_prob=0)

        self.remove_node(node)



    def find_best_edge(self):
        
        while self.number_of_edges() > 1:
            edges = self.edges(data=True, keys=True)
            edges = self.sorted_by(edges, "weight")
            for u,v,key,edge_data in edges:
            
                 
                if u == v:         
                    self.remove_edge(u,v,key)
                    break

                # Check if any node of the edge has a degree of 1
                leafview = self.is_leaf_edge((u,v))
                
                # If one node has degree one, remove that node and the edge
                if any([i[1] for i in leafview]):
                    if leafview[0][1]:
                        leaf_node = leafview[0][0]
                        better_node = leafview[1][0]
                    else:
                        leaf_node = leafview[1][0]
                        better_node = leafview[0][0]
                    self.remove_node(leaf_node)
                    # If the other node has a degree of two now, combine the edges of that node and remove it
                    if self.degree(better_node) == 2:
                        self.combine_edges(better_node)
                    break
            else:
                u,v,key,edge_data = edges[0]
                self.remove_edge(u,v,key)
                self.remove_degree_two_nodes()

        self.remove_degree_two_nodes()
        # Make end nodes to a single pixel
        self.remove_large_end_nodes()





    def remove_small_outer_edges(self, max_length=300):
        """Remove small edges with a length of threshold"""
        max_length = max_length / self.pixel_size
        while True:
            # Will break if no edges to remove have been found

            # Get edges and sort by weight --> remove smallest edges first
            edges = self.edges(data=True, keys=True)
            edges = self.sorted_by(edges, "weight")
            for u,v,key,edge_data in edges:

                if "weight" not in edge_data:
                    warnings.warn(f"weight not found in edge: {(u,v,key)}")
                    continue
                # Only remove the edge if its below the threshold
                if edge_data["weight"] > max_length:
                    continue
                
                if u == v:
                    if self.number_of_nodes() == 1:
                        return
                    self.remove_edge(u,v,key)
                    break
                if self.number_of_edges() == 1:
                    return
                # Check if any node of the edge has a degree of 1
                leafview = self.is_leaf_edge((u,v))
                
                # If all have degree 1 remove both nodes and the edge
                if all([i[1] for i in leafview]):
                    self.remove_nodes_from([i[0] for i in leafview])
                    break

                # If one node has degree one, remove that node and the edge
                elif any([i[1] for i in leafview]):
                    if leafview[0][1]:
                        leaf_node = leafview[0][0]
                        better_node = leafview[1][0]
                    else:
                        leaf_node = leafview[1][0]
                        better_node = leafview[0][0]
                    self.remove_node(leaf_node)
                    # If the other node has a degree of two now, combine the edges of that node and remove it
                    if self.degree(better_node) == 2:
                        self.combine_edges(better_node)
                    break
            else:
                break

        self.remove_degree_two_nodes()
        # Make end nodes to a single pixel
        self.remove_large_end_nodes()
        
    
    def is_leaf_edge(self, edge):
        """Check if an edge has nodes with degree one and returns a bool list for the nodes"""
        degreeview = self.degree(edge)
        return [(i[0], i[1] == 1) for i in degreeview]



    def sorted_by(self, what_to_use, attr):
        """sort tuple of (obj, dict) by an attribute in dict"""
        def get_sort_by_attr(attr):
            def sort_by_attr(obj):
                if hasattr(obj, attr):
                    return getattr(obj, attr)
                else:
                    return obj[-1][attr]
            return sort_by_attr
        if isinstance(what_to_use, str):
            paths = self.get_paths_and_or_cycles(what_to_use)
        else:
            paths = what_to_use
        paths = sorted(paths, key=get_sort_by_attr(attr))
        if isinstance(what_to_use, str):
            self.set_paths_and_or_cycles(what_to_use, paths)
        else:
            return paths


    def sort_by(self, what_to_use, attribute):
        
        attribute = attribute.replace(" ", "_").lower()
        return self.sorted_by(what_to_use, attribute)
                

    
    def remove_large_end_nodes(self):
        """Nodes in this setting can have multiple pixel points. This function makes nodes of degree 1 to single points"""
        for node, data in self.nodes(data=True):
            if self.degree(node) == 1 and len(data["pts"] > 1):
                edges =list(self.edges(node, data=True))
                
                edge_data = edges[0][2]
                new_point = next_to_point(edge_data["pts"], data["pts"], True)[0]
                new_point = np.expand_dims(new_point, 0)
                nx.set_node_attributes(self, {node: {"pts":new_point, "o":new_point}})

        
    


    def find_cycles(self, max_depth, min_length=0):
        """Finds cycle in graph which are not overlapping by too much"""
        min_length = min_length / self.pixel_size
        cycles = [Network_path(self, cycle, True, name=self.name +"_" + str(counter)) for counter, cycle in enumerate(find_all_cycles_multi(self, max_depth))]
        for cycle in cycles:


            if cycle.length >= min_length:
                self.cycles.append(cycle)
            else:
                self.too_short_cycles.append(cycle)
            # else:
            #     self.add_to_too_short(cycle)
        

        return None

   
    def find_paths(self, max_depth=10, min_length=120):
        """Find paths which do not use edge of cycles unless the edge_probability of that edge is above multi_edge_threshold (so thicker than usual)"""
        

        min_length = min_length / self.pixel_size

    


        # all_paths = set()

        # Find all Paths (is brute force, but usually not many nodes and edges are available, so should not be a problem)
        nodes = [i for i in self.nodes]
        if len(nodes) == 1:
            return
        nodes_combination = []
        for node_1_idx, node_1 in enumerate(nodes[:-1]):
            for node_2 in nodes[node_1_idx+1:]:
                nodes_combination.append((node_1, node_2))
        paths = []
        if self.pool is not None:
            nx_copy = nx.MultiGraph(self)
            for attr in ["pts"]:
                nx.set_edge_attributes(nx_copy,  None, attr)
            
            result = [self.pool.apply_async(find_path_for_nodes, args=[node_comb, nx_copy, max_depth]) for node_comb in nodes_combination]
            results = [res.get() for res in result]
            
            for res in results:
                for path in res:
                    path:Network_path
                    path.sknw_graph = self
                    if path.length >= min_length:
                        paths.append(path)
                
        else:
            results = []
            for node_comb in nodes_combination:
                results.append(find_path_for_nodes(node_comb, self, max_depth))
            for res in results:
                for path in res:
                    path:Network_path
                    path.sknw_graph = self
                    if path.length >= min_length:
                        paths.append(path)        
        self.paths = paths

    def get_skeleton(self):
        image = np.zeros(self.shape)
        for e in self.edges(data=True,keys=True):
            u,v,k,data = e
            y,x = data["pts"].T
            image[y,x] = 1
        for n in self.nodes(data=True):
            key, data = n
            y,x = data["pts"].T
            image[y,x] = 1

        return image