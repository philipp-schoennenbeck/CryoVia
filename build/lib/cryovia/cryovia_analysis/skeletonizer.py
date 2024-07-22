import numpy as np
from scipy.ndimage import label,convolve
from cryovia.cryovia_analysis.sknw_utils import Subgraph as SG
from skimage.morphology import skeletonize
import multiprocessing as mp
import cv2
import sparse
import cryovia.cryovia_analysis.sknw as sknw
from skimage.measure import label as skilabel 

from scipy.spatial.distance import cdist

try:
    from skimage.draw import disk, line
except:
    from skimage.draw import circle as disk
# from skimage.measure import label
from skimage.transform import resize
from skimage import img_as_bool
from skimage.filters import frangi
import getpass
import multiprocessing as mp
from matplotlib import pyplot as plt

# from cryovia.gui.segmentation_files.prep_training_data import predictMiddleFrangi


lookup_table = {0:[1,3], 2:[1,5],6:[3,7],8:[5,7]}





class Skeletonizer:
    def __init__(self, label_image, pixel_size, njobs=1,  max_nodes=5,):
        self.pixel_size = pixel_size
        self.label_image = label_image
        self.njobs = njobs
        
        
        self.max_nodes = max_nodes
    
    def find_skeletons(self, pool=None):
        all_skeleton_points = []
        for i in range(self.label_image.shape[0]):
                
            current_image = self.label_image[i].todense()
            current_skel = skeletonize(current_image)

            label_sknw = SG(sknw.build_sknw(current_skel, multi=True, iso=False, ring=True, full=False), njobs=self.njobs, pixel_size=self.pixel_size)

            label_sknw.find_best_edge()
            label_sknw.find_cycles(self.max_nodes, 1)

            if len(label_sknw.cycles)==0:
                label_sknw.find_paths(self.max_nodes, 1)
                if len(label_sknw.paths) == 0:
                    if len(label_sknw.too_short_cycles) > 0:
                        path = label_sknw.too_short_cycles[0]
                    
                else:
                    path = label_sknw.paths[0]
            else:
                path = label_sknw.cycles[0]
            pts = path.get_points()
            all_skeleton_points.append(pts)



        return all_skeleton_points




class RidgeDetector:
    def __init__(self, label_image, image, pixel_size, njobs=1, name=None,max_nodes=5, low=8, high=38):
        self.pixel_size = pixel_size
        self.label_image = label_image
        self.njobs = njobs
        self.name = name
        self.max_nodes = max_nodes
        self.image = image
        self.low = low
        self.high = high
    
    def find_skeletons(self, pool):

        
        ridge_skeletons = ridge_detection(self.label_image, self.image, self.njobs, self.pixel_size, self.low,self.high,pool,  self.name) 

        all_skeleton_points = []
        for i in range(ridge_skeletons.shape[0]):
                
            current_image = ridge_skeletons[i].todense()
            current_skel = skeletonize(current_image)
            label_sknw = SG(sknw.build_sknw(current_skel, multi=True, iso=False, ring=True, full=False), njobs=self.njobs, pixel_size=self.pixel_size)

            label_sknw.find_best_edge()
            label_sknw.find_cycles(self.max_nodes, 1)

            if len(label_sknw.cycles)==0:
                label_sknw.find_paths(self.max_nodes, 1)
                if len(label_sknw.paths) == 0:
                    if len(label_sknw.too_short_cycles) > 0:
                        path = label_sknw.too_short_cycles[0]
                    
                else:
                    path = label_sknw.paths[0]
            else:
                path = label_sknw.cycles[0]
            pts = path.get_points()
            all_skeleton_points.append(pts)


        return all_skeleton_points



def create_disk(pixel_size, min_distance, max_distance):
    min_distance = round(min_distance / pixel_size)
    max_distance = round(max_distance / pixel_size)

    shape = (max_distance*2-1,max_distance*2-1)
    try:

        y,x = disk((max_distance-1,max_distance-1),max_distance,shape=shape)
    except:
        y,x = disk(max_distance-1,max_distance-1,max_distance,shape=shape)
    disk_kernel = np.zeros(shape)
    disk_kernel[y,x] = 1
    try:
        y,x = disk((max_distance-1,max_distance-1),min_distance,shape=shape)
    except:
        y,x = disk(max_distance-1,max_distance-1,min_distance,shape=shape)
    disk_kernel[y,x] = 0

    disk_kernel[disk_kernel == 1] = 1/np.sum(disk_kernel[disk_kernel == 1])
    # disk_kernel[disk_kernel == -1] = 1/np.sum(disk_kernel[disk_kernel == -1])
    return disk_kernel


def find_new_contour(neighbourhood):
    neighbourhood = neighbourhood.flatten()
    for corner in [0,2,6,8]:
        if neighbourhood[corner] == 1:
            if all(neighbourhood[i] == 1 for i in lookup_table[corner]):
                neighbourhood[corner] = 0
    return np.reshape(neighbourhood,(3,3))
    


def checkallowed(neighbourhood):
    blobs, nr_of_blobs = label(neighbourhood, np.ones((3,3)))
    return nr_of_blobs == 1



def thin_closed(label_image, image):
    label_image = np.copy(label_image)
    not_allowed_to_delete = set()


    dummy_contour_image = np.zeros_like(label_image)
    calc_new_contour = True
    calc_new_contour_counter = 0
    calc_argsort_counter = 0
    while True:

        if calc_new_contour:
            calc_new_contour_counter += 1
            contours, hierarchy = cv2.findContours(label_image.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours) != 2:
                return None

            x,y = np.concatenate(contours).T
            x = x.flatten()
            y = y.flatten()
            dummy_contour_image = np.zeros_like(label_image)
            dummy_contour_image[y,x] = 1

        values = image[y,x]

        calc_argsort_counter += 1
        argsorted_values = np.argsort(values)

        for i in range(len(argsorted_values)):
            lowest_idx = argsorted_values[i]

            if (y[lowest_idx], x[lowest_idx]) in not_allowed_to_delete:
                continue

            label_image[y[lowest_idx], x[lowest_idx]] = 0

            neighbourhood = label_image[y[lowest_idx]-1:y[lowest_idx] + 2,x[lowest_idx]-1: x[lowest_idx] + 2]
            if not checkallowed(neighbourhood):
                label_image[y[lowest_idx], x[lowest_idx]] = 1
                not_allowed_to_delete.add((y[lowest_idx], x[lowest_idx]))
                continue

            allow_again = set()
            for n in range(y[lowest_idx]-1, y[lowest_idx] + 2):
                for m in range(x[lowest_idx]-1, x[lowest_idx] + 2):
                    allow_again.add((n,m))
            not_allowed_to_delete = not_allowed_to_delete.difference(allow_again)

            neighbourhood = label_image[y[lowest_idx]-1:y[lowest_idx] + 2,x[lowest_idx]-1: x[lowest_idx] + 2]
            contour_neighbourhood = find_new_contour(neighbourhood)
            y_to_add = []
            x_to_add = []
            for o in range(3):
                for p in range(3):
                    if contour_neighbourhood[o,p] == 1:
                        if dummy_contour_image[y[lowest_idx] + o - 1, x[lowest_idx] + p - 1] == 0:
                            y_to_add.append(y[lowest_idx] + o - 1)
                            x_to_add.append(x[lowest_idx] + p - 1)
                            dummy_contour_image[y[lowest_idx] + o - 1,  x[lowest_idx] + p - 1] = 1
            dummy_contour_image[y[lowest_idx], x[lowest_idx]] = 0

            y_to_add = np.array(y_to_add, dtype=np.int64)
            x_to_add = np.array(x_to_add, dtype=np.int64)

            y = np.delete(y, lowest_idx)
            x = np.delete(x, lowest_idx)

            y = np.append(y, y_to_add)
            x = np.append(x, x_to_add)
            calc_new_contour = False
            break
        else:
            break

    return label_image#

def thin_open(label_image, image, skeleton=None, end_points=None):
    assert (skeleton is not None or end_points is not None)
    label_image = np.copy(label_image)
    if skeleton is not None:
        ys,xs = np.nonzero(skeleton)

        end_points = []
        end_idxs = []
        for counter, (y,x) in enumerate(zip(ys,xs)):
            neighbourhood = skeleton[y-1:y+2, x-1:x+2]
            if np.sum(neighbourhood) != 3:
                end_points.append((y,x))
                end_idxs.append(counter)
        if len(end_points) != 2:

            return None
        
        skeleton_coords = np.array([ys,xs]).T
        label_coords = np.array(np.nonzero(label_image)).T
        distances = cdist(skeleton_coords, label_coords)
        argmin_distances = np.argmin(distances, axis=0)

        to_use_points = set()
        
        for idx in end_idxs:
            close_to_end_points_idxs = np.nonzero(argmin_distances== idx)
            
            close_to_end_points_coords = label_coords[close_to_end_points_idxs].T
            values = image[close_to_end_points_coords[0], close_to_end_points_coords[1]]

            max_idx = np.argmax(values)
            to_use_points.add((close_to_end_points_coords[0][max_idx],close_to_end_points_coords[1][max_idx]))
    else:
        to_use_points = set(end_points)

    not_allowed_to_delete = set(to_use_points)

    calc_new_contour = True
    dummy_contour_image = np.zeros_like(label_image)
    while True:


        if calc_new_contour:

            contours, hierarchy = cv2.findContours(label_image.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours) != 1:

                return None

            x,y = np.concatenate(contours).T
            x = x.flatten()
            y = y.flatten()
            dummy_contour_image = np.zeros_like(label_image)
            dummy_contour_image[y,x] = 1

        values = image[y,x]

        argsorted_values = np.argsort(values)
        
        for i in range(len(argsorted_values)):
            lowest_idx = argsorted_values[i]
            if (y[lowest_idx], x[lowest_idx]) in not_allowed_to_delete:
                continue
            label_image[y[lowest_idx], x[lowest_idx]] = 0

            neighbourhood = label_image[y[lowest_idx]-1:y[lowest_idx] + 2,x[lowest_idx]-1: x[lowest_idx] + 2]
            if not checkallowed(neighbourhood):
                label_image[y[lowest_idx], x[lowest_idx]] = 1
                not_allowed_to_delete.add((y[lowest_idx], x[lowest_idx]))
                continue

            allow_again = set()
            for n in range(y[lowest_idx]-1, y[lowest_idx] + 2):
                for m in range(x[lowest_idx]-1, x[lowest_idx] + 2):
                    if (n,m) not in to_use_points:
                        allow_again.add((n,m))
            not_allowed_to_delete = not_allowed_to_delete.difference(allow_again)

            neighbourhood = label_image[y[lowest_idx]-1:y[lowest_idx] + 2,x[lowest_idx]-1: x[lowest_idx] + 2]
            contour_neighbourhood = find_new_contour(neighbourhood)
            y_to_add = []
            x_to_add = []
            for o in range(3):
                for p in range(3):
                    if contour_neighbourhood[o,p] == 1:
                        if dummy_contour_image[y[lowest_idx] + o - 1, x[lowest_idx] + p - 1] == 0:
                            y_to_add.append(y[lowest_idx] + o - 1)
                            x_to_add.append(x[lowest_idx] + p - 1)
                            dummy_contour_image[y[lowest_idx] + o - 1,  x[lowest_idx] + p - 1] = 1
            dummy_contour_image[y[lowest_idx], x[lowest_idx]] = 0
            y_to_add = np.array(y_to_add, dtype=np.int64)
            x_to_add = np.array(x_to_add, dtype=np.int64)
            y = np.delete(y, lowest_idx)
            x = np.delete(x, lowest_idx)
            y = np.append(y, y_to_add)
            x = np.append(x, x_to_add)
            calc_new_contour = False
            break
        else:
            break

    return label_image


from scipy.ndimage import binary_dilation


def find_skeleton(label_image, shape, conv_data_disk, func, kwargs, ps,get_conv_data=False, name=None, data=None):

    if any(g != h for g,h in zip(label_image.shape, shape)):
        label_image = img_as_bool(resize(label_image, shape))


    # TEST
    label_image = skeletonize(label_image)
    
    label_image = binary_dilation(label_image, iterations=2)

    y,x = np.nonzero(label_image)
    
    if len(y) == 0:
        return
    y_max = np.clip(np.max(y) + 50 ,0, label_image.shape[0])
    y_min = np.clip(np.min(y) - 50, 0, label_image.shape[0])
    x_max = np.clip(np.max(x) + 50, 0, label_image.shape[1])
    x_min = np.clip(np.min(x) - 50, 0, label_image.shape[1]) 
    cropped_data = data[y_min:y_max, x_min:x_max]
    

    
    label_vesicle = label_image[y_min:y_max, x_min:x_max]
    conv_vesicle = np.copy(conv_data_disk[y_min:y_max, x_min:x_max])
    


    contours, hierarchy = cv2.findContours(label_vesicle.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    inner_contour = contours[np.argmin([len(i) for i in contours])]
    polygonimage = np.zeros(label_vesicle.shape, dtype=np.int8)
    polygonimage = cv2.fillPoly(polygonimage, [inner_contour], 1 )

    closed = len(contours) != 1
    border_vesicle = False
    if not closed:
        negative_image = np.logical_not(polygonimage.astype(bool))
        negative_blobs,_ = label(negative_image)
        if len(np.unique(negative_blobs)) == 3:
            border_vesicle = True

    conv_vesicle -= np.min(conv_vesicle)
    conv_vesicle /= np.max(conv_vesicle)

    
    ridge_map = func(conv_vesicle, **kwargs)


    ridge_map[label_vesicle == 0] = np.min(ridge_map)

    ridge_map -= np.min(ridge_map)
    ridge_map /= np.max(ridge_map)

        
    ridge_map = np.pad(ridge_map, 5, mode="constant", constant_values=np.min(ridge_map))
    label_vesicle = np.pad(label_vesicle, 5, mode="constant", constant_values=0)



    label_skeleton = skeletonize(label_vesicle)
    

    sg = SG(sknw.build_sknw(label_skeleton, multi=True, iso=False, ring=True, full=False),njobs=1, pixel_size=ps, shape=label_skeleton.shape)

    sg.find_best_edge()
    label_skeleton = sg.get_skeleton()

    if border_vesicle:
        ys,xs = np.nonzero(label_skeleton)

        for y,x in zip(ys,xs):
            if y < 10 or x < 10 or y > label_skeleton.shape[0] - 11 or x > label_skeleton.shape[1] - 11:
                label_skeleton[y,x] = 0
                
        
        blobs, nr_of_blobs = label(label_skeleton, np.ones((3,3)))
        if nr_of_blobs > 1:
            ub, blob_count = np.unique(blobs, return_counts=True)
            ub = ub[1:]
            blob_count = blob_count[1:]
            label_skeleton = (blobs == ub[np.argmax(blob_count)]) * 1
        elif nr_of_blobs == 0:
            label_skeleton = skeletonize(label_vesicle)



    # if not closed:

    #     skeleton = thin_open(label_vesicle, ridge_map, label_skeleton)
    # else:
    #     skeleton = thin_closed(label_vesicle, ridge_map)


    sg.find_cycles(5, 1)
    if len(sg.cycles)==0:
        sg.find_paths(5, 1)
        path = sg.paths[0]
    else:
        path = sg.cycles[0]
    sk = path.get_points()

    cropped_data = np.pad(cropped_data,5,constant_values=np.min(cropped_data))
    skeleton = predictMiddleFrangi(label_vesicle, ridge_map, cropped_data, sk,label_skeleton=label_skeleton, name=name)

    worked = skeleton is not None
    
    if skeleton is None:
        skeleton = label_skeleton
    skeleton = skeleton[5:-5,5:-5]
    ridge_map = ridge_map[5:-5,5:-5]


    np.testing.assert_allclose(skeleton.shape, conv_vesicle.shape)

    if get_conv_data:
        return skeleton, (y_min,y_max, x_min, x_max), worked, (np.sum(skeleton), np.sum(label_skeleton),  np.sum(label_skeleton) / np.sum(skeleton)), ridge_map
    return skeleton, (y_min,y_max, x_min, x_max), worked, (np.sum(skeleton), np.sum(label_skeleton),  np.sum(label_skeleton) / np.sum(skeleton))

from scipy.interpolate import splrep, BSpline

def getInterpolatedSkeleton(coords, distance, is_circle, shape, label_skeleton, name=None ):
    def dist(a,b):
        return cdist([a], [b], "euclidean")[0][0]

    coords = coords
    y,x = coords.T
    starting_distance = distance
    # while True:
    points = np.arange(len(y))

    number_of_points = int(len(y) * distance * 4)

    to_interp_points = np.linspace(0, int(len(y)-1), number_of_points)

    ty,cy,ky = splrep(points, y,per=is_circle)
    y_spline = BSpline(ty, cy, ky,)
    float_interp_y = y_spline(to_interp_points)
    interp_y = np.rint(float_interp_y).astype(np.int32)

    tx,cx,kx = splrep(points, x,per=is_circle)
    x_spline = BSpline(tx, cx, kx,)
    float_interp_x = x_spline(to_interp_points)
    interp_x = np.rint(float_interp_x).astype(np.int32)


    coords = np.array(np.array([interp_y, interp_x]).T)


    float_coords = np.array(np.array([float_interp_y, float_interp_x]).T)

    res = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=-1))
    if np.any(res > 1.5):
        distance += 1

        if distance > 2* starting_distance:
            return None
    
    if is_circle and dist(coords[0], coords[-1]) > 1.5:
        yy,xx = line(coords[-1][0],coords[-1][1],coords[0][0],coords[0][1],)
        new_points = np.array([yy[1:-1], xx[1:-1]])
        coords = np.concatenate([coords, new_points])
        float_coords = np.concatenate([float_coords, new_points])
    # 
    
    coord_round_distance = np.abs(np.sum(np.abs(float_coords - coords),1)) * - 1

    label_image = np.zeros(shape, dtype=np.uint8)
    confidence_image = np.zeros_like(label_image, dtype=np.float32)
    label_image[coords.T[0], coords.T[1]] = 1
    for c,d in zip(coords, coord_round_distance):
        if confidence_image[c[0], c[1]] > d:
            confidence_image[c[0], c[1]] = d
    # confidence_image[coords.T[0], coords.T[1]] = coord_round_distance
    

    lab,num_features = skilabel(np.pad(label_image,1)==0, connectivity=1,return_num=True)
    allowed_features = 1
    if is_circle:
        allowed_features = 2
    lab = lab[1:-1,1:-1] 

    if num_features > allowed_features:
        u, nu = np.unique(lab, return_counts=True)

        argsorted_uniques = np.argsort(nu)
        for b in range(num_features -allowed_features):

            label_image[lab==u[argsorted_uniques[b]]] = 1
            confidence_image[lab==u[argsorted_uniques[b]]] = np.nanmin(confidence_image)



    if is_circle:
        skeleton = thin_closed(label_image, confidence_image)
    else:
        skeleton = thin_open(label_image, confidence_image,end_points=(tuple(coords[0].tolist()), tuple(coords[-1].tolist())))
    if skeleton is None:
        pass

    return skeleton



def predictMiddleFrangi(segmentation, confidence, cropped, skeleton, distance=10, njobs=1, path=None, threshold=0.25, name=None, label_skeleton=None):
    
    y,x = disk((0,0), distance / 2,)

    
    conf = np.copy(confidence)

    is_circle = cdist(skeleton[:1], skeleton[-1:],metric="euclidean")[0][0] < 1.5

    corrected_points = []
    used_points_image = np.zeros_like(segmentation)

    for idx, point in enumerate(skeleton):
        if np.isnan(conf[point[0], point[1]]):
            continue 
        current_y, current_x = y + point[0], x + point[1]
        idxs = np.where((current_x >= 0) & (current_y >= 0) & (current_x < conf.shape[1]) & (current_y < conf.shape[0]))
        current_x = current_x[idxs]
        current_y = current_y[idxs]
        
        surrouding_confidence = conf[current_y, current_x]
        best_conf_value = np.nanmax(surrouding_confidence)
        if best_conf_value < threshold:
            new_point = point
        else:
            best_conf = np.nanargmax(surrouding_confidence)

            new_point = (current_y[best_conf], current_x[best_conf])
        corrected_points.append(new_point)
        conf[current_y, current_x] = np.nan
        conf[new_point[0] + y, new_point[1] + x] = np.nan

    corrected_points = np.array(corrected_points)
    skel_before = skeleton
    skeleton = getInterpolatedSkeleton(corrected_points, distance, is_circle, segmentation.shape,label_skeleton, name=name)
    # resizedskeleton = np.zeros_like(corrected_skeleton)
    # resizedskeleton[rc_y, rc_x] = 1
    if skeleton is None:
        pass


    return skeleton



def ridge_detection(label_volume, image,  njobs=1, pixel_size=1, min_size=8, max_size=38, pool=None,name=None, get_conv_data=False):
    
    data = np.copy(image)
    data -= np.min(data)
    data = data / np.max(data) * 255

    disk_kernel = create_disk(pixel_size, min_size,max_size)

    conv_data_disk = convolve(data, disk_kernel, mode="nearest")
    out_skeleton = np.zeros(label_volume.shape,dtype=np.uint8)
    kwargs = {'sigmas': np.arange(1,2), 'mode': 'reflect'}
    func = frangi
    name = str(name)
    if pool is None:

        if njobs > 1:
            with mp.get_context("spawn").Pool(njobs) as pool:
                if type(label_volume) is np.ndarray:
                    results = [pool.apply_async(find_skeleton, args=[label_volume[i, ...], data.shape, conv_data_disk, func,kwargs, pixel_size,False, f"{name}_{i}",data] ) for i in range(label_volume.shape[0])]
                else:
                    results = [pool.apply_async(find_skeleton, args=[label_volume[i, ...].todense(), data.shape, conv_data_disk, func,kwargs, pixel_size,False,  f"{name}_{i}", data] ) for i in range(label_volume.shape[0])]

                results = [res.get() for res in results]
        else:
            if type(label_volume) is np.ndarray:
                results = [find_skeleton(label_volume[i, ...], data.shape, conv_data_disk, func,kwargs, pixel_size,False, f"{name}_{i}",data)  for i in range(label_volume.shape[0])]
            else:
                results = [find_skeleton(label_volume[i, ...].todense(), data.shape, conv_data_disk, func,kwargs, pixel_size,False,  f"{name}_{i}", data)  for i in range(label_volume.shape[0])]
    else:
        if type(label_volume) is np.ndarray:
            results = [pool.apply_async(find_skeleton, args=[label_volume[i, ...], data.shape, conv_data_disk, func,kwargs, pixel_size,False, f"{name}_{i}",data],error_callback=print_error ) for i in range(label_volume.shape[0])]
        else:
            results = [pool.apply_async(find_skeleton, args=[label_volume[i, ...].todense(), data.shape, conv_data_disk, func,kwargs, pixel_size,False,  f"{name}_{i}", data],error_callback=print_error ) for i in range(label_volume.shape[0])]
        results = [res.get() for res in results]


    worked_list = []
    for i, result in enumerate(results):
        
        if result is None:
            worked_list.append(False)
            continue
        skeleton, coords, worked, ratio_etc = result
        worked_list.append(worked)
        
        out_skeleton[i, coords[0]:coords[1],coords[2]:coords[3]][skeleton == 1] = 1


        

    
    out_skeleton = sparse.as_coo(out_skeleton)

    return out_skeleton

def print_error(error):
    print(error, flush=True)