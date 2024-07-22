
def resizeSegmentation(image, shape):
    import sparse
    from skimage.transform import resize
    if hasattr(image, "todense"):
        return sparse.as_coo(resize(image.todense(), shape, 0))
    else:
        return sparse.as_coo(resize(image, shape, 0))


def resizeSegmentationFromCoords(coords,seg_shape, shape, max_width, pixel_size):
    import sparse
    from skimage.transform import resize
    import numpy as np
    from skimage.draw import disk


    segmentation = np.zeros(seg_shape)
    y,x = coords.T
    disk_y, disk_x = disk((0, 0),radius=(max_width/pixel_size)/2, )
    
    for y_,x_ in zip(y, x):
        current_disk_y = disk_y + y_
        current_disk_x = disk_x + x_

        idxs_to_use = np.where((current_disk_y >= 0) & (current_disk_y < seg_shape[0]) & (current_disk_x >= 0) & (current_disk_x < seg_shape[1]))
        current_disk_y = current_disk_y[idxs_to_use]
        current_disk_x = current_disk_x[idxs_to_use]
        segmentation[current_disk_y, current_disk_x] = 1
    



    return sparse.as_coo(resize(segmentation, shape, 0))




def resizeMicrograph(image, shape):
    from skimage.transform import resize
    return resize(image, shape, 0)



def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    import numpy as np
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h




def get_thickness_from_profile_orig(profile, middle_idx, min_thickness, max_thickness, idx=None):
    from scipy import signal
    import numpy as np


        

    neg_peaks, _ = signal.find_peaks(profile, distance=min_thickness/2, )


    if any([np.abs(middle_idx - peak) < min_thickness/2 for peak in neg_peaks]):
        peaks, properties = signal.find_peaks(profile * -1,prominence=0)
        # distance = np.abs(np.array(neg_peaks) - middle_idx)
        # best_middle_from_dist = neg_peaks[np.argmin(distance)]
        neg_peaks = [peak for peak in neg_peaks if np.abs(middle_idx - peak) < min_thickness/2]
        best_middle = neg_peaks[np.argmax([profile[peak] for peak in neg_peaks])]
        # if best_middle != best_middle_from_dist:
        # left_range = ( best_middle- max_thickness / 2, best_middle - min_thickness/2)
        left_range = ( best_middle- max_thickness / 2, best_middle - 1)
        # right_range =  (best_middle + min_thickness/2,  best_middle + max_thickness/2)
        right_range =  (best_middle + 1,  best_middle + max_thickness/2)

        left_peaks = [peak for peak in peaks if peak >= left_range[0] and peak <= left_range[1]]
        right_peaks = [peak for peak in peaks if peak >= right_range[0] and peak <= right_range[1] ]

        left_properties = [property for peak, property in zip(peaks, properties["prominences"] ) if peak >= left_range[0] and peak <= left_range[1]]
        right_properties = [property for peak, property in zip(peaks, properties["prominences"] ) if peak >= right_range[0] and peak <= right_range[1]]
        
        if len(left_peaks) > 0 and len(right_peaks) > 0:
            best_left = left_peaks[np.argmin([profile[i] for i in left_peaks])]
            best_right = right_peaks[np.argmin([profile[i] for i in right_peaks])]
            
            left_distance = abs(best_left - best_middle)
            right_distance = abs(best_right - best_middle)

            closest_other_peak_left = [abs(peak - best_left) for peak in left_peaks if peak != best_left and abs(peak-best_left) < left_distance]
            closest_other_peak_right = [abs(peak - best_right) for peak in right_peaks if peak != best_right and abs(peak-best_right) < right_distance]
            if len(closest_other_peak_left) > 0:
                closest_other_peak_left_value = profile[np.min(closest_other_peak_left)]
                ratio = abs(closest_other_peak_left_value - profile[best_left]) / abs(profile[best_middle] - profile[best_left])
                closest_other_peak_left = np.min(closest_other_peak_left) / left_distance 
                confidence_left = min(1,max(ratio, closest_other_peak_left))
            else:
                closest_other_peak_left = 1
            if len(closest_other_peak_right) > 0:
                closest_other_peak_right_value = profile[np.min(closest_other_peak_right)]
                ratio = abs(closest_other_peak_right_value - profile[best_right]) / abs(profile[best_middle] - profile[best_right])
                closest_other_peak_right = np.min(closest_other_peak_right) / right_distance 
                confidence_right = min(1,max(ratio, closest_other_peak_right))
            else:
                closest_other_peak_right = 1
            left_property = left_properties[np.argmin([profile[i] for i in left_peaks])]
            right_property = right_properties[np.argmin([profile[i] for i in right_peaks])]

            if best_right - best_left < min_thickness or best_right -best_left > max_thickness:

                return None
            return best_right - best_left, best_left, best_right, closest_other_peak_left, closest_other_peak_right
        else:

            return None

    return None


def get_thickness_from_profile(orig_profile, middle_idx, min_thickness, max_thickness, precision, idx=None):
    from scipy import signal
    import numpy as np
    max_thickness = max_thickness / precision
    min_thickness = min_thickness / precision
    # for sigma in range(int(3 / precision),int(10 / precision), int(1/precision)):
    sigma = int(4/precision)
    window_size = int(10/precision)
    # for window_size in range(1,len(orig_profile)//4):
    window = signal.windows.gaussian(window_size, sigma)
    padded_profile = np.concatenate([np.ones((window_size - 1) // 2)* orig_profile[0], orig_profile, np.ones((window_size - 1) // 2)* orig_profile[-1]])
    profile = signal.convolve(padded_profile, window/window.sum(), "valid")
    
    neg_peaks, _ = signal.find_peaks(profile, distance=min_thickness/2, )


    if any([np.abs(middle_idx - peak) < min_thickness/2 for peak in neg_peaks]):
        peaks,_  = signal.find_peaks(profile * -1)
        
        neg_peaks = [peak for peak in neg_peaks if np.abs(middle_idx - peak) < min_thickness/2]
        best_middle = neg_peaks[np.argmax([profile[peak] for peak in neg_peaks])]
        
        left_range = ( best_middle- max_thickness / 2, best_middle - 1)
        right_range =  (best_middle + 1,  best_middle + max_thickness/2)

        left_peaks = [peak for peak in peaks if peak >= left_range[0] and peak <= left_range[1]]
        right_peaks = [peak for peak in peaks if peak >= right_range[0] and peak <= right_range[1] ]
        if len(left_peaks) != 1 or len(right_peaks) != 1 :
            return None
        elif (right_peaks[0] - left_peaks[0]) < min_thickness:
            return None
        else:
            return (right_peaks[0] - left_peaks[0]) * precision, left_peaks[0], right_peaks[0], window_size, sigma, profile

                

    return None


def create_distance_map(membrane, ratio, membrane_image, micrograph_pixel_size, interp):
    import numpy as np

    from scipy.spatial.distance import cdist
    from skimage.morphology import skeletonize
    from cv2 import drawContours
    from scipy.ndimage import distance_transform_edt,label
    from matplotlib import pyplot as plt
 


    membrane_image = membrane_image.todense()

    resized_membrane_coordinate_y, resized_membrane_coordinate_x  = np.nonzero(membrane_image)

    y_min = max(0, np.min(resized_membrane_coordinate_y) - 20)
    x_min = max(0, np.min(resized_membrane_coordinate_x) - 20)

    y_max = min(membrane_image.shape[0], np.max(resized_membrane_coordinate_y) + 20)
    x_max = min(membrane_image.shape[1], np.max(resized_membrane_coordinate_x) + 20)
    # y_max = np.max(interp_y) + 20
    
    # x_max = np.max(interp_x) + 20
    interp_y, interp_x = interp
    interp_y -= y_min
    interp_x -= x_min
    
    # distance_maps_min_max[l] = {"y_min":y_min, "x_min":x_min}



    cropped_segmentation = membrane_image[y_min:y_max, x_min:x_max]

    cropped_skeleton = np.zeros_like(cropped_segmentation)

    cropped_skeleton[interp_y.astype(np.int32), interp_x.astype(np.int32)] = 1

    cropped_skeleton = skeletonize(cropped_skeleton)
    
    coords = membrane.coords
    resized_normal_skel = (coords * ratio) - np.array([y_min,x_min])
    nan_map =np.logical_not(cropped_segmentation)



    labels = None
    if not membrane.is_circle:
        ends = []
        padded_skel = np.pad(cropped_skeleton,1,"constant", constant_values=0)
        skel_y, skel_x = np.nonzero(padded_skel)
        for sy, sx in zip(skel_y, skel_x):
            neighbours = np.sum(padded_skel[sy-1:sy+2, sx-1:sx+2])
            if neighbours == 2:
                ends.append((sy-1, sx-1))
        ends = np.array(ends)
        idx = 5
        if len(ends) >= 2:
            orig_end_points = np.array([resized_normal_skel[0], resized_normal_skel[-1]])
            distances_end_points = cdist(orig_end_points, ends)
            min_distances_end_points = np.min(distances_end_points, 1)
            argmin_distances_end_points = np.argmin(distances_end_points, 1)

            if argmin_distances_end_points[0] != argmin_distances_end_points[1]:
                closest_point_to_start = ends[argmin_distances_end_points[0]]
                closest_point_to_end = ends[argmin_distances_end_points[1]]

                vector = closest_point_to_end - resized_normal_skel[max(-idx, -len(resized_normal_skel))]
                vector = vector/np.linalg.norm(vector)
                last_point = closest_point_to_end.copy()
            
                while True:
                    last_point = last_point + vector
                    if int(last_point[0]) < 0 or int(last_point[0]) >= cropped_segmentation.shape[0] or int(last_point[1]) < 0 or int(last_point[1]) >= cropped_segmentation.shape[1]:
                        break
                    if cropped_segmentation[int(last_point[0]), int(last_point[1])] == 0:
                        break
                    
                    cropped_skeleton[int(last_point[0]), int(last_point[1])] = 1

                    if np.any(cropped_segmentation[max(0, int(last_point[0]-1)):int(last_point[0]) + 2,max(0, int(last_point[1]-1)):int(last_point[1]) + 2 ] == 0):
                        break 

                vector = closest_point_to_start - resized_normal_skel[min(idx, len(resized_normal_skel)-1)]
                vector = vector/np.linalg.norm(vector)
                last_point = closest_point_to_start.copy()

                while True:
                    last_point = last_point + vector
                    if int(last_point[0]) < 0 or int(last_point[0]) >= cropped_segmentation.shape[0] or int(last_point[1]) < 0 or int(last_point[1]) >= cropped_segmentation.shape[1]:
                        break
                    if cropped_segmentation[int(last_point[0]), int(last_point[1])] == 0:
                        break
                    cropped_skeleton[int(last_point[0]), int(last_point[1])] = 1

                    if np.any(cropped_segmentation[max(0, int(last_point[0])-1):int(last_point[0]) + 2,max(0, int(last_point[1])-1):int(last_point[1]) + 2 ] == 0):
                        break 
                cropped_segmentation[cropped_skeleton == 1] = 0
                
                labels, num_features = label(cropped_segmentation, )
        
    
    
    cropped_skeleton = cropped_skeleton == 0

    current_distance_map, ind = distance_transform_edt(cropped_skeleton, return_indices=True) 
    current_distance_map = current_distance_map * micrograph_pixel_size 
    current_distance_map[nan_map] = np.nan


    seg_y = resized_membrane_coordinate_y - y_min
    seg_x = resized_membrane_coordinate_x - x_min
    
    # seg_y, seg_x = np.nonzero(np.logical_not(np.isnan(distance_maps[0])))
    cropped_skeleton = np.logical_not(cropped_skeleton)
    interp_skel_points = np.array(np.nonzero(cropped_skeleton)).T

    


    distances = cdist(resized_normal_skel, interp_skel_points)

    distances = np.argmin(distances, 0)

    new_indice_map = np.ones_like(cropped_segmentation, dtype=np.int32) * - 1
    interp_skel_points_y, interp_skel_points_x = interp_skel_points.T
    new_indice_map[interp_skel_points_y, interp_skel_points_x] = distances
    new_indice_map[seg_y, seg_x] = new_indice_map[ind[0][seg_y, seg_x], ind[1][seg_y, seg_x]]





    # cropped_image = np.copy(smoothed_image[np.min(resized_membrane_coordinate_y) - np.min(membrane_coords_y):np.max(resized_membrane_coordinate_y) + 1, 
    #                                        np.min(resized_membrane_coordinate_x) - np.min(membrane_coords_x):np.max(resized_membrane_coordinate_x) + 1])

    
    to_estimate=True
    if membrane.is_circle:
        dummy_image = np.zeros_like(current_distance_map, dtype=np.int8)
        coords = np.array((interp_x, interp_y)).T

        drawContours(dummy_image, [coords], -1, 1, -1)
        # dummy_images[l] = dummy_image                
        current_distance_map[dummy_image > 0] *= -1

    else:
        if num_features == 2:
            current_distance_map[labels == 1] *= -1
        else:
            u, c = np.unique(labels, return_counts=True)
            best_label = u[1:][np.argmax(c[1:])]
            current_distance_map[labels == best_label] *= -1
      


    return current_distance_map, new_indice_map, (y_min,y_max, x_min,x_max), to_estimate, cropped_skeleton



def getThicknessEstimation(distance_idxs, bins, original_values, min_thickness, max_thickness, is_membrane=False, save_str=None):
    import numpy as np
    from scipy.interpolate import splrep, BSpline
    from matplotlib import pyplot as plt

    new_bins = []
    profile = []
    distances = []
    for bin_value in np.unique(distance_idxs):
        new_bins.append(bins[bin_value - 1])
        idxs = np.argwhere(distance_idxs == bin_value)
        values = original_values[idxs]
        distances.append(bins[bin_value - 1])
        profile.append(np.mean(values))
    

    precision = 0.1
    
    nr_of_spline_points = int((np.max(distances) - np.min(distances)) * 1/precision)
    precision = (np.max(distances) - np.min(distances)) / float(nr_of_spline_points)
    
    # t,c,k = splrep(np.linspace(0, height, len(profile)), profile)
    t,c,k = splrep(distances, profile)
    profile = BSpline(t,c,k)(np.linspace(int(np.min(distances)), int(np.max(distances)), nr_of_spline_points))
    zero_idx = np.where(np.array(distances) == 0)[0]

    zero_idx = nr_of_spline_points / len(distances) * zero_idx


    result = get_thickness_from_profile(profile, zero_idx, min_thickness, max_thickness, precision)

    return result, zero_idx, profile





def identifyIce(membrane):
    y,x = membrane.coords.T
    pts = np.array(np.array([x,y]).T)

    vesicle = np.zeros(self.segmentation_shape, dtype=np.uint8)

    cv2.drawContours(vesicle, [pts], -1, 1, -1)
    resized_micrograph = self.getResizedMicrograph()
    resized_micrograph -= np.mean(resized_micrograph)
    resized_micrograph /= np.std(resized_micrograph)
    
    inner_mean = np.mean(resized_micrograph[vesicle != 0])
    return inner_mean



def leastsq_circle(pts):
    import numpy as np
    from scipy import optimize

    def calc_R(x,y, xc, yc):
        """ calculate the distance of each 2D points from the center (xc, yc) """
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f(c, x, y):
        """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(x, y, *c)
        return Ri - Ri.mean()

    def lestsq(pts):
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
    return lestsq(pts)




def estimate_thickness(membrane, distance_map, cropped_image, index, max_neighbour_dist, min_thickness, max_thickness, micrograph=""):
    import numpy as np
    if not membrane.to_estimate:
        return {}, [] 
    distance_values = distance_map.flatten()

    bins = np.linspace(np.floor(np.nanmin(distance_values)).astype(np.int64), np.ceil(np.nanmax(distance_values)).astype(np.int64), (np.ceil(np.nanmax(distance_values)).astype(np.int64) - np.floor(np.nanmin(distance_values)).astype(np.int64)+1))
    distance_idxs = np.digitize(distance_values[np.logical_not(np.isnan(distance_values))], bins, )

    # original_values = smoothed_image.flatten()[np.logical_not(np.isnan(distance_values))]

    original_values = cropped_image.flatten()[np.logical_not(np.isnan(distance_values))] 
    

    result, zero_idx, profile = getThicknessEstimation(distance_idxs, bins, original_values, min_thickness, max_thickness, True, save_str=f"{micrograph}_{membrane.membrane_idx}")

    membrane_attributes = {"smoothed_thickness_profile":{}}
    membrane_attributes["thickness_profile"] = profile

    membrane_attributes["smoothed_thickness_profile"]["middle_idx"] = zero_idx 
    ps_attributes = []



    if result is not None:
        thickness, fp, sp, window_size, sigma, smoothed_profile = result
        membrane_attributes["thickness"] = thickness
        membrane_attributes["smoothed_thickness_profile"]["fp"] = fp
        membrane_attributes["smoothed_thickness_profile"]["sp"] = sp
        membrane_attributes["smoothed_thickness_profile"]["window_size"] = window_size
        membrane_attributes["smoothed_thickness_profile"]["sigma"] = sigma
        membrane_attributes["smoothed_thickness_profile"]["profile"] = smoothed_profile


        distance_value_assignment = {}
        image_value_assignment = {}

        ys,xs = np.nonzero(index > -1)

        for y,x in zip(ys, xs):

            if index[y,x] not in distance_value_assignment:
                distance_value_assignment[index[y,x]] = []
                image_value_assignment[index[y,x]] = []
            distance_value_assignment[index[y,x]].append(distance_map[y,x])
            image_value_assignment[index[y,x]].append(cropped_image[y,x])



        

        for p in membrane.points():

            neighbour_idxs = p.get_idxs_of_neighbourhood(max_dist=max_neighbour_dist)

    

            distance_values = []
            original_values = []
            
            for i in neighbour_idxs:
                if i not in distance_value_assignment:
                    continue
                distance_values.extend(distance_value_assignment[i])
                original_values.extend(image_value_assignment[i])
            distance_values = np.array(distance_values)
            original_values = np.array(original_values)
            
            if len(distance_values) == 0:
                p_attribute = {"thickness_profile":{}}


                p_attribute["thickness_profile"]["middle_idx"] = None 
                
                p_attribute["thickness_profile"]["unsmoothed"] = None
                ps_attributes.append(p_attribute)
                continue
            bins = np.linspace(np.floor(np.nanmin(distance_values)).astype(np.int64), np.ceil(np.nanmax(distance_values)).astype(np.int64), (np.ceil(np.nanmax(distance_values)).astype(np.int64) - np.floor(np.nanmin(distance_values)).astype(np.int64)+1))
            distance_idxs = np.digitize(distance_values[np.logical_not(np.isnan(distance_values))], bins, )

            # FROM HERE
            result, zero_idx, profile = getThicknessEstimation(distance_idxs, bins, original_values, min_thickness, max_thickness, save_str=f"{micrograph}_{membrane.membrane_idx}_{p.idx}")

            # path_instance.thickness_profile_whole = np.linspace(0, height, nr_of_spline_points)
            # p.thickness_profile = profile
            
            p_attribute = {"thickness_profile":{}}


            p_attribute["thickness_profile"]["middle_idx"] = zero_idx 
            
            p_attribute["thickness_profile"]["unsmoothed"] = profile
            
            
        
            if result is not None:
                thickness, fp, sp, window_size, sigma, smoothed_profile = result
                p_attribute["thickness"] = thickness
                p_attribute["thickness_profile"]["fp"] = fp
                p_attribute["thickness_profile"]["sp"] = sp
                # p_attribute["thickness_profile"]["window_size"] = window_size
                # p_attribute["thickness_profile"]["sigma"] = sigma
                p_attribute["thickness_profile"]["smoothed"] = smoothed_profile
            ps_attributes.append(p_attribute)
    return membrane_attributes, ps_attributes