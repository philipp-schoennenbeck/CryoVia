
from matplotlib import image
import numpy as np
import mrcfile
from skimage.draw import disk
from scipy.signal import convolve
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt


def mask_carbon_edge_per_file(path, gridsizes, cut_off, ps, circles=None, idx=None, get_hist_data=False, to_resize=False, resize=7):
    if isinstance(path, (str, Path)):
        path = Path(path)
        if path.suffix in [".mrc", ".MRC", ".rec", ".REC"]:
            image = mrcfile.open(path, permissive=True).data*1
        else:
            image = np.array(Image.open(path).convert("L"))
    else:
        image = path
    data = image - np.mean(image)
    data /= np.std(data)
    
    if to_resize:
        original_ps = ps
        original_shape = data.shape
        ratio = ps / resize
        new_shape = None
        if ratio < 1:
            new_shape = [int(os * ratio) for os in original_shape]
            data = np.array(Image.fromarray(data).resize(new_shape[::-1]))
            ps = resize

    differences = []
    coords = []
    # convolved_images = []
    # convolved_images_neg = []
    if get_hist_data:
        hist_data = {}
    for gridsize in gridsizes:


        
        if circles is None:
            radius = int(gridsize/ps // 2)
            circle_coords = disk((radius,radius), radius)
            
            circle = np.zeros((radius*2, radius*2))
            circle[circle_coords[0], circle_coords[1]] = 1
            circle = circle/np.sum(circle)
            neg_circle = (circle==0)*1
            neg_circle = neg_circle / np.sum(neg_circle)
            combined_circle = circle - neg_circle
            diff_image = convolve(data, combined_circle, "full", method="fft")
            # convolved_image_neg = convolve(data, neg_circle, "full", method="fft")
        else:
            convolved_image = convolve(data, circles[gridsize][0], "full", method="fft")
            convolved_image_neg = convolve(data, circles[gridsize][1], "full", method="fft")
            diff_image = convolved_image - convolved_image_neg
        
        coord = np.argmax(diff_image)
        coord = np.unravel_index(coord, diff_image.shape)
        if circles is None:
            coord = coord[0] - circle.shape[0]//2, coord[1] - circle.shape[1]//2
        else:
            coord = coord[0] - circles[gridsize][0].shape[0]//2, coord[1] - circles[gridsize][0].shape[1]//2

        differences.append(np.max(diff_image))
        if to_resize and ratio < 1:
            coord = (np.array(coord) / ratio).astype(np.int32)
        coords.append(coord)
        # convolved_images.append(convolved_image)
        # convolved_images_neg.append(convolved_image_neg)

        if get_hist_data:
            values, edges = np.histogram(diff_image, bins=50)
            
            hist_data[gridsize] = {"values":values, "edges":edges, "threshold":cut_off, "center":coord}

    argmax = np.argmax(differences)
    coord = coords[argmax]
    if to_resize and ratio < 1:
        ps = original_ps

    mask = np.zeros_like(image, dtype=np.uint8)
    if np.max(differences ) > cut_off:
        yy,xx = disk(coord, gridsizes[np.argmax(differences)]/ps // 2, shape=image.shape,)
        mask[yy,xx] = 1
    else:
        mask = np.ones_like(image, dtype=np.uint8)

    if get_hist_data:
        return mask, hist_data, gridsizes[np.argmax(differences)]
    return mask

def mask_carbon_edge(paths_or_images, gridsizes=[11000,11500,12000,12500, 13000,19000,19500,20000,20500,21000], cut_off=0.02, ps=7, njobs=1):
    import multiprocessing as mp
    circles = {}
    for gridsize in gridsizes:
        radius = int(gridsize/ps // 2)
        circle_coords = disk((radius,radius), radius)
        circle = np.zeros((radius*2, radius*2))
        circle[circle_coords[0], circle_coords[1]] = 1
        circle = circle/np.sum(circle)
        neg_circle = (circle==0)*1
        neg_circle = neg_circle / np.sum(neg_circle)
        circles[gridsize] = [circle, neg_circle]

    circle_coords = {gridsize:disk((int(gridsize/ps // 2), int(gridsize/ps // 2)), int(gridsize/ps // 2)) for gridsize in gridsizes}
    # 

    if njobs > 1:
        with mp.Pool(njobs) as pool:
            results = [pool.apply_async(mask_carbon_edge_per_file, [path, gridsizes, cut_off, ps, circles, idx]) for idx, path in enumerate(paths_or_images)]
            masks = [res.get() for res in results]
    else:

        masks = [mask_carbon_edge_per_file(path, gridsizes, cut_off, ps, circles) for path in paths_or_images]
    
        
    return masks
    #
 
def gauss(fx,fy,sig):

    r = np.fft.fftshift(np.sqrt(fx**2 + fy**2))
    
    return np.exp(-2*np.pi**2*(r*sig)**2)

def gaussian_filter(im,sig,apix):
    '''
        sig (real space) and apix in angstrom
    '''
    sig = sig/2/np.pi
    fx,fy = np.meshgrid(np.fft.fftfreq(im.shape[1],apix),\
                        np.fft.fftfreq(im.shape[0],apix))

    im_fft = np.fft.fftshift(np.fft.fft2(im))
    fil = gauss(fx,fy,sig*apix)
    
    im_fft_filtered = im_fft*fil
    newim = np.real(np.fft.ifft2(np.fft.ifftshift(im_fft_filtered)))
    
    return newim

def find_grid_hole(to_size=100, radii=[12000], threshold=0.005, files=[], coverage_percentage=0.5, outside_percentage=1, outside_coverage_percentage=0.05, detect_ring=True, ring_width=1000, use_std=False):
    results = {}
    return_results = {}
    for r in radii:
        if not detect_ring:
            y,x = disk((0,0), r/to_size / 2)
        else:
            ring_r = int((r + (ring_width // 2))/to_size / 2)

            outer_y, outer_x = disk((ring_r*2,ring_r*2), (r + (ring_width // 2))/to_size / 2)
            inner_y, inner_x = disk((ring_r*2,ring_r*2), (r - (ring_width // 2))/to_size / 2)
            ring_img = np.zeros((ring_r * 4, ring_r*4))
            ring_img[outer_y, outer_x] = 1
            ring_img[inner_y, inner_x] = 0
            y,x = np.nonzero(ring_img)
            y = np.array(y) - ring_r * 2
            x = np.array(x) - ring_r * 2
            circle_y,circle_x = disk((0,0), r/to_size / 2)
            # plt.imshow(ring_img)
            # plt.show()
            # print(len(y), len(x))
            # break

        radius = r/to_size // 2
        
        for counter, current_file in enumerate(files):
            
            current_file = Path(current_file)
            if current_file.suffix == ".mrc":            
                
                data = mrcfile.open(current_file)
                ps = data.voxel_size["x"]
                data = data.data*1
                middle = np.median(data)
                std = np.std(data)
                left = middle - std * 4
                right = middle + std * 4
                data = np.clip(data, left, right)
                data = data -np.min(data)
                data = data / np.max(data)
                ratio = ps / to_size
                new_shape = [round(s * ratio) for s in data.shape]
                new_data = np.array(Image.fromarray(data).resize(new_shape[::-1]))

                sig = int(1500 / to_size)
                sig += (sig + 1) % 2
                filtered = gaussian_filter(new_data,0,to_size) - gaussian_filter(new_data,sig,to_size)

                cut_off = 5
                new_data = filtered[cut_off:-cut_off,cut_off:-cut_off]
                low = int(- radius *outside_percentage)
                high_0 = int(new_data.shape[0] + radius * outside_percentage)
                high_1 = int(new_data.shape[1] + radius * outside_percentage)
                output = np.zeros((high_0 - low, high_1 - low ))
                pixels = filtered.size
                min_length = pixels * coverage_percentage
                outside_min_length = pixels * outside_coverage_percentage
                for i in range(low, high_0):
                    for j in range(low, high_1):
                        current_y = y + i
                        current_x = x + j

                        usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                        
                        # print(new_data.shape)
                        current_y = current_y[usable_idxs]
                        current_x = current_x[usable_idxs]
                        

                        idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                        outside_values = np.delete(new_data.flatten(), idxs)
                        inside_values = new_data[current_y, current_x]

                        if detect_ring:
                            current_y = circle_y + i
                            current_x = circle_x + j

                            usable_idxs = np.where((current_y >= 0) & (current_y < new_data.shape[0]) & (current_x>=0) & (current_x<new_data.shape[1]))
                            
                            # print(new_data.shape)
                            current_y = current_y[usable_idxs]
                            current_x = current_x[usable_idxs]
                            

                            idxs = np.ravel_multi_index((current_y, current_x), new_data.shape)
                            length_out = len(np.delete(new_data.flatten(), idxs))
                            length_inside = len(current_y)
                        else:
                            length_out = len(outside_values)
                            length_inside = len(inside_values)

                        if length_out < outside_min_length:
                            output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0 
                        elif length_inside < min_length:
                            output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = 0
                        else:
                            if not use_std:
                                outside = np.nanmean(outside_values)
                                inside = np.nanmean(inside_values)
                            else:
                                outside = np.nanstd(outside_values)
                                inside = np.nanstd(inside_values)
                            if detect_ring:
                                output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = outside - inside

                            else:
                                output[i + int(radius * outside_percentage),j + int(radius * outside_percentage)] = inside - outside

                
                
                circle_center = np.unravel_index(np.argmax(output), output.shape)
                


                mask = np.zeros_like(output)
                current_y = y + circle_center[0]
                current_x = x + circle_center[1]
                usable_idxs = np.where((current_y >= 0) & (current_y < output.shape[0]) & (current_x>=0) & (current_x<output.shape[1]))
                        
                        
                current_y = current_y[usable_idxs]
                current_x = current_x[usable_idxs]
                mask[current_y, current_x] = 1
                


                orig_center = ( np.array(circle_center) + cut_off + low ) / (ps/to_size)
                orig_y, orig_x = disk(orig_center, r/ps / 2, shape=data.shape)
                orig_mask = np.zeros_like(data, dtype=np.uint8)
                orig_mask[orig_y, orig_x] = 1

                result_mask = np.zeros_like(data, dtype=np.uint8)
                if np.max(output) > threshold:
                    result_mask[orig_y, orig_x] = 1
                if current_file not in results:
                    results[current_file] = {}
                results[current_file][r] = (orig_mask,result_mask, output, new_data, filtered, mask)
                if current_file not in return_results:
                    return_results[current_file] = {"mask":result_mask, "max":np.max(output)}
                else:
                    if np.max(output) > return_results[current_file]["max"]:
                        return_results[current_file] = {"mask":result_mask, "max":np.max(output)}
    return results, return_results



if __name__ == "__main__":
    pass