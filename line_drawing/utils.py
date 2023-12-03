import os
import cv2
import pickle
import numpy as np

from scipy.stats import mode
from collections import deque
from PIL import Image

def canny_detection(image, threshold1, threshold2, **kwargs):
    return cv2.Canny(image, 
                     threshold1=threshold1,
                     threshold2=threshold2,
                     **kwargs)
    
def gaussian_blur(image, kernel_size:tuple, sigmaX, sigmaY):
    return cv2.GaussianBlur(image, kernel_size, sigmaX, sigmaY)

def median_blur(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def bilateral_blur(image, d, sC, sS):
    return cv2.bilateralFilter(image, d, sC, sS)

def make_edge_thicker(edge, kernel_size=2):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.dilate(edge, kernel)
    
def color_merge(image, k):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = k
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image

def fill_connected_components(arr, target, diagonal=False):
    # Initialize the second array with zeros
    rows, cols = arr.shape
    result = np.zeros((rows, cols), dtype=int)
    if diagonal:
        near = [(-1, -1), (-1, 0), (-1, 1), (1, 0), (1, -1), (0, -1), (0, 1), (1, 1)]
    else:
        near = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    # Function to get neighbors of a cell
    def get_neighbors(r, c):
        for dr, dc in near:  # Up, Down, Left, Right
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    # BFS function
    def bfs(start, value):
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for nr, nc in get_neighbors(r, c):
                if arr[nr, nc] == target and result[nr, nc] == 0:
                    result[nr, nc] = value
                    queue.append((nr, nc))

    # Main loop
    current_value = 1
    for r in range(rows):
        for c in range(cols):
            if arr[r, c] == target and result[r, c] == 0:
                result[r, c] = current_value
                bfs((r, c), current_value)
                current_value += 1

    return result

def remove_noise(edge, threshold=100, diagonal=True):
    output_array = fill_connected_components(1-edge//255, diagonal=diagonal)

    mx = [0]
    new_ary = output_array.copy()
    for i in range(np.unique(output_array)[-1]):
        if i == 0:
            continue
        else:
            if (output_array==i).sum() < threshold:
                new_ary[output_array==i] -= i
                
    return new_ary

def fill_large_chunks(SAM_colormaps, cartoon_images):
    output = []
    
    rgb_to_gray = lambda x : 0.299*x[:, :, 0] + 0.587*x[:, :, 1] + 0.114* x[:, :, 2]
    for cmap, cimg in zip(SAM_colormaps, cartoon_images):
        new = cmap.copy()
        tmp = fill_connected_components(gaussian_blur(rgb_to_gray(cmap), (5, 5), 3, 3), 0)
        
        for i in np.unique(tmp):
            if (tmp==i).sum() < 100 or i == 0:
                continue
            idx1, idx2 = np.where(tmp == i)
            c = np.median(cimg[idx1, idx2, :], axis=0)/255
            new[idx1, idx2, :] += c
        output.append(new)
    
    return output            


def load_mask(data_dir) -> list:
    mask_dirs = os.listdir(data_dir)
    mask_dirs.sort()
    masks = []
    mask_filtered = []
    
    for dd in mask_dirs:
        with open(os.path.join(data_dir, dd), "rb") as f:
            mask = pickle.load(f)    
            masks.append(mask)
            
        overlap = np.zeros(mask[0]['segmentation'].shape)
        tmp = []
        for m in mask:
            A = (m['segmentation'] != 0).sum()
            B = ((overlap != 0) * (m['segmentation'] != 0)).sum()
            # if B /  A  > 0.7:
            #     print(1)
            #     continue
            
            overlap[m['segmentation']!=0] += 1
            tmp.append(m)
            
        mask_filtered.append(tmp)
            
    return masks

def get_SAM_edges(data_dir, edges, score_threshold=0.9, area_threshold=1000, threshold1=50, threshold2=100, thickness=2):
    output = []
    masks_lst = load_mask(data_dir)
    
    for i, masks in enumerate(masks_lst):
        t = np.zeros(masks[0]['segmentation'].shape).astype(np.uint8)
        for edge_v, m in enumerate(masks):
            if m['stability_score'] > score_threshold:
                if m['segmentation'].sum() < area_threshold:
                    continue
                
                new = np.zeros(m['segmentation'].shape).astype(np.uint8)
                
                new[m['segmentation']] += 100
                new = make_edge_thicker(canny_detection(new, threshold1, threshold2), kernel_size=2)
                
                t[new!=0] += (edge_v + 1)
        output.append(t)
        
    return output

def get_SAM_colormap(data_dir, cartoon_images, score_threshold=0.8, area_threshold=50):
    output = []
    masks_lst = load_mask(data_dir)
    
    for img_id, masks in enumerate(masks_lst):
        sam_color_mask = np.zeros(list(masks[0]['segmentation'].shape)+[3])
        
        masks.sort(key=lambda x : x['segmentation'].sum(), reverse=True)
        
        for i, m in enumerate(masks):
            if m['stability_score'] > score_threshold:
                if m['segmentation'].sum() < area_threshold:
                    continue
                
            indexing_expanded = m['segmentation'][:, :, np.newaxis]

            # Apply the indexing
            target_index = np.where(indexing_expanded, cartoon_images[img_id], 0)
            idx1, idx2 = np.where(m['segmentation'])
            mean_color = np.median(cartoon_images[img_id][idx1, idx2], axis=0)
            # print((m['segmentation'][:, :, np.newaxis]) * mean_color[np.newaxis, np.newaxis, :])
            # mean_color = np.median(indexed_target[indexed_target!=0], axis=(0,1))

            # if (sam_color_mask[idx1, idx2]!=0).sum() > 10000:
            #     continue
            sam_color_mask[idx1, idx2] *= 0
            sam_color_mask += ((m['segmentation'][:, :, np.newaxis]) * mean_color[np.newaxis, np.newaxis, :])
            
            # sam_color_mask[m['segmentation']] += np.mean(cartoon_images[1][m['segmentation']], axis=2)

            # break

        sam_color_mask[0, 0, :] = np.array([1, 1, 1])
        
        output.append(sam_color_mask/255)

    return output

def colormap_postprocessor(colormap, origin):
    factor = 1
    
    if colormap.max() < 1.1 and origin.max() < 1.1:
        factor = 255
    
    tmp = Image.fromarray((origin * factor).astype(np.uint8))
    origin_gray = np.array(tmp.convert("L"))
    origin_rgb = np.array(tmp.convert("RGB"))
    
    tmp = Image.fromarray((colormap * factor).astype(np.uint8))
    colormap_gray = np.array(tmp.convert("L"))
    colormap_rgb = np.array(tmp.convert("RGB"))
     
    _, segment = cv2.connectedComponents(colormap_gray)
    
    for v in np.unique(segment):
        if v and ((origin_gray * (colormap_gray == v)) - (colormap_gray * (colormap_gray == v)) > 20).sum() > (colormap_gray == v).sum()*0.5:
            idx1, idx2 = np.where(colormap_gray == v)
            rfill = np.median(origin_rgb[idx1, idx2, 0])
            gfill = np.median(origin_rgb[idx1, idx2, 1])
            bfill = np.median(origin_rgb[idx1, idx2, 2])
            
            colormap_rgb[idx1, idx2, 0] *= 0
            colormap_rgb[idx1, idx2, 0] += rfill.astype(np.uint8)
            
            colormap_rgb[idx1, idx2, 1] *= 0
            colormap_rgb[idx1, idx2, 1] += gfill.astype(np.uint8)
            
            colormap_rgb[idx1, idx2, 2] *= 0
            colormap_rgb[idx1, idx2, 2] += bfill.astype(np.uint8)
            
    return colormap_rgb
 
def get_mode(patch):
    patch_gray = patch.sum(axis=2)
    non_zero_patch = [i for i in patch_gray.reshape(-1) if i]
    mode_value = mode(non_zero_patch).mode
    idx1, idx2 = np.where(patch_gray == mode_value[0])
    return patch[idx1[0], idx2[0]]
 
def fill_zero_values(cmap):
    idx1, idx2 = np.where(cmap.sum(axis=2)==0)
    
    for i in range(len(idx1)):
        patch = cmap[max(idx1[i]-2, 0):idx1[i]+3, min(idx2[i]-2, cmap.shape[1]):idx2[i]+3]
        
        kernel_size = 5
        while 1:
            if patch.sum() != 0:
                break
            else:
                patch = cmap[max(idx1[i]-int(kernel_size//2), 0):idx1[i]+int(kernel_size//2), 
                             min(idx2[i]-int(kernel_size//2), cmap.shape[1]):idx2[i]+int(kernel_size//2)]
                kernel_size += 2
                
        mode_value = get_mode(patch)
        cmap[idx1[i], idx2[i], :] = mode_value
       
    return cmap
