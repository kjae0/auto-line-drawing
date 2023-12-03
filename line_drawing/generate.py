from PIL import Image
from tqdm import tqdm
from rembg import remove

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import utils

def main(args):
    # load data
    images_files = os.listdir(os.path.join(args.data_dir, "cartoonized"))
    images_files.sort()

    origin_images_files = os.listdir(os.path.join(args.data_dir, "origin"))
    origin_images_files.sort()

    images = []
    cartoon_images = []
    backgrounds = []

    for im in images_files:
        img = Image.open(os.path.join(args.data_dir, 'cartoonized', im)).convert("RGB")
        print(im)
        print(f"image size : {img.size}")    
        cartoon_images.append(np.array(img))
        img = remove(np.array(img))
        backgrounds.append(img[:, :, -1]==0)
        img = Image.fromarray(img)
        img = np.array(img.convert("RGB"))
        images.append(img)
        
    origin_images = []

    for im in origin_images_files:
        img = Image.open(os.path.join(args.data_dir, 'origin', im)).convert("RGB")
        print(im)
        print(f"image size : {img.size}")    
        
        origin_images.append(np.array(img))
        
    cartoon_backgrounds = []
    for i in range(len(backgrounds)):
        idx = np.where(backgrounds[i])
        cartoon_backgrounds.append(cartoon_images[i] * np.expand_dims(backgrounds[i].astype(float), axis=2))

    thick_edges = []
    for i in range(len(images)):
        smoothing = utils.bilateral_blur(images[i].astype(np.uint8), 3, 150, 5)
        smoothing = utils.median_blur(smoothing, 7)
        smoothing = utils.gaussian_blur(smoothing, (5, 5), 3, 3)
        
        edge = utils.make_edge_thicker(utils.canny_detection(smoothing, 70, 100, apertureSize=3), kernel_size=2)
        thick_edges.append(edge)
        
    sam_edges = utils.get_SAM_edges(os.path.join(args.data_dir, "masks"), thick_edges, score_threshold=0.7)
    colormap = utils.get_SAM_colormap(os.path.join(args.data_dir, "masks"), cartoon_images, score_threshold=0.7)    
    colormap = utils.fill_large_chunks(colormap, cartoon_images)

    color_segment = []
    for i in range(len(images)):
        img_with_edge = colormap[i].copy()
        # img_with_edge = utils.gaussian_blur(img_with_edge, (5, 5), 3, 3)
        img_with_edge[thick_edges[i]!=0] = 0
        img_with_edge[sam_edges[i]!=0] = 0
        color_segment.append((img_with_edge.copy()*255).astype(np.uint8))
        
    result = []
    for i in range(len(color_segment)):
        res = utils.colormap_postprocessor(color_segment[i], cartoon_images[i])
        print(res.shape)
        res = utils.fill_zero_values(res)
        idx1, idx2 = np.where(images[i].sum(axis=1)==0)
        res[idx1, idx2, :] *= 0
        res[thick_edges[i]!=0] = 0
        result.append(res.copy())
        
    os.makedirs("./result", exist_ok=True)
    
    for i in range(len(result)):
        img = Image.fromarray(result[i])
        img.save(f"./result/{i}.png")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", required=True, type=str)
    # args.add_argument("--save_dir", required=True, type=str)

    args = args.parse_args()

    main(args)
    