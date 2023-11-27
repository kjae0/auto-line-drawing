from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from PIL import Image
from tqdm import tqdm

import pickle as pkl
import numpy as np
import argparse
import os

def main(args):
    model = sam_model_registry["default"](checkpoint=args.model_dir)
    mask_generator = SamAutomaticMaskGenerator(model)

    os.makedirs(args.save_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(args.data_dir))
    for im in tqdm(image_files, total=len(image_files), ncols=60):
        img = Image.open(os.path.join(args.data_dir, im))
        img = np.array(img)
        
        masks = mask_generator.generate(img)
        
        with open(os.path.join(args.save_dir, f"{''.join(im.split('/')[-1].split('.')[:-1])}_mask.pkl"), "wb") as f:
            pkl.dump(masks, f)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    
    args.add_argument("--data_dir", type=str, required=True)
    args.add_argument("--save_dir", type=str, required=True)
    args.add_argument("--model_dir", type=str, required=True)
    
    args = args.parse_args()
    
    print("MASK GENERATION STARTED.")
    main(args)
    print("Done!")
    