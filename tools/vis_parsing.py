""" This script serves to colorize the grayscale parsing labels
"""
import os
import PIL.Image as Image
import numpy as np
from tqdm import tqdm

# color map
label_colors = [(0,0,0),(128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85),
                (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0), (0,0,255), 
                (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]

def decode_labels(mask, num_classes=20):
    """Decode batch of segmentation masks.
    
    Args:
      mask: size of (batch_size,1,height,width), result of inference after taking argmax.
      num_classes: number of classes to predict (including background).
    
    Returns:
      A batch with RGB images of the same (height,width) as the input. 
    """
    h, w= mask.shape
    img = Image.new('RGB', (w,h))
    pixels = img.load()
    for j_, j in enumerate(mask): # j_: row_id, row
        for k_, k in enumerate(j): # k_: col_id, k: value of mask[i, j, k]
            if k < num_classes: # k: [0, num_classes-1]
                pixels[k_, j_] = label_colors[k]
    
    return img

if __name__ == '__main__':
    parse_root = '/data/Datasets/DiOR/DeepFashionDX/testM_lip_new'
    dst_root = '/data/Datasets/DiOR/DeepFashionDX/testM_lip_new_vis'
    os.makedirs(dst_root, exist_ok=True)
    for parse_name in tqdm(os.listdir(parse_root)):
        parse_path = os.path.join(parse_root, parse_name)
        parse_gray = np.asarray(Image.open(parse_path))
        parse_rgb = decode_labels(parse_gray)
        parse_rgb.save(os.path.join(dst_root, parse_name))