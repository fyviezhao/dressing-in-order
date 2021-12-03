import os
import cv2
import numpy as np
import tqdm

im_root = '/data/Datasets/DiOR/Dance50k_noshuffle/test'
parse_root = '/data/Datasets/DiOR/Dance50k_noshuffle/testM_lip'

dst_root = '/data/Datasets/DiOR/Dance50k_noshuffle/test_fg'

for im_name in tqdm.tqdm(os.listdir(im_root)):
    im_path = os.path.join(im_root, im_name)
    parse_name = im_name.replace('.jpg','.png')
    parse_path = os.path.join(parse_root, parse_name)

    im = cv2.imread(im_path)
    parse = cv2.imread(parse_path)
    mask = (parse > 0).astype(np.float32)
    im_masked = im * mask
    cv2.imwrite(os.path.join(dst_root, im_name), im_masked.astype(np.uint8))