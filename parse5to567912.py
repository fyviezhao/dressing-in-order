import cv2
import numpy as np
import os
import tqdm

parse_root = '/data/Datasets/DiOR/DeepFashionDX_noshuffle/testM_lip'
dst_root = '/data/Datasets/DiOR/DeepFashionDX_noshuffle/testM_lip_new'
os.makedirs(dst_root, exist_ok=True)

for parse_name in tqdm.tqdm(os.listdir(parse_root)):
    parse_path = os.path.join(parse_root, parse_name)
    parse = cv2.imread(parse_path,0)
    parse[np.where(parse==6)] = 5
    parse[np.where(parse==7)] = 5
    parse[np.where(parse==9)] = 5
    parse[np.where(parse==12)] = 5
    cv2.imwrite(os.path.join(dst_root,parse_name), parse)
