import os
import json
import numpy as np
import torch
from utils import pose_utils
import tqdm

def load_pose_from_json(pose_json, target_size=(256,256), orig_size=(256,256)):
    with open(pose_json, 'r') as f:
        anno = json.load(f)
    if len(anno['people']) < 1:
        a,b = target_size
        return torch.zeros((18,a,b))
    anno = list(anno['people'][0]['pose_keypoints_2d'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])
    
    x[8:-1] = x[9:]
    y = np.array(anno[::3])
    y[8:-1] = y[9:]
    x[x==0] = -1
    y[y==0] = -1
    coord = np.concatenate([x[:,None], y[:,None]], -1)
    pose  = pose_utils.cords_to_map(coord, target_size, orig_size)
    pose = np.transpose(pose,(2, 0, 1))
    pose = torch.Tensor(pose)
    return pose[:18]

if __name__ == '__main__':
    data_list = ['douyin-2/videos_2001_0703_op', 'douyin/videos_2001_like_op', 'douyin-3/videos_2001_0920_op', 'douyin-4/videos_4004_0920_op']
    data_root = '/data/Datasets'
 
    # train_txt = open('/data/Datasets/DiOR/Dance50k/train.lst', 'r')
    test_txt = open('/data/Datasets/DiOR/Dance50k_noshuffle/test.lst', 'r')
    # train_list = []
    # for line in train_txt.readlines():
    #     train_list.append(line.rstrip())
    test_list = []
    for line in test_txt.readlines():
        test_list.append(line.rstrip())

    # pose_train_txt = open('/data/Datasets/DiOR/Dance50k/dance50k-annotation-train2.cvs', 'w')
    # pose_train_txt.write('name:keypoints_y:keypoints_x\n')
    pose_test_txt = open('/data/Datasets/DiOR/Dance50k_noshuffle/fasion-annotation-test.csv', 'w')
    pose_test_txt.write('name:keypoints_y:keypoints_x\n')
    im_name_list = []
    for data in data_list:
        data_path = os.path.join(data_root, data)
        for pose_dir in tqdm.tqdm(os.listdir(data_path)):
            pose_dir_path = os.path.join(data_path, pose_dir)
            for pose_name in os.listdir(pose_dir_path):
                pose_path = os.path.join(pose_dir_path, pose_name)
                pose_25 = np.load(pose_path)
                pose_18 = np.vstack((pose_25[0:8], pose_25[9:19]))
                pose18_x = str(list(pose_18[:,0]))
                pose18_y = str(list(pose_18[:,1]))

                im_name = pose_dir + '-' + pose_name[:-4]
                # if im_name not in im_name_list:
                    # if im_name in train_list:
                    #     pose_train_txt.write(im_name + ':' + str(pose18_y) + ':' + pose18_x + '\n')
                if im_name in test_list:
                    pose_test_txt.write(im_name + ':' + str(pose18_y) + ':' + pose18_x + '\n')
                
                # im_name_list.append(im_name)
