import os
import collections
from collections import defaultdict
import shutil
from itertools import chain
import tqdm

def checkIfDuplicates(listOfElems):
    ''' Check if given list contains any duplicates '''
    if len(listOfElems) == len(set(listOfElems)):
        return False
    else:
        return True

if __name__ == '__main__':
    data_root = '/data/Datasets/'
    data_list = ['douyin-1/videos_2001_like', 'douyin-2/videos_2001_0703', 'douyin-3/videos_2001_0920', 'douyin-4/videos_4004_0920']

    vids = []
    for data_name in data_list:
        data_path = os.path.join(data_root, data_name)
        for vid_dir in tqdm.tqdm(os.listdir(data_path)):
            for im_name in os.listdir(os.path.join(data_path, vid_dir)):
                im_path = os.path.join(data_path, vid_dir, im_name)
                im_name_new = vid_dir + '-' + im_name
                shutil.copy(im_path, '/data/Datasets/DiOR/Dance50k/img_highres/' + im_name_new)
     
    




    

