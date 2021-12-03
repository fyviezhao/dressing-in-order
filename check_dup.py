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
    # data_list = ['douyin-1/videos_2001_like', 'douyin-2/videos_2001_0703', 'douyin-3/videos_2001_0920', 'douyin-4/videos_4004_0920']
    data_list = ['douyin-1', 'douyin-2', 'douyin-3', 'douyin-4']

    vids = []
    for data_name in data_list:
        data_path = os.path.join(data_root, data_name)
        val_list_path = os.path.join(data_path, 'val_pair.txt')
        val_list = open(val_list_path, 'r')
        for line in val_list.readlines():
            vid = line.rstrip().strip().split()[0]
            vids.append(vid)

    print(checkIfDuplicates(vids))
     
    




    

