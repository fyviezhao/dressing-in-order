import os
import shutil
import tqdm

if __name__ == '__main__':
    data_root = '/data/Datasets'
    data_list = ['douyin-1/videos_2001_like', 'douyin-2/videos_2001_0703', 'douyin-3/videos_2001_0920', 'douyin-4/videos_4004_0920']
    
    f_test_lst = open('/data/Datasets/DiOR/Dance50k/test.lst', 'w')
    f_test_pair = open('/data/Datasets/DiOR/Dance50k/dance50k-pairs-test.scv', 'w')
    f_test_pair.write(',from,to\n')
    cnt = 0
    for data in tqdm.tqdm(data_list):
        test_pair_path = os.path.join(data_root, data.split('/')[0], 'val_pair.txt')
        f = open(test_pair_path, 'r')
        for line in f.readlines():
            vid = line.rstrip().strip().split()[0]
            vid_path = os.path.join(data_root, data, vid)
            im_names = []
            for im_name in os.listdir(vid_path):
                im_name_new = vid + '-' + im_name
                f_test_lst.write(str(im_name_new) + '\n')
                im_path = os.path.join(data_root, 'DiOR/Dance50k/img_highres', im_name_new)
                shutil.copy(im_path, '/data/Datasets/DiOR/Dance50k/test/'+im_name_new)
                im_names.append(im_name_new)
            im_names.sort()
            inds = [[int(i[0]), int(i[2])] for i in line.rstrip().strip().split()[1:]]
            for ind in inds:
                tmp = str(cnt)+','+str(im_names[ind[0]])+','+str(im_names[ind[1]])+'\n'
                f_test_pair.write(str(cnt)+','+str(im_names[ind[0]])+','+str(im_names[ind[1]])+'\n')
                cnt += 1
    f_test_lst.close()
    f_test_pair.close()


