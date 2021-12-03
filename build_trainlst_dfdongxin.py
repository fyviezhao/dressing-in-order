import os
import shutil
import tqdm

if __name__ == '__main__':
    data_root = '/data/Datasets/deepfashion_dongxin/img_highres'
    
    f_test_lst = open('/data/Datasets/DiOR/DeepFashionDX/test.lst', 'w')
    f_test_pair = open('/data/Datasets/DiOR/DeepFashionDX/test-pairs-test.csv', 'w')
    f_test_pair.write(',from,to\n')
    cnt = 0

    test_pair_path = '/data/Datasets/deepfashion_dongxin/test_up_pair.txt_up'
    f = open(test_pair_path, 'r')
    for line in f.readlines():
        pid = line.rstrip().strip().split()[0]
        pid_path = os.path.join(data_root, pid)
        im_names = []
        for im_name in os.listdir(pid_path):
            im_name_new = pid.replace('/','-') + '-' + im_name
            f_test_lst.write(str(im_name_new) + '\n')
            im_path = os.path.join(pid_path, im_name)
            shutil.copy(im_path, '/data/Datasets/DiOR/DeepFashionDX/test/'+im_name_new)
            im_names.append(im_name_new)
        im_names.sort()
        inds = [[int(i.split(',')[0]), int(i.split(',')[1])] for i in line.rstrip().strip().split()[1:]]
        for ind in inds:
            # tmp = str(cnt)+','+str(im_names[ind[0]])+','+str(im_names[ind[1]])+'\n'
            f_test_pair.write(str(cnt)+','+str(im_names[ind[0]])+','+str(im_names[ind[1]])+'\n')
            cnt += 1
    f_test_lst.close()
    f_test_pair.close()


