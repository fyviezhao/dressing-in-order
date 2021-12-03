import os
import shutil
import tqdm

if __name__ == '__main__':
    data_root = '/data/Datasets/deepfashion_dongxin/img_highres/'
    
    f_test_lst = open('/data/Datasets/DiOR/DeepFashionDX_noshuffle/test.lst', 'w')
    f_test_pair = open('/data/Datasets/DiOR/DeepFashionDX_noshuffle/fasion-pairs-test.csv', 'w')
    f_test_pair.write(',from,to\n')
    cnt = 0

    test_pair_path = '/data/Datasets/DiOR/DeepFashionDX/deepfashion_test_up_no_shuffle.txt'
    f = open(test_pair_path, 'r')
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        source_pid = line.rstrip().strip().split()[0]
        source_parse_pid = source_pid.replace('.jpg','.png')
        target_pid = line.rstrip().strip().split()[1]
        target_parse_pid = target_pid.replace('.jpg','.png')
        source_pid_path = os.path.join(data_root, source_pid)
        source_parse_path = source_pid_path.replace('img_highres', 'img_highres_hrparse_cihp').replace('.jpg','.png')
        target_pid_path = os.path.join(data_root, target_pid)
        target_parse_path = target_pid_path.replace('img_highres', 'img_highres_hrparse_cihp').replace('.jpg','.png')
        shutil.copy(source_pid_path, '/data/Datasets/DiOR/DeepFashionDX_noshuffle/test/'+source_pid.replace('/','-'))
        shutil.copy(target_pid_path, '/data/Datasets/DiOR/DeepFashionDX_noshuffle/test/'+target_pid.replace('/','-'))
        shutil.copy(source_parse_path, '/data/Datasets/DiOR/DeepFashionDX_noshuffle/testM_lip/'+source_parse_pid.replace('/','-'))
        shutil.copy(target_parse_path, '/data/Datasets/DiOR/DeepFashionDX_noshuffle/testM_lip/'+target_parse_pid.replace('/','-'))
        f_test_lst.write(source_pid.replace('/','-')+'\n')
        f_test_lst.write(target_pid.replace('/','-')+'\n')
        f_test_pair.write(str(cnt)+','+source_pid.replace('/','-')+','+target_pid.replace('/','-')+'\n')
        cnt += 1
    f_test_lst.close()
    f_test_pair.close()


