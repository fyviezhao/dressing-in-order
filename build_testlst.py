import os
import shutil
import tqdm

if __name__ == '__main__':
    data_root = '/data/Datasets/'
    data_list = ['douyin/videos_2001_like', 'douyin-2/videos_2001_0703', 'douyin-3/videos_2001_0920', 'douyin-4/videos_4004_0920']
    
    f_test_lst = open('/data/Datasets/DiOR/Dance50k_noshuffle/test.lst', 'w')
    f_test_pair = open('/data/Datasets/DiOR/Dance50k_noshuffle/fasion-pairs-test.csv', 'w')
    f_test_pair.write(',from,to\n')
    cnt = 0
    test_pair_path = '/data/Datasets/DiOR/Dance50k/douyin_test_up_no_shuffle.txt'
    f = open(test_pair_path, 'r')
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        source_path = line.rstrip().strip().split()[0]
        source_parse_path = source_path.split('/')[0] + '/' + source_path.split('/')[1] + '_hrparse_cihp' + '/' + source_path.split('/')[2] + '/' + source_path.split('/')[3].replace('.jpg','.png')
        target_path = line.rstrip().strip().split()[1]
        target_parse_path = target_path.split('/')[0] + '/' + target_path.split('/')[1] + '_hrparse_cihp' + '/' + target_path.split('/')[2] + '/' + target_path.split('/')[3].replace('.jpg','.png')
        source_vid = source_path.split('/')[-2] + '-' + source_path.split('/')[-1]
        target_vid = target_path.split('/')[-2] + '-' + target_path.split('/')[-1]
        shutil.copy(data_root+source_path, '/data/Datasets/DiOR/Dance50k_noshuffle/test/'+source_vid)
        shutil.copy(data_root+target_path, '/data/Datasets/DiOR/Dance50k_noshuffle/test/'+target_vid)
        shutil.copy(data_root+source_parse_path, '/data/Datasets/DiOR/Dance50k_noshuffle/testM_lip/'+source_vid.replace('.jpg','.png'))
        shutil.copy(data_root+target_parse_path, '/data/Datasets/DiOR/Dance50k_noshuffle/testM_lip/'+target_vid.replace('.jpg','.png'))
        f_test_lst.write(source_vid + '\n')
        f_test_lst.write(target_vid + '\n')
        f_test_pair.write(str(cnt)+','+source_vid+','+target_vid+'\n')
        cnt += 1
    f_test_lst.close()
    f_test_pair.close()


