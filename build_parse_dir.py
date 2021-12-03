import shutil
import os
import tqdm

data_root = '/data/Datasets'
data_list = ['douyin-1/videos_2001_like_hrparse_cihp', 'douyin-2/videos_2001_0703_hrparse_cihp',
            'douyin-3/videos_2001_0920_hrparse_cihp', 'douyin-4/videos_4004_0920_hrparse_cihp']

train_txt = open('/data/Datasets/DiOR/Dance50k/train.lst', 'r')
test_txt = open('/data/Datasets/DiOR/Dance50k/test.lst', 'r')
train_list = []
for line in train_txt.readlines():
    train_list.append(line.rstrip())
test_list = []
for line in test_txt.readlines():
    test_list.append(line.rstrip())

for data in data_list:
    data_path = os.path.join(data_root, data)
    for parse_dir in tqdm.tqdm(os.listdir(data_path)):
        parse_dir_path = os.path.join(data_path, parse_dir)
        for parse_name in os.listdir(parse_dir_path):
            parse_path = os.path.join(parse_dir_path, parse_name)
            parse_name_new = parse_dir + '-' + parse_name
            im_name = parse_dir + '-' + parse_name.replace('.png', '.jpg')
            # if im_name in train_list:
            #     shutil.copy(parse_path, '/data/Datasets/DiOR/Dance50k/trainM_cihp/'+parse_name_new)
            if im_name in test_list:
                shutil.copy(parse_path, '/data/Datasets/DiOR/Dance50k/testM_lip/'+parse_name_new)
            else:
                continue
