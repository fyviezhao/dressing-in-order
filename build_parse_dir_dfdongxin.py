import shutil
import os
import tqdm

data_root = '/data/Datasets/deepfashion_dongxin/img_highres_hrparse_cihp'

train_txt = open('/data/Datasets/DiOR/DeepFashionDX/train.lst', 'r')
test_txt = open('/data/Datasets/DiOR/DeepFashionDX/test.lst', 'r')
train_list = []
for line in train_txt.readlines():
    train_list.append(line.rstrip())
test_list = []
for line in test_txt.readlines():
    test_list.append(line.rstrip())

for data in os.listdir(data_root):
    data_path = os.path.join(data_root, data)
    for attr_dir in os.listdir(data_path):
        attr_dir_path = os.path.join(data_path, attr_dir)
        for parse_dir in os.listdir(attr_dir_path):
            parse_dir_path = os.path.join(attr_dir_path, parse_dir)
            for parse_name in os.listdir(parse_dir_path):
                parse_path = os.path.join(parse_dir_path, parse_name)
                parse_name_new = data + '-' + attr_dir + '-' + parse_dir + '-' + parse_name
                im_name = parse_name_new.replace('.png', '.jpg')
                if im_name in train_list:
                    shutil.copy(parse_path, '/data/Datasets/DiOR/DeepFashionDX/trainM_lip/'+parse_name_new)
                if im_name in test_list:
                    shutil.copy(parse_path, '/data/Datasets/DiOR/DeepFashionDX/testM_lip/'+parse_name_new)
