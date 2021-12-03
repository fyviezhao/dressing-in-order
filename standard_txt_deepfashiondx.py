import os

shuffle_f = open('/data/Datasets/DiOR/DeepFashionDX/deepfashion_test_up_no_shuffle.txt','r')
lines = shuffle_f.readlines()

std_txt = open('/data/Datasets/DiOR/DeepFashionDX_noshuffle/standard_test_anns.txt', 'w')
std_txt.write('pose\n')
std_txt.write('v0200c120000bs0leo700hftjqr66jg0-frmaes_0132.jpg\n') # 随便给了个姿势
std_txt.write('attr\n')

for i, line in enumerate(lines):
    source = line.rstrip().strip().split()[0].replace('/','-') # for deepfashiondx
    # source = line.rstrip().strip().split()[0] # for dance50k
    # source = source.split('/')[2] + '-' + source.split('/')[3]
    std_txt.write(str(i) + ', ' + source + '\n')
    target = line.rstrip().strip().split()[1].replace('/','-')
    # target = line.rstrip().strip().split()[1]
    # target = target.split('/')[2] + '-' + target.split('/')[3]
    std_txt.write(str(i) + ', ' + target + '\n')
