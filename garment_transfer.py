
from datasets import create_visual_ds
from options.test_options import TestOptions
from models import create_model
import os, torch
from tqdm import tqdm

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import imageio, os

GID = [2,5,1,3]
PID = [0,4,6,7]

def define_visualizer(model_name):
    return FlowVisualizer 

    
class FlowVisualizer:
        
    @staticmethod
    def swap_garment(data, model, gid=5, display_mask=False, step=-1, prefix="", generate_out_dir='.'):
        model.eval()
        #display_mask = True # display_mask and 'seg' in model.visual_names
        imgs, parses, poses = data
        imgs = imgs.to(model.device)
        parses = parses.float().to(model.device)
        poses = poses.float().to(model.device)
        
        # all_mask = [torch.zeros(imgs.size())[None].to(model.device)]
        all_mask = [(parses.unsqueeze(1) == gid).float().expand_as(imgs).to(model.device).unsqueeze(0)]
        all_fake = [imgs[None]]
        # import pdb; pdb.set_trace()
        for img, parse, pose in zip(imgs, parses, poses):
            curr_to_pose = pose[None].expand_as(poses)
            seg = model.encode_single_attr(imgs, parses, poses, curr_to_pose, gid)
            gsegs = model.encode_attr(img[None].expand_as(imgs),
                                            parse[None].expand_as(parses),
                                            curr_to_pose,
                                            curr_to_pose,
                                            GID)
            psegs = model.encode_attr(img[None].expand_as(imgs),
                                            parse[None].expand_as(parses),
                                            curr_to_pose,
                                            curr_to_pose,
                                            PID)
            
            gsegs[GID.index(gid)] = seg
            
            fake_img = model.decode(curr_to_pose, psegs, gsegs) #, attns)
            all_fake += [fake_img[None]]
            if display_mask:
                # import pdb; pdb.set_trace()
                N,C,H,W = imgs.size()
                all_mask += [model.get_seg_visual(gid).expand(N,3,H,W)[None]]

        # display
        all_fake = torch.cat(all_fake)
        _,_,H,W = fake_img.size()
        all_fake[0] = F.interpolate(all_fake[0], (H,W))
        #all_mask[0] = F.interpolate(all_mask[0], (H,W))
        if display_mask:
            all_mask = torch.cat(all_mask)
        ret = []
        for i in range(all_fake.size(1)):
            if display_mask:
                print_img = torch.cat([all_fake[:,i], all_mask[:,i]],2)
            else:
                print_img = all_fake[:, i]
            ret.append(print_img)
        print_img = torch.cat(ret, 2)
        #print_img = (all_fake[:,i] + 1) / 2

        print_img = (print_img + 1) / 2
        print_img = print_img.float().cpu().detach()
        tmp = print_img.permute(0,2,3,1)
        tmp = (tmp.numpy() * 255).astype(np.uint8)
        # tmp = np.concatenate((tmp[0],tmp[1],tmp[2]),axis=1)
        tmp = np.concatenate((tmp[0][:512,...],tmp[0][512:,...],tmp[2][:512,...]), axis=1)
        
        imageio.imwrite(generate_out_dir + '/' + f'{prefix}.jpg', tmp)
    
def generate_val_img(visual_ds, model, opt, step=0, generate_out_dir='.'):
    error_log = open('/data/Projects-warehouse/DiOR/outputs/error_log.txt', 'w')
    model.eval()
    Visualizer = define_visualizer(opt.model)
    with torch.no_grad():
        # patches = visual_ds.get_patches()
        for cata in visual_ds.selected_keys:
            try:
                data = visual_ds.get_attr_visual_input(cata)
            except:
                error_log.write(cata + '\n')
                continue
            Visualizer.swap_garment(data, model, prefix=cata, step=step, gid=5, generate_out_dir=generate_out_dir)
            print("[visualize] swap garments - %s" % cata)

if __name__ == '__main__':
    opt = TestOptions().parse()   # get training options
    if opt.square:
        opt.crop_size = (opt.crop_size, opt.crop_size)
    else:
        opt.crop_size = (opt.crop_size, max(1,int(opt.crop_size*1.0/256*176)))

    dataset = create_visual_ds(opt)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    load_iter = model.setup(opt)
    model.eval()

    generate_out_dir = os.path.join(opt.eval_output_dir + "_%s"%opt.epoch)
    print("generate images at %s" % generate_out_dir)
    os.makedirs(generate_out_dir, exist_ok=True)
    model.isTrain = False

    # generate
    count = 0
    generate_val_img(dataset, model, opt, generate_out_dir=generate_out_dir)
