export NGF=32
export DATAROOT=/data/Datasets/DiOR/DeepFashionDX
export NET_G=dior
export NAME=dior_deepfashiondx
export PRETRAINED_FLOWNET_PATH=/data/Projects-warehouse/DiOR/checkpoints/flownet_warmup_deepfashiondx/latest_net_Flow.pth
export CHECKPOINT_DIR=/data/projects-warehouse/DiOR/checkpoints

python train.py --model dior \
--name $NAME --dataroot $DATAROOT \
--batch_size 4 --lr 1e-4 --init_type orthogonal \
--loss_coe_seg 0.1 --checkpoints_dir $CHECKPOINT_DIR \
--netG $NET_G --ngf $NGF \
--netD gfla --ndf 32 --n_layers_D 4 \
--n_epochs 160002 --n_epochs_decay 0 --lr_update_unit 4000 \
--print_freq 200 --display_freq 5000 --save_epoch_freq 10000 --save_latest_freq 2000 \
--n_cpus 8 --gpu_ids 3 --continue_train \
--flownet_path $PRETRAINED_FLOWNET_PATH --frozen_flownet \
--random_rate 0.8 --perturb --square --crop_size 512


rm -rf $CHECKPOINT_DIR/$NAME/latest_net_D*

