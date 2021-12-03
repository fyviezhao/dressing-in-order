export DATAROOT=/data/Datasets/DiOR/DeepFashionDX

python train.py --dataroot $DATAROOT \
--name 'flownet_warmup_deepfashiondx' \
--batch_size 8 --lr 1e-4 \
--init_type orthogonal \
--n_epochs 45000 --n_epochs_decay 45000 --lr_update_unit 10000 \
--print_freq 50 --display_freq 2000 --save_epoch_freq 20000 --save_latest_freq 2000 \
--loss_coe_sty 0 --loss_coe_rec 0 --loss_coe_per 0 \
--loss_coe_flow_cor 2 \
--n_cpus 8 --gpu_ids 2 \
--continue_train \
--square  --crop_size 512 \
--model flow --no_trial_test