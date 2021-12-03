export OUTPUT_DIR=/data/Projects-warehouse/DiOR/outputs/dior_deepfashiondx_noshuffle
export NAME=dior_deepfashiondx
export LOAD_EP=latest
export NET_G=dior
export NET_E=adgan
export NGF=32
export DATAROOT=/data/Datasets/DiOR/DeepFashionDX


# generate images
python garment_transfer.py --model dior --dataroot $DATAROOT \
--name $NAME --epoch $LOAD_EP --eval_output_dir $OUTPUT_DIR  \
--netE $NET_E --netG $NET_G --ngf $NGF \
--n_cpus 4 --gpu_ids 5  --batch_size 8 --square --crop_size 512
