
python3 train.py \
--dataroot ../../../data/selfie2anime \
--name anime_FastCUT_monce_new \
--CUT_mode FastCUT \
--gpu_ids 3 \
--load_size 128 \
--crop_size 128 \
--batch_size 16 \
--lambda_NCE 1 \
--flip_equivariance False