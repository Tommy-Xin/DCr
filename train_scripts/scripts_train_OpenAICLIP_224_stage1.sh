export CUDA_VISIBLE_DEVICES=0,1,2,3
accelerate launch --config_file "train_configs/accelerate_config.yaml" --ddp_timeout 180000000 \
train_OpenAICLIP_stage1.py --config "train_configs/train_OpenAICLIP_224_stage1.yaml"