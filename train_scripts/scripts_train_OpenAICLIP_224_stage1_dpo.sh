export CUDA_VISIBLE_DEVICES=4,5,6,7
accelerate launch --config_file "train_configs/accelerate_config.yaml" --ddp_timeout 180000000 \
train_OpenAICLIP_stage1_dpo.py --config "train_configs/train_OpenAICLIP_224_stage1_dpo.yaml"
