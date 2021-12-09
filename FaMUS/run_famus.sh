#!/bin/bash


### 
## torch==1.0.0 torchvision==0.2.1
###


split_file='red_noise_nl_0.4'
mkdir -p ./output/${split_file}
CUDA_VISIBLE_DEVICES=0  nohup python -u train_famus_redminiimage.py \
    --lr 0.02 \
    --name ${split_file}  \
    --warmup_epochs 30  \
    --lambda_u 50 \
    --split ${split_file} \
    --network 'PreActResNet18' \
    -b 64 \
    --alpha 1.5 \
    --step1 200 \
    --step2 250 \
    &> ./output/${split_file}/train_log.txt &


# split_file='red_noise_nl_0.2'
# mkdir -p ./output/${split_file}
# CUDA_VISIBLE_DEVICES=0  nohup python -u train_famus_redminiimage.py \
#     --lr 0.02 \
#     --name ${split_file}  \
#     --warmup_epochs 30  \
#     --lambda_u 50 \
#     --split ${split_file} \
#     --network 'PreActResNet18' \
#     -b 64 \
#     --alpha 1.0 \
#     --step1 200 \
#     --step2 250 \
#     &> ./output/${split_file}/train_log.txt &



# split_file='red_noise_nl_0.6'
# mkdir -p ./output/${split_file}
# CUDA_VISIBLE_DEVICES=0 nohup python -u train_famus_redminiimage.py \
#     --lr 0.02 \
#     --name ${split_file}  \
#     --warmup_epochs 30  \
#     --lambda_u 50 \
#     --split ${split_file} \
#     --network 'PreActResNet18' \
#     -b 64 \
#     --alpha 1.5  \
#     --step1 100 \
#     --step2 200  \
#     &> ./output/${split_file}/train_log.txt &


# split_file='red_noise_nl_0.8'
# mkdir -p ./output/${split_file}
# CUDA_VISIBLE_DEVICES=0  nohup python -u train_famus_redminiimage.py \
#     --lr 0.02 \
#     --name ${split_file}  \
#     --warmup_epochs 30  \
#     --lambda_u 100 \
#     --split ${split_file} \
#     --network 'PreActResNet18' \
#     -b 64 \
#     --alpha 2.0 \
#     --step1 100 \
#     --step2 200  \
#     &> ./output/${split_file}/train_log.txt &



