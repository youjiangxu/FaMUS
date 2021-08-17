#!/bin/bash

# #  pip install --upgrade tensorflow-probability==0.8.0 --user
# NL=0.4
# ALPHA=8
# PERCENTILE=0.5
# split_file=red_noise_nl_${NL}
# gpu_id=0
# EXPDIR=./output/mini_imagenet_models/resnet32/${split_file}/mentormix/
# nohup python code/mini_imagenet_train_mentormix.py \
#   --batch_size=128 \
#   --dataset_name=mini_imagenet \
#   --split_name=${split_file} \
#   --loss_p_percentile=${PERCENTILE} \
#   --trained_mentornet_dir=mentornet_models/mentornet_pd \
#   --burn_in_epoch=10 \
#   --data_dir=data/red_mini_imagenet_s32/ \
#   --train_log_dir="${EXPDIR}/train" \
#   --studentnet=resnet32 \
#   --max_number_of_steps=200000 \
#   --device_id=${gpu_id} \
#   --num_epochs_per_decay=30 \
#   --mixup_alpha=${ALPHA} \
#   --nosecond_reweight &> ./output/train_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &

# nohup python -u code/mini_imagenet_eval.py \
#   --dataset_name=mini_imagenet \
#   --checkpoint_dir="${EXPDIR}/train"\
#   --data_dir=data/red_mini_imagenet_s32 \
#   --eval_dir="${EXPDIR}/eval_val" \
#   --studentnet=resnet32 \
#   --device_id=${gpu_id} &> ./output/eval_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &




# NL=0.2
# PERCENTILE=0.7
# ALPHA=2
# split_file=red_noise_nl_${NL}
# gpu_id=0
# EXPDIR=./output/mini_imagenet_models/resnet32/${split_file}/mentormix/
# nohup python code/mini_imagenet_train_mentormix.py \
#   --batch_size=128 \
#   --dataset_name=mini_imagenet \
#   --split_name=${split_file} \
#   --loss_p_percentile=${PERCENTILE} \
#   --trained_mentornet_dir=mentornet_models/mentornet_pd \
#   --burn_in_epoch=10 \
#   --data_dir=data/red_mini_imagenet_s32/ \
#   --train_log_dir="${EXPDIR}/train" \
#   --studentnet=resnet32 \
#   --max_number_of_steps=200000 \
#   --device_id=${gpu_id} \
#   --num_epochs_per_decay=30 \
#   --mixup_alpha=${ALPHA} \
#   --nosecond_reweight &> ./output/train_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &

# nohup python -u code/mini_imagenet_eval.py \
#   --dataset_name=mini_imagenet \
#   --checkpoint_dir="${EXPDIR}/train"\
#   --data_dir=data/red_mini_imagenet_s32 \
#   --eval_dir="${EXPDIR}/eval_val" \
#   --studentnet=resnet32 \
#   --device_id=0 &> ./output/eval_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &




# NL=0.6
# ALPHA=4
# PERCENTILE=0.3
# split_file=red_noise_nl_${NL}
# gpu_id=1
# EXPDIR=./output/mini_imagenet_models/resnet32/${split_file}/mentormix/
# nohup python code/mini_imagenet_train_mentormix.py \
#   --batch_size=128 \
#   --dataset_name=mini_imagenet \
#   --split_name=${split_file} \
#   --loss_p_percentile=${PERCENTILE} \
#   --trained_mentornet_dir=mentornet_models/mentornet_pd \
#   --burn_in_epoch=10 \
#   --data_dir=data/red_mini_imagenet_s32/ \
#   --train_log_dir="${EXPDIR}/train" \
#   --studentnet=resnet32 \
#   --max_number_of_steps=200000 \
#   --device_id=${gpu_id} \
#   --num_epochs_per_decay=30 \
#   --mixup_alpha=${ALPHA} \
#   --nosecond_reweight &> ./output/train_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &

# nohup python -u code/mini_imagenet_eval.py \
#   --dataset_name=mini_imagenet \
#   --checkpoint_dir="${EXPDIR}/train"\
#   --data_dir=data/red_mini_imagenet_s32 \
#   --eval_dir="${EXPDIR}/eval_val" \
#   --studentnet=resnet32 \
#   --device_id=${gpu_id} &> ./output/eval_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &




NL=0.8
ALPHA=8
PERCENTILE=0.1
split_file=red_noise_nl_${NL}
gpu_id=0
EXPDIR=./output/mini_imagenet_models/resnet32/${split_file}/mentormix/
nohup python code/mini_imagenet_train_mentormix.py \
  --batch_size=128 \
  --dataset_name=mini_imagenet \
  --split_name=${split_file} \
  --loss_p_percentile=${PERCENTILE} \
  --trained_mentornet_dir=mentornet_models/mentornet_pd \
  --burn_in_epoch=10 \
  --data_dir=data/red_mini_imagenet_s32/ \
  --train_log_dir="${EXPDIR}/train" \
  --studentnet=resnet32 \
  --max_number_of_steps=200000 \
  --device_id=${gpu_id} \
  --num_epochs_per_decay=30 \
  --mixup_alpha=${ALPHA} \
  --nosecond_reweight &> ./output/train_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &

nohup python -u code/mini_imagenet_eval.py \
  --dataset_name=mini_imagenet \
  --checkpoint_dir="${EXPDIR}/train"\
  --data_dir=data/red_mini_imagenet_s32 \
  --eval_dir="${EXPDIR}/eval_val" \
  --studentnet=resnet32 \
  --device_id=${gpu_id} &> ./output/eval_log_R${NL}_A${ALPHA}_P${PERCENTILE}.txt &
