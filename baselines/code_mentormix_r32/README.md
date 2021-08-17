## MentorMix on CNWL with the resolution of 32x32
- The result of MentorMix on CNWL with the resolution of 32x32 is below:

  |Model|0.2|0.4|0.6|0.8|Mean|
  |---|---|---|---|---|---|
  |MentorMix|51.02|47.14|43.80|33.46|43.85|



- `NL` means the noise rate which is ranged from `[0.2, 0.4, 0.6, 0.8]`. Please note that for different `NL` values, some hyper-parameters are different. Details can be found in `train_script.sh`
- Please download the TFRecord data from the [link](https://drive.google.com/file/d/12KLkFaIDLlQ4bYIQL7xVeqX3H1d0r4_o/view?usp=sharing). Please organize the file structure as like `data/red_mini_imagenet_s32/red_noise_nl_0.4/data` in the case of the `NL=0.4`.
- An training and evaluation example:

```shell
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
```
