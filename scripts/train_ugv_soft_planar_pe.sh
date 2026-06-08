#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/semantic-OETR

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

nohup /root/miniconda3/bin/python -m torch.distributed.launch \
  --use-env \
  --nproc_per_node=1 \
  --master_port=29512 \
  train.py \
  --config_path configs/baseline/carla_fullimg_ugvrelay_soft_planar_finetune.py \
  --batch_size 4 \
  --num_workers 4 \
  --epoch 12 \
  --learning_rate 2e-5 \
  --seed 42 \
  --validation \
  > /root/autodl-tmp/semantic-OETR/OUTPUT/train_ugv_soft_planar_pe.log 2>&1 &

echo "started pid=$!"
echo "log: /root/autodl-tmp/semantic-OETR/OUTPUT/train_ugv_soft_planar_pe.log"
