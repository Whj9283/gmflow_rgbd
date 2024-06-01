#! /bin/bash

# sintel
CHECKPOINT_DIR=checkpoints/sintel-gmflow && \
mkdir -p ${CHECKPOINT_DIR} && \
python main.py \
--launcher none \
--checkpoint_dir ${CHECKPOINT_DIR} \
--stage sintel \
--batch_size 4 \
--val_dataset sintel \
--lr 2e-4 \
--image_size 320 896 \
--padding_factor 16 \
--upsample_factor 8 \
--with_speed_metric \
--val_freq 20000 \
--save_ckpt_freq 10000 \
--num_steps 100000 \
2>&1 | tee -a ${CHECKPOINT_DIR}/train.log