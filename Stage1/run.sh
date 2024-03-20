#!/bin/bash

python main_train.py \
--save_path exp/exp1 \
--batch_size 64 \
--lr 0.001 \
--lr_decay 0.90 \
--train_list /home/ai03/lsy/dataset/deepship/train_cut.txt \
--test_list /home/ai03/lsy/dataset/deepship/test_cut.txt \
--train_path /home/ai03/lsy/dataset/deepship/train \
--test_path /home/ai03/lsy/dataset/deepship/test \
--musan_path /data08/Others/musan_split \
--test_interval 1 \
--eval