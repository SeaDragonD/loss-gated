#!/bin/bash

python main_train.py \
--save_path exp/exp2 \
--batch_size 64 \
--lr 0.001 \
--lr_decay 0.5 \
--train_list /home/ai03/lsy/dataset/deepship/train_cut_new.txt \
--test_list /home/ai03/lsy/dataset/deepship/test_cut_new.txt \
--train_path /home/ai03/lsy/dataset/deepship/train \
--test_path /home/ai03/lsy/dataset/deepship/test \
--musan_path /data08/Others/musan_split \
--test_interval 1