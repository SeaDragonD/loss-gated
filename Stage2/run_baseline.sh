#!/bin/bash

python main_train.py \
--save_path exp/exp2 \
--batch_size 128 \
--lr 0.001 \
--train_list /home/ai03/lsy/dataset/deepship/train_cut_new_3.txt \
--test_list /home/ai03/lsy/dataset/deepship/test_cut_new_3.txt \
--train_path /home/ai03/lsy/dataset/deepship/train \
--test_path /home/ai03/lsy/dataset/deepship/test \
--val_path /home/ai03/lsy/dataset/deepship/snr_experiment/test_soundscape_p15 \
--musan_path /data08/Others/musan_split \
--rir_path /data08/Others/RIRS_NOISES/simulated_rirs \
--init_model /home/ai03/lsy/common_model/loss_gated/Loss-Gated-V3/Stage1/exp/exp8/model/model000000053.model \
--test_interval 1 \

