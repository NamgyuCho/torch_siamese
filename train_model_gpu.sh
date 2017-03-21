#!/bin/bash

echo "train the model using GPU"

th main.lua ./data/train.t7 1000 -batch_size 100 -snapshot_epoch 50 -gpu -log outlog_GPU.log -dataset_size 0 -weights ./snapshot/snapshot_epoch_2000_saved.net -criterion ./snapshot/_criterion.net
