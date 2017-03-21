#!/bin/bash

echo "train the model using CPU"

th main.lua ./data/train.t7 1000 -batch_size 50 -snapshot_epoch 50 -log outlog_CPU.log -dataset_size 3000
