#!/bin/sh

BATCH_SIZE=4
WORKER_SIZE=4
GPU_SIZE=2
CPU_SIZE=8
DATASET="sr-hack-2019-dataset"
MAX_EPOCHS=30

nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --use_attention --max_epochs $MAX_EPOCHS"
