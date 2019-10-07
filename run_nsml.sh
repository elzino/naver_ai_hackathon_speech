#!/bin/sh

BATCH_SIZE=16
WORKER_SIZE=2
GPU_SIZE=1
CPU_SIZE=4
DATASET="sr-hack-2019-50000"
MAX_EPOCHS=60

nsml run -g $GPU_SIZE -c $CPU_SIZE -d $DATASET -a "--batch_size $BATCH_SIZE --workers $WORKER_SIZE --max_epochs $MAX_EPOCHS"
