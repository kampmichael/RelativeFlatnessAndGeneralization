#!/bin/bash

TASK_NAME="rte"

export APPLY_FAM=true
export FAM_LAMBDA=3e6

for ((i=1; i<=5; i=i+1))
do

python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_predict \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 20 \
  --overwrite_output_dir \
  --output_dir results/$TASK_NAME/apply_fam_${APPLY_FAM}_lambda_${FAM_LAMBDA}/${i} \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_steps 50 \
  --logging_strategy steps \
  --seed ${i} \
  --apply_fam $APPLY_FAM \
  --fam_lambda $FAM_LAMBDA \
  --fp16
done