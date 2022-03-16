#!/usr/bin/env bash

set -x

export TOKENIZERS_PARALLELISM=false

PYTHON=/usr/bin/python3.7
git clone https://github.com/huggingface/transformers
cp run_summarization.py ./transformers/examples/pytorch/summarization/
TRAIN_SCRIPT=./transformers/examples/pytorch/summarization/run_summarization.py
TRAIN=/home/dhruv.kumar/IteraTeR/dataset/human_sent_level/train.json
VALID=/home/dhruv.kumar/IteraTeR/dataset/human_sent_level/dev.json
OUTPUT=bart_model/
sha1sum ${TRAIN_SCRIPT}

${PYTHON} ${TRAIN_SCRIPT} \
  --model_name_or_path facebook/bart-large \
  --do_train \
  --do_eval \
  --train_file "${TRAIN}" \
  --validation_file "${VALID}" \
  --per_device_train_batch_size=2 \
  --per_device_eval_batch_size=2 \
  --num_train_epochs 5 \
  --gradient_accumulation_steps 8 \
  --evaluation_strategy "steps" \
  --eval_steps 200 \
  --save_steps 100 \
  --predict_with_generate \
  --logging_steps 50 \
  --output_dir "${OUTPUT}" \
  --overwrite_output_dir \
  --text_column before_sent \
  --summary_column after_sent \
  --learning_rate 3e-5