#!/bin/bash

export WANDB_DISABLED=true
wandb offline
kge_models=("data/distmult.pt" "data/rgcn.pt" "data/transe.pt" "data/rotate.pt")
adapter_types=("qformer")
task_type="subgraph_mc"

for kge_model in "${kge_models[@]}"; do
  for adapter_type in "${adapter_types[@]}"; do
    output_dir="${task_type}/${adapter_type}-$(basename $kge_model .pt)"
    log_file="log_${task_type}_${adapter_type}_$(basename $kge_model .pt).txt"

    echo "Running experiment with kge_model=$kge_model, adapter_type=$adapter_type, task_type=$task_type"
    echo "Output directory: $output_dir"
    echo "Log file: $log_file"

    CUDA_VISIBLE_DEVICES=0 python finetune.py \
      --base_model 'MODEL PATH' \
      --data_path "data/${task_type}/train.json" \
      --output_dir "$output_dir" \
      --num_epochs 3 \
      --learning_rate 2e-4 \
      --batch_size 8 \
      --micro_batch_size 8 \
      --kge_model "$kge_model" \
      --cutoff_len 256 \
      --adapter_type "$adapter_type"
    wait
  done
done
