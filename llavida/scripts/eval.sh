#!/usr/bin/env bash
set -euo pipefail

model_name=exp/sft/LLaVA-Reasoner-SFT-context
output_dir=exp/sft/post_sft_eval_result

sft_dir=exp/data
pretrain_dir=$IMAGE_INSTRUCTION_DIR/pretrain

# data
data_paths="$sft_dir/data_val.jsonl"
output_paths="$output_dir/answers.jsonl"

export TOKENIZERS_PARALLELISM=false

# ========= GPUs =========
# Use 1+ GPU IDs (comma-separated). Example below uses 4 GPUs.
gpu_ids="6"                     # <-- set this to the devices you want
export CUDA_VISIBLE_DEVICES="$gpu_ids"
n_gpu=$(echo "$gpu_ids" | tr "," "\n" | wc -l | xargs)
echo "Using $n_gpu GPU(s): $gpu_ids"
# ========================

export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + rand % 1000))

cache_dir=$CACHE_DIR
export cache_dir

image_folder=$IMAGE_DATA_DIR

save_name=$(basename "$output_dir")
export WANDB_PROJECT=llava-llama3-reasoning
export WANDB_NAME="${save_name}"
export report_to=wandb
wandb_args="--report_to $report_to"

echo "input model:  $model_name"
echo "output dir:   $output_dir"
mkdir -p "$output_dir"

version=llava_llama_3

# launch (multi-proc if n_gpu>1; still works with n_gpu=1)
nohup torchrun --standalone --nnodes=1 --nproc_per_node="$n_gpu" --master_port "$port" \
  llava/eval/model_vqa_loader.py \
    --model-path "$model_name" \
    --question-file "$data_paths" \
    --answers-file "$output_paths" \
    --image-folder "$image_folder" \
    --temperature 0.2 \
    --top_p 0.2 \
    --max_new_tokens 4096 \
    --conv-mode llava_llama_3 \
  > "eval.log" 2>&1 &
