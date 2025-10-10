model_name=exp/ckpt/Open-LLaVA-NeXT-LLaMA3-8B
output_dir=exp/sft/LLaVA-Reasoner-SFT-context

sft_dir=exp/data
pretrain_dir=$IMAGE_INSTRUCTION_DIR/pretrain

# data composition
# pretrain 2k mix + direct + cot

data_paths="\
$sft_dir/sft_data_train.jsonl \
"
export TOKENIZERS_PARALLELISM=false
gpu_ids=3,4,5,6
export CUDA_VISIBLE_DEVICES=$gpu_ids
n_gpu=$(echo $gpu_ids | tr "," "\n" | wc -l)
echo "Using $n_gpu GPUs: $gpu_ids"
export PYTHONPATH=.
rand=$RANDOM
port=$((19000 + $rand % 1000))

cache_dir=$CACHE_DIR
export cache_dir=$cache_dir

image_folder=$IMAGE_DATA_DIR

save_name=$(basename $output_dir)
export WANDB_PROJECT=llava-llama3-reasoning
export WANDB_NAME=${save_name}
export report_to=wandb
wandb_args="--report_to $report_to"

echo input model: $model_name
echo output model: $output_dir
mkdir -p $output_dir

# tokenizer_name=$model_name_or_path
version=llava_llama_3

# pretrain
# total batch 4 node * 8 gpu * 4 batch = 128
# BASE_LR=2e-5
# VIT_LR=2e-6

# sft
batch_size=1
grad_cum=1
BASE_LR=5e-6
VIT_LR=5e-7

# continue train doesn't require
# --pretrain_mm_mlp_adapter checkpoints/llava-v1.6-8b_llama3-8b_pretrain_lcs-558k_ft-mlp-lr-1e-3/mm_projector.bin \
# --save_only_model True

nohup torchrun --nproc_per_node=$n_gpu --master_port=$port llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --unfreeze_mm_vision_tower True --mm_vision_tower_lr ${VIT_LR} \
    --model_name_or_path $model_name \
    --version $version \
    --data_paths $data_paths \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio anyres \
    --group_by_modality_length True --dataloader_drop_last True \
    --mm_vision_select_layer -2 \
    --mm_vision_select_feature patch \
    --mm_patch_merge_type spatial_unpad \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir $output_dir \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${grad_cum} \
    --evaluation_strategy "no" \
    --save_strategy "steps" --save_steps 0.333 --save_total_limit 1 --save_only_model True \
    --learning_rate ${BASE_LR} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 15000 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    ${wandb_args} 2>&1 | tee $output_dir/train.log > sft.log 2>&1 &
