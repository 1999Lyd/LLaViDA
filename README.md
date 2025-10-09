# Large Language Vision Driving Assistant

## Dataset
[SFT-Training](https://huggingface.co/datasets/Share4oReasoning/sft_data) 23k cot+trajectory predictions 
[Validation](https://huggingface.co/datasets/Share4oReasoning/dpo_data) 6k samples
[Validation(direct)](https://huggingface.co/datasets/Share4oReasoning/dpo_data) 6k samples with direct answer instructions

## Model ckpt
[SFT](https://huggingface.co/Share4oReasoning/Open-LLaVA-NeXT-LLaMA3-8B): base model 

[SFT+TPO](https://huggingface.co/Share4oReasoning/LLaVA-Reasoner-SFT-preview): SFT 3 epochs + TPO 1 epochs

[Direct](https://huggingface.co/Share4oReasoning/LLaVA-Reasoner-SFT): SFT +TPO with mixed data, optimized for fast inference


## setup 
```
# setup environment, need to fill in the required fields
source setup/setup_env.sh
```
## sft
```
cd llavida
bash scripts_sft/run_sft.sh \
$SAVE_DIR/sft/LLaVA-Reasoner-SFT-context
```

## dpo
```
cd llavida
bash scripts_dpo/run_dpo.sh \
$SAVE_DIR/dpo/LLaVA-Reasoner-DPO-context
```
## citation
```

```

## Acknowledge
Thanks to 

(open-llava-next)[https://github.com/xiaoachen98/Open-LLaVA-NeXT]: for base model and sft training

(LLaVA-Hound)[https://github.com/RifleZhang/LLaVA-Hound-DPO/tree/main]: for dpo related 

