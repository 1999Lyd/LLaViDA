# Large Language Vision Driving Assistant

## Dataset
[SFT-Training](https://drive.google.com/drive/folders/1tRhodZ-tRRluO_4yVm9wRBzxPHZ-WtRb?usp=sharing) 23k cot+trajectory predictions 
[Validation](https://drive.google.com/drive/folders/1tRhodZ-tRRluO_4yVm9wRBzxPHZ-WtRb?usp=sharing) 6k samples
[Validation(direct)](https://drive.google.com/drive/folders/1tRhodZ-tRRluO_4yVm9wRBzxPHZ-WtRb?usp=sharing) 6k samples with direct answer instructions
[Nuscenes](https://www.nuscenes.org/nuscenes#download) Unzip all file into one folder(nuscenes) and put into exp/data
[NuPlan](https://www.nuscenes.org/nuplan) Unzip all file into one folder(nuplan) and put into exp/data

## Model ckpt
[LLaVA-NeXT](https://huggingface.co/Share4oReasoning/Open-LLaVA-NeXT-LLaMA3-8B): base model 

[SFT+TPO](https://drive.google.com/drive/folders/1tRhodZ-tRRluO_4yVm9wRBzxPHZ-WtRb?usp=sharing): SFT 3 epochs + TPO 1 epochs

[Direct](https://drive.google.com/drive/folders/1tRhodZ-tRRluO_4yVm9wRBzxPHZ-WtRb?usp=sharing): SFT +TPO with mixed data, optimized for fast inference


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

## open loop evaluation
```
cd llavida
# cot version
bash scripts/eval.sh
# direct version
bash scripts/eval_direct.sh
```
## close loop evaluation
Setting up NuPlan dataset following [official doc](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html).

Setting up similation env and run the script
```
# install nuplan-devkit
git clone https://github.com/motional/nuplan-devkit.git && cd nuplan-devkit
pip install -e .
pip install -r requirements.txt
bash sim_llavida_planner_runner.sh
```

## citation
```

```

## Acknowledge
Thanks to 

(open-llava-next)[https://github.com/xiaoachen98/Open-LLaVA-NeXT]: for base model and sft training

(LLaVA-Hound)[https://github.com/RifleZhang/LLaVA-Hound-DPO/tree/main]: for dpo related 

