# ShifCon: Enhancing Non-Dominant Language capabilities with a Shift-based Contrastive Framework
This is the code repository of "ShifCon: Enhancing Non-Dominant Language capabilities with a Shift-based Contrastive Framework"


# 🚀 Introduction

To enhance the performance of non-dominant languages in LLMs, we propose a Shift-based Contrastive framework (ShifCon) that includes shift-toward and shift-backward projections, as well as multilingual contrastive learning (MCL).

<img src="./img/overview.png" width="700" >

## 📄 Get Started

## 📝 Setup

```
conda env create -n llm python==3.11 -y

conda activate llm

pip install -r requirements.txt

# important package
transformers==4.39.3
deepspeed==0.13.1
accelerate==0.22.0
bitsandbytes==0.41.1
ninja==1.11.1.1
vllm==0.2.4
```


## 💻 Models
- You can download our already trained first stage model from huggingface hub [Link] and fill the variable `model_dict["mgsm-llama-2-7b"]` (in `pipeline_train-eval.sh`) with its path content.


## 📥 Data
- First, download the data from QAlign repo: [Link](https://github.com/NJUNLP/QAlign)
- Then, go to `data` directory run `python build_gen_data.py` and `python build_trans_data.py` respectively.
- Finally, fill variable `data_dict["MGSM8k-gen"]` and `data_dict["MGSM8k-trans"]` in `pipeline_train-eval.sh` with their corresponding paths.


## ⛳️ Run

```shell
#! batch size
BATCH_SIZE_PER_GPU=14
TOTAL_BATCH_SIZE=56


#! model path
declare -A model_dict
model_dict["mgsm-llama-2-7b"]="/your/path/to/math-llama-model" 


GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"


MODEL_NAME=mgsm-llama-2-7b
MODEL_PATH=${model_dict["${MODEL_NAME}"]}

EN_MAX_LENGTH=100
TARGET_MAX_LENGTH=256
TARGET_INSTRUCTION_MAX_LEN=512 

CONTRASTIVE_TRAIN_BATCH=12
SCHEDULER_TYPE=cosine #! linear, constant, cosine
WARMUP_RATIO=0.01 

DS_CONFIG=${BASE_DIR}/ds_configs/stage3_offloading_accelerate.conf 
MAIN_PORT=29500


declare -A data_dict
data_dict["MGSM8k-gen"]='/your/data/path/to/mgsm_instruct_batch_{batch_size}.jsonl'
data_dict["MGSM8k-trans"]='/your/data/path/to/mgsm_trans_{batch_size}.jsonll'




DATASET_NAME=MGSM8k-gen
CONTRASTIVE_DATASET_NAME=MGSM8k-trans
```
- set the path of your model and training data to the `model_dict` and `data_dict`, and then execute `bash pipeline_train_eval.sh` to train your model.

