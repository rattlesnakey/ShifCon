export CUDA_VISIBLE_DEVICES=0,1,2,3
export SAFETENSORS_FAST_GPU=1
export CUDA_LAUNCH_BLOCKING=1

BASE_DIR=/root/local/projects/llm
train(){
    accelerate launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --num_processes $NUM_GPUS \
        --use_deepspeed \
        --deepspeed_config_file ${DS_CONFIG} \
        --main_process_port ${MAIN_PORT} \
        ${BASE_DIR}/src/finetune_contrast.py \
        --model_name_or_path ${MODEL_PATH} \
        --use_flash_attn \
        --lora_dropout 0.1 \
        --tokenizer_name ${MODEL_PATH} \
        --use_slow_tokenizer \
        --train_file ${TRAIN_FILE_PATH} \
        --preprocessing_num_workers 16 \
        --checkpointing_steps epoch \
        --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
        --learning_rate ${LR} \
        --lr_scheduler_type ${SCHEDULER_TYPE} \
        --warmup_ratio ${WARMUP_RATIO} \
        --weight_decay 0. \
        --num_train_epochs ${TRAIN_EOPCHS} \
        --output_dir ${OUTPUT_DIR} \
        --with_tracking \
        --report_to tensorboard \
        --logging_steps 1 \
        --gradient_checkpointing ${GRAD_CKPT} \
        --target_modules ${TARGET_MODULES} \
        --contrastive_loss_ratio ${CONTRASTIVE_LOSS_RATIO} \
        --generation_loss_ratio ${GENERATION_LOSS_RATIO} \
        --contrastive_train_batch ${CONTRASTIVE_TRAIN_BATCH} \
        --contrastive_data_file ${CONTRASTIVE_DATA_FILE} \
        --en_max_seq_length ${EN_MAX_LENGTH} \
        --target_max_seq_length ${TARGET_MAX_LENGTH} \
        --target_instruction_max_seq_length ${TARGET_INSTRUCTION_MAX_LEN} \
        --layer_to_shift_forward ${LAYER_TO_SHIFT_FORWARD} \
        --layer_to_shift_back ${LAYER_TO_SHIFT_BACK} \
        2>&1 | tee ${OUTPUT_DIR}/training.log
}



mathoctopus_shift_eval(){
    python ${BASE_DIR}/src/generate_and_eval.py \
    --model_path ${OUTPUT_DIR} \
    --streategy Parallel \
    --batch_size ${EVAL_BATCH_SIZE} \
    --save_dir ${SAVE_DIR}/mgsm8k_shift_${LAYER_TO_SHIFT_FORWARD}_${LAYER_TO_SHIFT_BACK} \
    --lang_only ${LANG_ONLY} \
    --layer_to_shift_forward ${LAYER_TO_SHIFT_FORWARD} \
    --layer_to_shift_back ${LAYER_TO_SHIFT_BACK} \
    --langs_vector_dir_path ${LANGS_VEC_DIR_PATH}


}

######################################################### Training ######################################################### 

MODEL_SIZE=7b
NUM_GPUS=4
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

LR=1e-5
TRAIN_EOPCHS=1
TRAIN_FILE_PATH=${data_dict["${DATASET_NAME}"]}
CONTRASTIVE_DATA_FILE=${data_dict["${CONTRASTIVE_DATASET_NAME}"]}


CONTRASTIVE_LOSS_RATIO=2.0
GENERATION_LOSS_RATIO=0.5

GRAD_CKPT=True
TODAY="$(date +%d-%m-%y)"


LAYER_TO_SHIFT_FORWARD=12
LAYER_TO_SHIFT_BACK=15

OUTPUT_DIR=${BASE_DIR}/output/${TODAY}/${DATASET_NAME}-${CONTRASTIVE_DATASET_NAME}-gen_loss-${GENERATION_LOSS_RATIO}-contrast_loss-${CONTRASTIVE_LOSS_RATIO}-${MODEL_NAME}_${LR}_epoch-${TRAIN_EOPCHS}_gen-batch-${TOTAL_BATCH_SIZE}-contrast-batch-${CONTRASTIVE_TRAIN_BATCH}-shift-forward-${LAYER_TO_SHIFT_FORWARD}-back-${LAYER_TO_SHIFT_BACK}
mkdir -p ${OUTPUT_DIR}

train







    




#########################################################  Evaluation ######################################################### 

LANG_ONLY=None
EVAL_BATCH_SIZE=8
SAVE_DIR=${OUTPUT_DIR}/math_eval
LANG_ONLY=None
LANGS_VEC_DIR_PATH=${OUTPUT_DIR}/langs_vecs
mathoctopus_shift_eval