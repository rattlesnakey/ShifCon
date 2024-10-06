#!/usr/bin/env python
# coding=utf-8
import argparse
import logging
import math
import os
import random
import datasets
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
from deepspeed.accelerator import get_accelerator
import time
import gc
import json
from utils import count_learnable_params, str2bool, DataCollatorForContrastive, DataCollatorForSeq2Seq_GenShift
from transformers.models.llama.modeling_llama import  LLAMA_INPUTS_DOCSTRING, add_start_docstrings_to_model_forward
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import List, Optional, Tuple, Union
from types import MethodType

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizerFast,
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from model_forward_fn import original_llama_forward, modify_model_forward

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=0,
        help="The rank of lora.",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--en_max_seq_length",
        type=int,
        default=100,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    
    parser.add_argument(
        "--target_max_seq_length",
        type=int,
        default=30,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    
    parser.add_argument(
        "--target_instruction_max_seq_length",
        type=int,
        default=256,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=str2bool,
        default=True,
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        "--use_qlora",
        action="store_true",
        help=(
            "Use qLoRA training - main thing is initialising model in quantised form. Not compatible with deepspeed."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--use_8bit_optimizer',
        action='store_true',
        help='Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed (use deepspeed config instead).',
    )
    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )

    
    parser.add_argument('--layers_to_transform',
                        type=str,
                        default="None",
                        help='e.g., 0,1,2,3,4,5 the layers to apply lora'
    )
    parser.add_argument('--target_modules',
                        type=str,
                        default=None,
                        help='"q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"'
    )
   
    parser.add_argument('--contrastive_scale',
                        type=float,
                        default=1.0,
                        help='contrastive temperature'
    )
    

    parser.add_argument('--contrastive_loss_ratio',
                        type=float,
                        default=1.0,
                        help='contrastive temperature'
    )
    parser.add_argument('--generation_loss_ratio',
                        type=float,
                        default=1.0,
                        help='contrastive temperature'
    )

    parser.add_argument('--contrastive_data_file',
                        type=str,
                        default='None',
                        help='contrastive translation data file'
    )
    parser.add_argument('--contrastive_train_batch',
                        type=int,
                        default=32,
                        help='batch size for contrastive learning'
    )
    
    parser.add_argument(
        "--layer_to_shift_forward",
        type=int,
        help="layer_idx to shift to English Like Space",
        default = 15
    )
    
    parser.add_argument(
        "--layer_to_shift_back",
        type=int,
        help="layer_idx to shift to back to its original representation space",
        default = 20
    )
    
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args






def encode_with_contrastive_format(example, tokenizer, en_max_seq_length=100, target_max_seq_length=30, add_bos=False):
    en_texts = example['en']; target_texts = example['target'] 
    en_tokenized_example = tokenizer(en_texts, return_tensors='pt', max_length=en_max_seq_length, truncation=True)
    en_attention_mask = torch.ones_like(en_tokenized_example.input_ids)
    target_tokenized_example = tokenizer(target_texts, return_tensors='pt', max_length=target_max_seq_length, truncation=True)
    target_attention_mask = torch.ones_like(target_tokenized_example.input_ids)

    return {
        'en_input_ids': en_tokenized_example.input_ids.flatten(),
        'en_attention_mask': en_attention_mask.flatten(),
        'target_input_ids': target_tokenized_example.input_ids.flatten(),
        'target_attention_mask': target_attention_mask.flatten(),
        'target_langs': example['target_lang']
    }

  


def encode_with_prompt_completion_mgsm8k(example, tokenizer, target_instruction_max_seq_length, add_bos=False):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    
    example_text = example['prompt'] + example['chosen']
    example_text = example_text + tokenizer.eos_token

    
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=target_instruction_max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=target_instruction_max_seq_length, truncation=True)
    # mask the prompt part for avoiding loss
    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
    attention_mask = torch.ones_like(input_ids)
    
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
        'lang': example['lang']
    }




def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    unwrapped_model = accelerator.unwrap_model(model)
    # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
    # Otherwise, sometimes the model will be saved with only part of the parameters.
    # Also, accelerator needs to use the wrapped model to get the state_dict.
    state_dict = accelerator.get_state_dict(model)
    
    flag = 0
    if args.lora_rank > 0:
        # When using lora, the unwrapped model is a PeftModel, which doesn't support the is_main_process 
        # and has its own save_pretrained function for only saving lora modules.
        # We have to manually specify the is_main_process outside the save_pretrained function.
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
            flag = 1
   
    if not flag:
        # don't use safetensors for saving for now
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False
        )



def main():
    args = parse_args()
    json.dump(vars(args), open(f"{args.output_dir}/args.json", 'w+'), indent=4)
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    # if you get timeouts (e.g. due to long tokenization) increase this.
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        raw_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
        
    if args.contrastive_data_file != "None":
        data_files = {}
        dataset_args = {}
        data_files["train"] = args.contrastive_data_file
        
        contrastive_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
        

    # Load pretrained model and tokenizer
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
        )
    else:
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported by this script."
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:
        # Warn user if tokenizer and model use different revisions; this is an unusual
        # use case.
        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        if args.use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            device_index = accelerator.local_process_index
            device_map = {"": device_index} # force data-parallel training.
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                load_in_4bit=True,
                quantization_config=bnb_config,
                device_map=device_map,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
                low_cpu_mem_usage=args.low_cpu_mem_usage,
                use_flash_attention_2=True if args.use_flash_attn else False,
                revision=args.model_revision
            )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)
        
        

    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast) or isinstance(tokenizer, PreTrainedTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "</s>",
            "eos_token": "</s>",
            "unk_token": "</s>",
            "pad_token": "[PAD]",
        })
        print(f'######################## num_added_tokens: {num_added_tokens} ########################')
        assert num_added_tokens in [0, 1, 2], "Llama Tokenizer should only add two special token - the pad_token and unk token, or no tokens if pad token present."
        
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, OLMoTokenizerFast):
        # only the eos for olmo, but we use it as bos
        tokenizer.bos_token = tokenizer.eos_token
        assert args.add_bos, "For OLMo, you must add bos token to the beginning of the input sequence."
    

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    # gather deepspeed to get "real" embedding size    
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))
            embedding_size = len(tokenizer)
            


    if args.lora_rank > 0:
        if args.use_qlora:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        
        
        if args.layers_to_transform != "None":
            layers_to_transform = list(map(int, args.layers_to_transform.split('|')))
            
        else:
            layers_to_transform = None
            
        if not args.target_modules:
            target_modules = ["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = args.target_modules.split(',')

            
            
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_rank * 2,
            lora_dropout=args.lora_dropout,
            layers_to_transform=layers_to_transform,
            target_modules=target_modules 
        )
        
        model = get_peft_model(model, peft_config)
        if args.gradient_checkpointing:
            model = model._prepare_model_for_gradient_checkpointing(model)
        
        
        model.print_trainable_parameters()
   
        
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        accelerator.print('using gradient checkpoint ...')
    

    

    
     
                    
    count_learnable_params(model, accelerator)
    accelerator.print("***** trainable module name  *****")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            accelerator.print(f"module: {name}")

    # Preprocessing the datasets.
    if "prompt" in raw_datasets["train"].column_names and "chosen" in raw_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_mgsm8k,
            tokenizer=tokenizer,
            target_instruction_max_seq_length=args.target_instruction_max_seq_length,
            add_bos=args.add_bos
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    
        
    
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids",  "attention_mask", "labels", "lang"]],
            desc="Tokenizing and reformatting instruction data",
        )
        lm_datasets.set_format(type="pt")

    train_dataset = lm_datasets["train"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq_GenShift(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )
    

    if args.contrastive_data_file != "None":
        contrastive_encode_function = partial(
            encode_with_contrastive_format,
            tokenizer=tokenizer,
            en_max_seq_length=args.en_max_seq_length,
            target_max_seq_length=args.target_max_seq_length,
            add_bos=args.add_bos
        )
        
        with accelerator.main_process_first():
            contrastive_encoded_datasets = contrastive_datasets.map(
                contrastive_encode_function,
                batched=False,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                remove_columns=[name for name in contrastive_datasets["train"].column_names if name not in ["en_input_ids",  "en_attention_mask", "target_input_ids", "target_attention_mask", "target_lang"]],
                desc="Tokenizing and reformatting instruction data",
            )
            contrastive_encoded_datasets.set_format(type="pt")
       

        contrastive_train_dataset = contrastive_encoded_datasets["train"]

    # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
            
        contrastive_train_dataloader = DataLoader(
            contrastive_train_dataset, 
            shuffle=False, 
            collate_fn=DataCollatorForContrastive(tokenizer=tokenizer, model=model, padding="longest"),
            batch_size=args.contrastive_train_batch
        )
        
    
    def get_specific_group_params(layer_idxs):
        layer_idxs = layer_idxs.split('|')
        final_layer_idx = [f'layers.{li}.' for li in layer_idxs]
        
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if any(layer_idx in n for layer_idx in final_layer_idx) and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(layer_idx in n for layer_idx in final_layer_idx) and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_parameters
    

    if args.layers_to_transform != "None":
        optimizer_grouped_parameters = get_specific_group_params(args.layers_to_transform)
    else:
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

    

    

    if args.use_qlora:
        from bitsandbytes.optim import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            optim_bits=8 if args.use_8bit_optimizer else 32,
            is_paged=True
        )
        
    else:
        # enhance_area_optimizer = torch.optim.AdamW(enhance_area_params, lr=args.learning_rate)
        # decode_area_optimizer = torch.optim.AdamW(decode_area_params, lr=args.learning_rate)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    
   
    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.contrastive_data_file != "None":
        contrastive_train_dataloader = accelerator.prepare(
            contrastive_train_dataloader
        )
    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("tensorboard", experiment_config)



    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps


    logger.info("***** Running training *****")
    if args.contrastive_data_file != "None":
        logger.info(f"  Num Generation examples = {len(train_dataset)} Num Contrastive examples = {len(contrastive_train_dataset)}")
        logger.info(f"  Instantaneous Generation batch size per device = {args.per_device_train_batch_size} Contrastive Batch Size per device = {args.contrastive_train_batch}")
    else:
        logger.info(f"  Num Generation examples = {len(train_dataset)}")
        logger.info(f"  Instantaneous Generation batch size per device = {args.per_device_train_batch_size}")
        
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total Generation optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[
                -1
            ]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    training_logs = []
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_generation_loss = 0
        
        # if args.contrastive_data_file != "None":
        total_contrastive_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
            if args.contrastive_data_file != "None":
                active_contrastive_dataloader = accelerator.skip_first_batches(
                    contrastive_train_dataloader, resume_step
                )
        else:
            active_dataloader = train_dataloader
            if args.contrastive_data_file != "None":
                active_contrastive_dataloader = contrastive_train_dataloader
                iter_active_contrastive_dataloader = iter(active_contrastive_dataloader)


        for step, batch in enumerate(active_dataloader):
            if args.contrastive_data_file != "None":
                try:
                    contrastive_batch = next(iter_active_contrastive_dataloader)
                except StopIteration:
                    iter_active_contrastive_dataloader = iter(active_contrastive_dataloader)
                    contrastive_batch = next(iter_active_contrastive_dataloader)
                

            
            with accelerator.accumulate(model):
                if args.contrastive_data_file != "None":
                    if args.lora_rank > 0:
                        obj = model.module.base_model.model.model
                        obj.forward = MethodType(original_llama_forward(), obj)
                    else:
                        obj = model.module.model
                        obj.forward = MethodType(original_llama_forward(), obj)
                    
                    en_outputs = model(
                            input_ids=contrastive_batch['en_input_ids'], 
                            attention_mask=contrastive_batch['en_attention_mask'],
                            output_hidden_states=True,
                            use_cache=False
                        )

                    target_outputs = model(
                            input_ids=contrastive_batch['target_input_ids'], 
                            attention_mask=contrastive_batch['target_attention_mask'],
                            output_hidden_states=True,
                            use_cache=False
                        )
                
                    en_attention_mask = contrastive_batch['en_attention_mask']
                    target_attention_mask = contrastive_batch['target_attention_mask']
                    assert len(set(contrastive_batch['target_langs'])) == 1, 'lang in a batch need to be the same!'
                    cur_contrastive_batch_target_lang = contrastive_batch['target_langs'][0]
                    

                    if cur_contrastive_batch_target_lang not in all_langs_vecs:

                        all_langs_vecs[cur_contrastive_batch_target_lang] = dict()
                        all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_forward] = {'embed':torch.zeros(model.module.config.hidden_size, dtype=en_outputs['hidden_states'][0].dtype).to(en_outputs['hidden_states'][0].device), "count": 0}
                        all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_back] = {'embed':torch.zeros(model.module.config.hidden_size, dtype=en_outputs['hidden_states'][0].dtype).to(en_outputs['hidden_states'][0].device), "count": 0}

                    if 'en' not in all_langs_vecs:
                        all_langs_vecs['en'] = dict()
                        all_langs_vecs['en'][args.layer_to_shift_forward] = {'embed':torch.zeros(model.module.config.hidden_size, dtype=en_outputs['hidden_states'][0].dtype).to(en_outputs['hidden_states'][0].device), "count": 0}
                        all_langs_vecs['en'][args.layer_to_shift_back] = {'embed':torch.zeros(model.module.config.hidden_size, dtype=en_outputs['hidden_states'][0].dtype).to(en_outputs['hidden_states'][0].device), "count": 0}
                        
                    
                    def embedding_without_pad(batch_embed, batch_mask):
                        valid_sen_embeds = []
                        for single_embed, single_mask in zip(batch_embed, batch_mask):
                            valid_tokens = single_embed[single_mask==1]
                            valid_sen_embed = valid_tokens.mean(dim=0)
                            valid_sen_embeds.append(valid_sen_embed)
                        return torch.stack(valid_sen_embeds)
                    
                    
                    
            
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='mean') 
                    en_embeds = en_outputs['hidden_states'][args.layer_to_shift_forward:args.layer_to_shift_back+1]
                
                    
                    target_embeds = target_outputs['hidden_states'][args.layer_to_shift_forward:args.layer_to_shift_back+1]
                    
                    
                    contrastive_loss = 0; count_contrast_layer = 0; layers_num = len(en_embeds)
                    for idx, (en_embed, target_embed) in enumerate(zip(en_embeds, target_embeds)):
                        en_sen_embed = embedding_without_pad(en_embed, en_attention_mask)
                        target_sen_embed = embedding_without_pad(target_embed, target_attention_mask)
                        en_sen_embed = en_sen_embed / en_sen_embed.norm(dim=-1, keepdim=True)
                        target_sen_embed = target_sen_embed / target_sen_embed.norm(dim=-1, keepdim=True)
                        
                        en_sen_embed_mean = torch.mean(en_sen_embed, dim=0)
                        target_sen_embed_mean = torch.mean(target_sen_embed, dim=0)
                        
                        if idx != layers_num -1:
                            target_sen_embed = target_sen_embed - target_sen_embed_mean + en_sen_embed_mean
                        
                        if idx == 0:
                            all_langs_vecs['en'][args.layer_to_shift_forward]['embed'] += en_sen_embed_mean.detach(); all_langs_vecs['en'][args.layer_to_shift_forward]['count'] += 1
                            all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_forward]['embed'] += target_sen_embed_mean.detach(); all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_forward]['count'] += 1
                        
                        if idx == layers_num -1:
                            all_langs_vecs['en'][args.layer_to_shift_back]['embed'] += en_sen_embed_mean.detach(); all_langs_vecs['en'][args.layer_to_shift_back]['count'] += 1
                            all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_back]['embed'] += target_sen_embed_mean.detach(); all_langs_vecs[cur_contrastive_batch_target_lang][args.layer_to_shift_back]['count'] += 1
                        
                        

                        if idx != layers_num -1:
                            logits_per_en = args.contrastive_scale * en_sen_embed @ target_sen_embed.t()
                            ground_truth = torch.arange(len(logits_per_en)).long().to(logits_per_en.device)
                            current_contrastive_loss = (
                                loss_fct(logits_per_en, ground_truth)
                                + loss_fct(logits_per_en.t(), ground_truth)
                            ) / 2
                            contrastive_loss += current_contrastive_loss
                            count_contrast_layer += 1
                        
                    contrastive_loss = contrastive_loss / count_contrast_layer
                    
                try:
                    assert len(set(batch['langs'])) == 1, 'lang in a batch need to be the same!'
                    cur_gen_batch_lang = batch['langs'][0]
                except AssertionError:
                    import pdb; pdb.set_trace()
                
                if args.lora_rank > 0:
                    if cur_gen_batch_lang in all_langs_vecs:
                        obj = model.module.base_model.model.model
                        obj.forward = MethodType(modify_model_forward(all_langs_vecs['en'], all_langs_vecs[cur_gen_batch_lang], args.layer_to_shift_forward, args.layer_to_shift_back, cur_gen_batch_lang), obj)
                else:
                    if cur_gen_batch_lang in all_langs_vecs:
                        obj = model.module.model
                        obj.forward = MethodType(modify_model_forward(all_langs_vecs['en'], all_langs_vecs[cur_gen_batch_lang], args.layer_to_shift_forward, args.layer_to_shift_back, cur_gen_batch_lang), obj)
                        
                        
                outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], use_cache=False)
                generation_loss = outputs.loss
                
                if args.contrastive_data_file != "None":
                    loss = args.generation_loss_ratio * generation_loss + args.contrastive_loss_ratio * contrastive_loss
                    accelerator.backward(loss)
                else:
                    loss = generation_loss
                    accelerator.backward(generation_loss)
                    contrastive_loss = generation_loss - generation_loss
                
                # We keep track of the loss at each logged step
                total_generation_loss += generation_loss.detach().float()
                total_contrastive_loss += contrastive_loss.detach().float()
                # clip gradient norm. don't do this with deepspeed
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step() 
                
                

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_generation_loss = accelerator.gather(total_generation_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    avg_contrastive_loss = accelerator.gather(total_contrastive_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {loss}, Generation Loss: {avg_generation_loss}, Contrastive Loss: {avg_contrastive_loss}, Generation loss ratio: {args.generation_loss_ratio}, Contrastive Loss ratio: {args.contrastive_loss_ratio}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "generation_loss": avg_generation_loss,
                                "contrastive_loss": avg_contrastive_loss,
                            },
                            step=completed_steps,
                        )
                        if accelerator.is_main_process:
                            training_logs.append(
                                {
                                    "learning_rate": lr_scheduler.get_last_lr()[0],
                                    "generation_loss": avg_generation_loss,
                                    "contrastive_loss": avg_contrastive_loss,
                                    "step":completed_steps
                                }
                            )
                            
                    total_generation_loss = 0
                    total_contrastive_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        if accelerator.is_main_process:
                            tokenizer.save_pretrained(output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break
        
        
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            output_dir = f"{args.output_dir}/epoch_{epoch}"
            os.makedirs(output_dir, exist_ok=True)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                import json
                json.dump(training_logs, open(f"{output_dir}/training_logs.json", 'w+'), indent=4)
                
                save_langs_vecs(f'{output_dir}/langs_vecs', all_langs_vecs)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    if args.with_tracking:
        accelerator.end_training()

    
def save_langs_vecs(output_dir, all_langs_vecs):
    os.makedirs(output_dir, exist_ok=True)
    for lang, layers_item in all_langs_vecs.items():
        for layer_idx, item in layers_item.items():
            cur_lang_output_dir = f'{output_dir}/{lang}'
            os.makedirs(cur_lang_output_dir, exist_ok=True)
            cur_lang_output_path = f'{output_dir}/{lang}/{layer_idx}.pth'
            cur_lang_vec = item['embed'] / item['count']
            torch.save(cur_lang_vec, cur_lang_output_path)

if __name__ == "__main__":
    all_langs_vecs = dict()
    main()
