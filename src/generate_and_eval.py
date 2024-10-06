
from vllm import LLM, SamplingParams
import json
import re
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from typing import Optional, Dict, Sequence, List, Tuple
import argparse
import os
from vllm.model_executor.input_metadata import InputMetadata
from types import MethodType
KVCache = Tuple[torch.Tensor, torch.Tensor]



def modify_model_forward(model, en_lang_layers_vec, cur_target_lang_layers_vec, layer_to_shift_forward, layer_to_shift_back, cur_lang):
    def llama_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            cache_event = None if cache_events is None else cache_events[i]
            layer = self.layers[i]
            
            if i == (layer_to_shift_forward - 1) and cur_lang != 'English':
                cur_layer_en_lang_vec = torch.tensor(en_lang_layers_vec[layer_to_shift_forward], dtype=hidden_states.dtype).to(hidden_states.device)
                cur_layer_cur_lang_vec = torch.tensor(cur_target_lang_layers_vec[layer_to_shift_forward], dtype=hidden_states.dtype).to(hidden_states.device)
                hidden_states = (hidden_states - cur_layer_cur_lang_vec) + cur_layer_en_lang_vec 
            
            if i == (layer_to_shift_back - 1) and cur_lang != 'English':
                cur_layer_en_lang_vec = torch.tensor(en_lang_layers_vec[layer_to_shift_back], dtype=hidden_states.dtype).to(hidden_states.device)
                cur_layer_cur_lang_vec = torch.tensor(cur_target_lang_layers_vec[layer_to_shift_back], dtype=hidden_states.dtype).to(hidden_states.device)
                hidden_states = (hidden_states - cur_layer_en_lang_vec) + cur_layer_cur_lang_vec 

                
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                input_metadata,
                cache_event,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
    return llama_forward
    
    

def main(
    args,
    gsm8k_test_jsonl: str = "./data/mgsm",
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    
    
    
    if args.use_vllm:
        model = LLM(
                model=model_path,
                tokenizer=model_path,
                tensor_parallel_size=torch.cuda.device_count(),
                trust_remote_code=True,
                dtype=torch.bfloat16 if is_bf16 else 'auto',
            )
        tokenizer = model.llm_engine.tokenizer
    else:
        model, tokenizer = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")
    
    

    batch_llama = get_batch_llama(model, tokenizer, args)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = f"{model_path}/output_{model.dtype}_bs{batch_size}/{args.streategy}/msgm"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.lang_only == 'None':
        langs = ['Swahili', 'English','Chinese','Bengali', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'Thai']
    else:
        langs = args.lang_only.split(',')
        # langs = [args.lang_only]
    sources = []
    targets = []
    results = {}
    langs_mapping_dict = {'Swahili':"sw", 'English':"en",'Chinese':"zh",'Bengali':"bn", 'German':"de", 'Spanish':"es", 'French':"fr", 'Japanese':"ja", 'Russian':"ru", 'Thai':"th"}
    
    
    if args.langs_vector_dir_path != "None":
        en_lang_layers_vec = dict()
        en_lang_layers_vec[args.layer_to_shift_forward] = torch.load(f"{args.langs_vector_dir_path}/{langs_mapping_dict['English']}/{args.layer_to_shift_forward}.pth")
        en_lang_layers_vec[args.layer_to_shift_back] = torch.load(f"{args.langs_vector_dir_path}/{langs_mapping_dict['English']}/{args.layer_to_shift_back}.pth")

    
    for lang in langs:
        print(f'===========we are testing in {lang}====================')
    
        if args.langs_vector_dir_path != "None":
            
            cur_target_lang_layers_vec = dict()
            cur_target_lang_layers_vec[args.layer_to_shift_forward] = torch.load(f"{args.langs_vector_dir_path}/{langs_mapping_dict[lang]}/{args.layer_to_shift_forward}.pth")
            cur_target_lang_layers_vec[args.layer_to_shift_back] = torch.load(f"{args.langs_vector_dir_path}/{langs_mapping_dict[lang]}/{args.layer_to_shift_back}.pth")
            
            
            obj = model.llm_engine.workers[0].model_runner.model.model
            obj.forward = MethodType(modify_model_forward(model, en_lang_layers_vec, cur_target_lang_layers_vec, args.layer_to_shift_forward, args.layer_to_shift_back, lang), obj)
            
             
        if args.streategy == 'Parallel':
            if lang ==  'En_gsm8k':
                with open('./data/test_use.jsonl', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
    
            else:
                with open(f'{gsm8k_test_jsonl}/mgsm_{lang}.json', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
        else:
            if lang ==  'En_gsm8k':
                with open('./data/test_use.jsonl', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
    
            else:
               
                with open(f'{gsm8k_test_jsonl}/mgsm_English.json', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
        lang_dir = f'{save_dir}/{lang}'
        Path(lang_dir).mkdir(parents=True, exist_ok=True)
        gen_datas_jsonl = Path(lang_dir) / f"gen_{lang}_datas.jsonl"
        
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
            cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
            input_str_list, output_str_list = gsm8k_batch_gen(lang, 
                [d["query"] for d in cur_gsm8k_batch], batch_llama
            )
            for j, (gsm8k_data, input_str, output_str) in enumerate(
                zip(cur_gsm8k_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a") as f:
                    json.dump(
                        dict(
                            index=i + j,
                            gsm8k_data=gsm8k_data,
                            input_str=input_str,
                            output_str=output_str,
                        ),
                        f,
                        ensure_ascii=False,
                    )
                    f.write("\n")

        with open(gen_datas_jsonl) as f:
            gen_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for gen in gen_datas:
            result = dict(
                **gen,
                extract_true_num=extract_last_num(gen["gsm8k_data"]["response"]),
                extract_pred_num=extract_last_num(gen["output_str"]),
                is_correct=None,
            )
            if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
                result["is_correct"] = True
                correct_results.append(result)
            else:
                result["is_correct"] = False
                wrong_results.append(result)

        print(f'=======done {lang}============')
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
        print(result)
        with open(Path(lang_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(Path(lang_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        num_result = float(result.split('=')[-1])
        if lang != 'En_gsm8k':
            results[lang] = num_result
        else:
            gsm8k = num_result
    average = sum(results.values()) / len(results)
    print(average)
    import csv
    with open(Path(save_dir) / f"MSGM_evaluate_results_bs{batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Accuracy'])
        for key, value in results.items():
            writer.writerow([key, value])
        writer.writerow(['Average', average])
    


def gsm8k_batch_gen(
    lang_, gsm8k_questions, batch_llm
):
    lang = lang_ if lang_ != 'En_gsm8k' else 'English'
    prompt_no_input = (
      "Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request in {lang}. Please answer in {lang}.\n\n"
        "### Instruction:\n{query}\n\n### Response:"
    )

    input_str_list = [prompt_no_input.format(query=q) for q in gsm8k_questions]
    output_str_list = batch_llm(input_str_list)
    return input_str_list, output_str_list


def get_batch_llama(model: LlamaForCausalLM, tokenizer: LlamaTokenizer, args):

    if args.use_vllm:
        def batch_llama(input_strs):
            sampling_params = SamplingParams(temperature=0, max_tokens=600)
            generations = model.generate(input_strs, sampling_params)
            prompt_to_output = {
                g.prompt: g.outputs[0].text for g in generations
            }
            outputs = [prompt_to_output[prompt].strip() if prompt in prompt_to_output else "" for prompt in input_strs]
            return outputs
        
    else:
        @torch.inference_mode()
        def batch_llama(input_strs):
            input_ids_w_attnmask = tokenizer(
                input_strs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)
            output_ids = model.generate(
                input_ids=input_ids_w_attnmask.input_ids,
                attention_mask=input_ids_w_attnmask.attention_mask,
                generation_config=GenerationConfig(
                    max_new_tokens=600,
                    do_sample=False,
                    temperature=0.0,  # t=0.0 raise error if do_sample=True
                ),
            ).tolist()
            real_output_ids = [
                output_id[len(input_ids_w_attnmask.input_ids[i]) :] for i, output_id in enumerate(output_ids)
            ]
            output_strs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)
            return output_strs

    return batch_llama



def get_model(model_path: str, is_bf16: bool = False):
    print(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path, padding_side="left")
    print(tokenizer.pad_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print('new pad ', tokenizer.pad_token)
    print(tokenizer.bos_token)
    print(tokenizer.unk_token)
    print(tokenizer.eos_token)
    print(tokenizer.truncation_side)
    print(tokenizer.padding_side)

    if is_bf16:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_path,
        ).cuda()
    model.eval()
    print(model.dtype)

    return model, tokenizer


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    import fire


    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--streategy",
        type=str,
        help="which streategy to evaluate the model",
        required=True,
        choices=['Parallel','Cross']
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="save results dir",
        required=True
    )
    parser.add_argument(
        "--use_vllm",
        type=str2bool,
        help="use vllm or not",
        default=True
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        help="specific language to test",
        default = ''
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
    
    parser.add_argument(
        "--langs_vector_dir_path",
        type=str,
        help="the dir path of each language's vector",
        default = "None"
    )
    
    args = parser.parse_args()

    main(args=args)