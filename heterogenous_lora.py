from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import random
import numpy as np
from scipy.stats import norm
import torch


def reload_lora_into_model(target_modules, model, client_rank, model_name, dataset_name, adapter_name='default', dropout=0):
    torch.manual_seed(42)   # fix the seed to make initialization deterministic
    lora_config = LoraConfig(
        r=client_rank,
        lora_alpha=client_rank,
        lora_dropout=dropout,
        target_modules=get_target_diff_lora_modules(model_name, target_modules),
        task_type=get_task_lora_type(dataset_name),
    )
    if not isinstance(model, PeftModel):
        model = get_peft_model(
            model=model, 
            peft_config=lora_config, 
            adapter_name=adapter_name,
        )
    else:
        if adapter_name in model.peft_config:
            model.delete_adapter(adapter_name)
        model.add_adapter(adapter_name, lora_config)
    return model


def get_target_lora_modules(model_name):
    model_target_modules = {
        "roberta-base": ['attention.self.query','attention.self.key','attention.self.value',\
                        'attention.output.dense'],
        "roberta-large": ['attention.self.query','attention.self.key','attention.self.value',\
                        'attention.output.dense'],
        "microsoft/deberta-v2-xxlarge": ['attention.self.query_proj','attention.self.key_proj','attention.self.value_proj',\
                                         'attention.output.dense'],
        "meta-llama/Meta-Llama-3.1-8B": ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'],
    }
    return model_target_modules[model_name]

def get_task_lora_type(dataset_name):
    dataset_task_type = {
        "rte": TaskType.SEQ_CLS,
        "mrpc": TaskType.SEQ_CLS,
        "cola": TaskType.SEQ_CLS,
        "qnli": TaskType.SEQ_CLS,
        "squad": TaskType.QUESTION_ANS,
        "conll2003": TaskType.TOKEN_CLS,
    }
    return dataset_task_type[dataset_name]


def get_target_diff_lora_modules(model_name, target_modules):
    if target_modules=="q,k,v,o":
        model_target_modules = {
            "roberta-base": ['attention.self.query','attention.self.key','attention.self.value',\
                            'attention.output.dense'],
            "roberta-large": ['attention.self.query','attention.self.key','attention.self.value',\
                            'attention.output.dense'],
            "microsoft/deberta-v2-xxlarge": ['attention.self.query_proj','attention.self.key_proj','attention.self.value_proj',\
                                            'attention.output.dense'],
            "meta-llama/Meta-Llama-3.1-8B": ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj'],
        }
    elif target_modules=="q,k,v":
        model_target_modules = {
            "roberta-base": ['attention.self.query','attention.self.key','attention.self.value'],
            "roberta-large": ['attention.self.query','attention.self.key','attention.self.value'],
            "microsoft/deberta-v2-xxlarge": ['attention.self.query_proj','attention.self.key_proj','attention.self.value_proj'],
            "meta-llama/Meta-Llama-3.1-8B": ['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'],
        }
    elif target_modules=="q,v":
        model_target_modules = {
            "roberta-base": ['attention.self.query','attention.self.value'],
            "roberta-large": ['attention.self.query','attention.self.value'],
            "microsoft/deberta-v2-xxlarge": ['attention.self.query_proj','attention.self.value_proj'],
            "meta-llama/Meta-Llama-3.1-8B": ['self_attn.q_proj','self_attn.v_proj'],
        }
    elif target_modules=="q":
        model_target_modules = {
            "roberta-base": ['attention.self.query'],
            "roberta-large": ['attention.self.query'],
            "microsoft/deberta-v2-xxlarge": ['attention.self.query_proj'],
            "meta-llama/Meta-Llama-3.1-8B": ['self_attn.q_proj'],
        }
    elif target_modules=="all-linear":
        model_target_modules = {
            "roberta-large": ['attention.self.query','attention.self.key','attention.self.value',\
                            'attention.output.dense','output.dense','intermidiate.dense'],
        }
    else:
        raise ValueError("Invalid target_modules")
    return model_target_modules[model_name]