from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoModelForTokenClassification
import torch

def load_model_and_tokenizer(model_name, dataset_name, device="cuda"):
    if dataset_name in ["rte", "mrpc", "cola", "qnli"]:
        if model_name in ["meta-llama/Meta-Llama-3.1-8B"]:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map=device)
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif dataset_name in ["squad"]:
        if model_name in ["meta-llama/Meta-Llama-3.1-8B"]:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name, torch_dtype=torch.bfloat16).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(model_name).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name)
    elif dataset_name in ["conll2003"]:
        if model_name in ["meta-llama/Meta-Llama-3.1-8B"]:
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9, torch_dtype=torch.bfloat16).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, use_fast=False)
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=9).cuda()
            tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    return model, tokenizer

