import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset,load_metric
import numpy as np
import copy
import random
from heterogenous_lora import reload_lora_into_model
import time
import logging
import os
from data_utils import prepare_datasets, split_dataset_dirichlet, split_dataset_iid
from transformers import EvalPrediction
from func_utils import adjust_lr, setup_args
from model_utils import load_model_and_tokenizer
from training_utils import get_trainer
from early_stop import EarlyStopping, TASK_METRICS
import evaluate
import time

def aggregate_svd(target_modules, local_param_list, num_sample_list, rank_list, global_param, model, model_name, dataset_name):
    whole_param = {}
    for n,p in global_param.items():
        if "lora_A" in n:
            whole_p = torch.zeros_like(global_param[n.replace("lora_A","lora_B")].data @ p.data)
            # Aggregate client parameters
            for local_param, num_sample, rank in zip(local_param_list, num_sample_list, rank_list):
                whole_p += local_param[n.replace("lora_A","lora_B")].data @ local_param[n].data * num_sample / sum(num_sample_list) 
            whole_param[n.replace("lora_A.client","base_layer")] = whole_p.cpu()
        elif "lora_B" in n:
            continue
        else:
            whole_p = torch.zeros_like(global_param[n])
            # Aggregate client parameters
            for local_param, num_sample, rank in zip(local_param_list, num_sample_list, rank_list):
                whole_p += local_param[n].data * num_sample / sum(num_sample_list) 
            whole_param[n] = whole_p.cpu()
    # Update global model parameters
    model = reload_lora_into_model(target_modules, model, max(rank_list), model_name, dataset_name, "client")
    model.set_adapter("client")
    # logging.info(whole_param.keys())
    for n, p in model.named_parameters():
        # logging.info(n)
        if n in whole_param:
            if "base_layer" in n:
                p.data += whole_param[n].clone().cuda()
            elif any([x in n for x in ["classifier", "qa_outputs", "pooler","score"]]):
                p.data = whole_param[n].clone().cuda()
        else:
            continue
    return model, whole_param

def load_global_param_svd(target_modules, model, global_param, client_rank, model_name, dataset_name):
    model = reload_lora_into_model(target_modules, model, client_rank, model_name, dataset_name, "client")
    model.set_adapter("client")
    for n,p in model.named_parameters():
        if p.requires_grad:
            if "lora_A" in n:
                p.data = global_param[n][:client_rank,:].clone()
            elif "lora_B" in n:
                p.data = global_param[n][:,:client_rank].clone()
            else:
                p.data = global_param[n].clone()
    return model


def main_svd(args):
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load BERT tokenizer and model
    involved_ranks = [int(rank) for rank in args.ranks.split(",")]
    
    model,tokenizer = load_model_and_tokenizer(args.model_name, args.dataset)
    model = reload_lora_into_model(args.target_modules, model, max(involved_ranks), args.model_name, args.dataset, "client")
    model.set_adapter("client")
    global_param = copy.deepcopy({n: p.data for n, p in model.named_parameters() if p.requires_grad})

    # Load dataset
    train_dataset, validation_dataset, label_list = prepare_datasets(args, tokenizer)

    # split trainset into client datasets
    if args.iid:
        train_splits = split_dataset_iid(train_dataset, args.num_clients)
    else:
        train_splits = split_dataset_dirichlet(train_dataset, args.num_clients, args.dirichlet_alpha, args.dataset)

    # Randomly select flipping_rate% of clients as malicious clients for label flipping attack (flip positive labels to negative and vice versa)
    if args.flipping_rate > 0:
        malicious_clients = []
        for i,client_dataset in enumerate(train_splits):
            if random.random() < args.flipping_rate:
                malicious_clients.append(i)
                train_splits[i] = client_dataset.map(lambda x: {"label": 1 - x["label"]})
        logging.info(f"Malicious clients: {malicious_clients}")

    # Set evaluation metrics
    if args.dataset == 'squad':
        metric = evaluate.load('evaluate/metrics/squad', trust_remote_code=True)
    elif args.dataset == 'conll2003':
        metric = evaluate.load('evaluate/metrics/seqeval', trust_remote_code=True)
    else:
        metric = evaluate.load('evaluate/metrics/glue', args.dataset, trust_remote_code=True)
    def compute_metrics(p:EvalPrediction):
        if args.dataset == 'squad':
            return metric.compute(predictions=p.predictions, references=p.label_ids)
        elif args.dataset == 'conll2003':
            predictions, labels = p
            predictions = np.argmax(predictions, axis=-1)
            labels = labels.reshape(predictions.shape)
            true_predictions = [
                [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = metric.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        else:
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            result['default'] = result.get('f1', result.get('accuracy', 0.))
            return result

    # Set client ranks
    client_ranks_list = [random.choice(involved_ranks) for _ in range(args.num_clients)]

    # Start training
    early_stopping = EarlyStopping(patience=args.patience)
    final_results = {}
    random.seed(args.seed)

    # resume training
    result_dir = f"./results/{args.model_name}/{args.output_name}"
    ckpt_file = os.path.join(result_dir, "global_param.pth")
    if args.resume and os.path.exists(ckpt_file):
        global_param = torch.load(ckpt_file)
        with open(f"{result_dir}/checkpoint_state.json", "r") as f:
            checkpoint_state = json.load(f)
        start_epoch = checkpoint_state["epoch"]+1
        logging.info(f"Resume training from epoch {start_epoch}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        epoch_lr = adjust_lr(epoch, args.learning_rate, args.warmup_epochs, args.num_epochs)
        selected_client_ids = random.sample(range(args.num_clients), int(args.num_clients * args.sample_rate))
        local_param_list, num_sample_list, rank_list = [], [], []
        epoch_ranks_list = [random.choice(involved_ranks) for _ in range(args.num_clients)] if args.dynamic_ranks else client_ranks_list

        for client_id in selected_client_ids:
            client_dataset = train_splits[client_id]
            client_rank = epoch_ranks_list[client_id]
            
            # Reinitialize model each time and load global_param into the model
            model, tokenizer = load_model_and_tokenizer(args.model_name, args.dataset, "cpu")  # Without this line, RecursionError occurs
            model = load_global_param_svd(args.target_modules, model, global_param, client_rank, args.model_name, args.dataset)
            
            # Initialize trainer
            trainer = get_trainer(args, model, epoch_lr, client_dataset, validation_dataset, tokenizer, compute_metrics)

            # Train model
            client_train_result = trainer.train()
            logging.info(f"Finish training client {client_id} with rank {client_ranks_list[client_id]} and {len(client_dataset)} samples: {client_train_result}")

            # Get client model parameters
            local_param = copy.deepcopy({n: p.data for n, p in model.named_parameters() if p.requires_grad})
            local_param_list.append(local_param)
            num_sample_list.append(len(client_dataset))
            rank_list.append(client_rank)

        # Aggregate client model parameters
        model, whole_param = aggregate_svd(args.target_modules, local_param_list, num_sample_list, rank_list, global_param, model, args.model_name, args.dataset)

        # Evaluate aggregated model
        trainer = get_trainer(args, model, epoch_lr, client_dataset, validation_dataset, tokenizer, compute_metrics)
        eval_result = trainer.evaluate()
        logging.info(f"Epoch {epoch} | Evaluation result: {eval_result}")

        final_results[epoch] = eval_result

        # Restore to base_model
        start_time = time.time()
        max_rank = max(involved_ranks)
        for n,p in model.named_parameters():
            if n in whole_param and "base_layer" in n:
                p.data -= whole_param[n].cuda()
                u,s,v = torch.svd(whole_param[n])
                U = u[:,:max_rank]
                S = s[:max_rank]
                V = v.T[:max_rank,:]
                global_param[n.replace("base_layer", "lora_A.client")] = V.clone().cuda()
                global_param[n.replace("base_layer", "lora_B.client")] = (U @ torch.diag(S)).clone().cuda()
            elif n in whole_param and "classifier" in n:
                global_param[n] = whole_param[n].clone().cuda()
            elif n in whole_param and "pooler" in n:
                global_param[n] = whole_param[n].clone().cuda()
            elif n in whole_param and "qa_outputs" in n:
                global_param[n] = whole_param[n].clone().cuda()
            elif n in whole_param and "score" in n:        # for llama
                global_param[n] = whole_param[n].clone().cuda()
            else:
                continue
        logging.info(f"SVD Time cost: {time.time()-start_time}")
        del whole_param

        # check early stopping
        if args.early_stop and epoch > args.start_early_epoch:
            metric_name = TASK_METRICS[args.dataset]
            early_stopping(eval_result[metric_name], model)
            if early_stopping.early_stop:
                logging.info("Early stopping")
                break
        
        if args.logging:
            # Save model checkpoint
            checkpoint_state = {
                "epoch": epoch,
                "eval_result": eval_result
            }
            # Save global_param and previous_error_dict
            torch.save(global_param, f"{result_dir}/global_param.pth")
            # Save checkpoint_state to json
            with open(f"{result_dir}/checkpoint_state.json", "w") as f:
                json.dump(checkpoint_state, f)

    # Save final_results to json
    import json
    with open(f"{result_dir}/final_results.json", "w") as f:
        json.dump(final_results, f)