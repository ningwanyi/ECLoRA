from datasets import load_dataset, concatenate_datasets
from torch.utils.data import ConcatDataset
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk
from utils_qa import prepare_train_dataset_qa, prepare_validation_dataset_qa

def prepare_squad_data(tokenizer):
    squad = load_from_disk("./datasets/squad")
    prepare_train_dataset = lambda exs: prepare_train_dataset_qa(exs, tokenizer, 384, 128)
    prepare_eval_dataset = lambda exs: prepare_validation_dataset_qa(exs, tokenizer, 384, 128)
    train_examples, validation_examples = squad["train"], squad["validation"]
    train_dataset = train_examples.map(
            prepare_train_dataset,
            batched=True,
            remove_columns=train_examples.column_names
        )
    validation_dataset = validation_examples.map(
            prepare_eval_dataset,
            batched=True,
            remove_columns=validation_examples.column_names
        )
    return train_dataset, validation_dataset, None


def prepare_conll2003_data(tokenizer):
    dataset = load_from_disk("./datasets/conll2003")
    label_list = dataset["train"].features["ner_tags"].feature.names
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, padding="max_length", max_length=128)
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)
    return tokenized_datasets['train'], tokenized_datasets['validation'], label_list


def prepare_datasets(args, tokenizer):
    if args.dataset=="squad":
        return prepare_squad_data(tokenizer)
    elif args.dataset=="conll2003":
        return prepare_conll2003_data(tokenizer)
    task_to_keys = {
        'cola': ('sentence', None),
        'mrpc': ('sentence1', 'sentence2'),
        'qnli': ('question', 'sentence'),
        'rte': ('sentence1', 'sentence2'),
    }
    sentence1_key, sentence2_key = task_to_keys[args.dataset]

    def preprocess_function(examples):
        # Tokenize the texts
        preprocess_args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*preprocess_args, padding='max_length', truncation=True, max_length=128)
        if 'label' in examples:
            # In all cases, rename the column to labels because the model will expect that.
            result['labels'] = examples['label']
        return result

    dataset = load_dataset('glue', args.dataset)
    
    processed_datasets = dataset.map(preprocess_function, batched=True)
    train_dataset = processed_datasets['train']
    if args.dataset == 'mnli':
        validation_datasets = {
            'validation_matched': processed_datasets['validation_matched'],
            'validation_mismatched': processed_datasets['validation_mismatched']
        }
    else:
        validation_datasets = {
            'validation': processed_datasets['validation']
        }
    validation_dataset = ConcatDataset([d for d in validation_datasets.values()])

    return train_dataset, validation_dataset, None


def split_non_cls_dataset(dataset, num_splits, alpha=10):
    total_size = len(dataset)
    # generate random proportions and ensure no split is empty
    while True:
        proportions = np.random.dirichlet(alpha * np.ones(num_splits))
        splits_lens = (proportions * total_size).astype(int)
        splits_lens[-1] = total_size - sum(splits_lens[:-1])
        # check for 0 values and ensure total length is correct
        if 0 not in splits_lens and sum(splits_lens) == total_size:
            break
        print("Resplitting dataset due to 0 length or incorrect total size...")
    splits = []
    # generate random indices and split dataset
    indices = np.arange(total_size)
    np.random.shuffle(indices)
    start_index = 0
    for i in range(num_splits):
        split_indices = indices[start_index:(start_index + splits_lens[i])]
        split_dataset = dataset.select(split_indices)
        splits.append(split_dataset)
        start_index += splits_lens[i]
    return splits


def split_dataset_iid(dataset, num_splits):
    """ 
    Split dataset into IID (Independent and Identically Distributed) subsets.

    Args:
        dataset: A dataset (Hugging Face Dataset format).
        num_splits: The number of subsets to split into.

    Returns:
        A list of datasets, each representing a subset of the original dataset.
    """
    total_size = len(dataset)
    indices = np.arange(total_size)
    np.random.shuffle(indices)  # Shuffle data indices

    # Calculate the size of each subset
    split_sizes = [total_size // num_splits] * num_splits
    for i in range(total_size % num_splits):  # Handle remainder
        split_sizes[i] += 1

    # Split dataset
    subsets = []
    start_idx = 0
    for size in split_sizes:
        split_indices = indices[start_idx: start_idx + size]
        subsets.append(dataset.select(split_indices))
        start_idx += size

    return subsets


def split_dataset_dirichlet(dataset, num_splits, alpha, task_name):
    if task_name in ["squad", "conll2003"]:
        return split_non_cls_dataset(dataset, num_splits, alpha)

    # get labels according to task name
    if task_name in ['qnli', 'rte', 'cola', 'mrpc']:
        labels = np.array(dataset['label'])
    else:
        raise ValueError("Unsupported GLUE task.")

    num_classes = 2
    # Initialize subset list
    subsets = [[] for _ in range(num_splits)]
    for c in range(num_classes):
        # Get all samples' indices belonging to the current class
        idxs = np.where(labels == c)[0]
        
        # generate random proportions for Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_splits))

        # ensure all samples of each class are allocated
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]

        # allocate sample indices to different subsets according to the allocation probabilities
        splits = np.split(idxs, proportions)

        # Check for empty subsets and reallocate if necessary
        splits = [list(split) for split in splits]  # Convert to list
        for i in range(num_splits):
            if len(splits[i]) == 0:
                non_empty_splits = [split for split in splits if len(split) > 1]
                selected_split = np.random.choice(non_empty_splits)
                selected_idx = np.random.choice(selected_split)
                splits[i].append(selected_idx)
                selected_split.remove(selected_idx)
        
        for i in range(num_splits):
            subsets[i].extend(splits[i])

    # Create new subsets according to the generated indices
    subsets = [dataset.select(indices) for indices in subsets]
    
    return subsets

