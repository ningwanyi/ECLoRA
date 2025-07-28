from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForQuestionAnswering, DefaultDataCollator, DataCollatorForTokenClassification
import random
from utils_qa import QuestionAnsweringTrainer
from datasets import load_dataset, load_metric, load_from_disk


def get_trainer(args, model, epoch_lr, client_dataset, validation_dataset, tokenizer, compute_metrics):
    # set up training arguments
    if args.model_name == "microsoft/deberta-v2-xxlarge" and args.dataset=="squad":
        training_args = TrainingArguments(
            output_dir=f"./buffer/outputs",
            save_strategy="no",
            learning_rate=epoch_lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=2,
            num_train_epochs=args.local_epochs,
            weight_decay=0.01,
            logging_dir='./buffer/logs',
            do_eval=False,
            evaluation_strategy="no",
            seed=random.randint(0, 10000)
        )
    elif args.model_name in ["meta-llama/Meta-Llama-3.1-8B"]:
        training_args = TrainingArguments(
            output_dir=f"./buffer/outputs",
            save_strategy="no",
            learning_rate=epoch_lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            num_train_epochs=args.local_epochs,
            weight_decay=0.01,
            logging_dir='./buffer/logs',
            do_eval=False,
            evaluation_strategy="no",
            seed=random.randint(0, 10000),
            bf16=True,
            label_names=['labels']
        )
    else:
        training_args = TrainingArguments(
            output_dir=f"./buffer/outputs",
            save_strategy="no",
            learning_rate=epoch_lr,
            lr_scheduler_type='constant',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=args.local_epochs,
            weight_decay=0.01,
            logging_dir='./buffer/logs',
            do_eval=False,
            evaluation_strategy="no",
            seed=random.randint(0, 10000),
        )

    # initialize Trainer
    if args.dataset=="squad":
        squad = load_from_disk("./datasets/squad")
        validation_examples = squad["validation"]
        trainer = QuestionAnsweringTrainer(
            model=model,
            args=training_args,
            train_dataset=client_dataset,
            eval_dataset=validation_dataset,
            eval_examples=validation_examples,
            tokenizer=tokenizer,
            data_collator=DefaultDataCollator(),
            compute_metrics=compute_metrics,
        )
    elif args.dataset=="conll2003":
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=client_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            data_collator = DataCollatorForTokenClassification(tokenizer),
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=client_dataset,
            eval_dataset=validation_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
    return trainer