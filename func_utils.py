import argparse
import logging
import os


def adjust_lr(epoch, init_lr, warmup_epochs, total_epochs):
    if epoch < warmup_epochs:
        # Warmup phase: linearly increase learning rate
        lr = init_lr * (epoch + 1) / warmup_epochs
    else:
        # linear decay phase: decrease learning rate linearly
        lr = init_lr * (1 - (epoch - warmup_epochs) / (total_epochs - warmup_epochs))
    return lr


def setup_logging(args):
    if args.logging:
        log_file = os.path.join("./results", args.model_name, args.output_name, "logging.log")
    else:
        log_file = "./results/test.log"
    # If log directory does not exist, create it
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="roberta-base")
    parser.add_argument("--ranks", type=str, default="4,8,16,32,64")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--sample_rate", type=float, default=0.1)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=2)
    parser.add_argument("--dirichlet_alpha", type=float, default=1)
    parser.add_argument("--iid", action='store_true')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aggregation", type=str, default="homo")
    parser.add_argument("--dataset", type=str, default="mrpc")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--compensation_beta", type=float, default=1)
    parser.add_argument("--logging", action='store_true', help="whether to record logs")
    parser.add_argument("--resume", action='store_true', help="resume the saved ckpts")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--early_stop", action='store_true')
    parser.add_argument("--start_early_epoch", type=int, default=50)
    parser.add_argument("--target_modules", type=str, default="q,k,v,o")
    parser.add_argument("--dynamic_ranks", action='store_true')
    parser.add_argument("--output_name", type=str, default="default")
    parser.add_argument("--flipping_rate", type=float, default=0)
    args = parser.parse_args()
    if args.output_name == "default":
        args.output_name = f"{args.dataset}_agg_{args.aggregation}_seed_{args.seed}"
    return args