import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ['WANDB_DISABLED'] = 'true'

import logging
from func_utils import adjust_lr, setup_args, setup_logging

from run_fedit import main_homo
from run_hetelora import main_zeropad
from run_flexlora import main_svd
from run_eclora import main_rsvd
from run_flora import main_flora
import random
import numpy as np
import torch


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):
    seed_everything(args.seed)
    if args.aggregation=="fedit":
        main_homo(args)
    elif args.aggregation=="hetelora":
        main_zeropad(args)
    elif args.aggregation=="flexlora":
        main_svd(args)
    elif args.aggregation=="eclora":
        main_rsvd(args)
    elif args.aggregation=="flora":
        main_flora(args)
    else:
        raise NotImplementedError
    

if __name__ == "__main__":
    args = setup_args()
    setup_logging(args)
    logging.info(args)
    main(args)