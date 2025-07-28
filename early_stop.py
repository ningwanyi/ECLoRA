import numpy as np
import torch
import os

TASK_METRICS = {
        "mrpc":"eval_accuracy", 
        "cola":"eval_matthews_correlation", 
        "rte":"eval_accuracy", 
        "qnli":"eval_accuracy",
        "squad":"eval_exact_match",
        "conll2003":"eval_f1",
    }

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, save_path=None, verbose=False, delta=0):
        """
        Args:
            save_path (str): Path for saving the best model.
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = 0
        self.delta = delta

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.save_path is None:
            return
        if self.verbose:
            print(f'Validation score increased ({self.val_score_max:.6f} --> {score:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)	# This will save the parameters of the best model so far
        self.val_score_max = score


