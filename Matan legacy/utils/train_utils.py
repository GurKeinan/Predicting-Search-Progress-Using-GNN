from typing import Dict
from torch import Tensor
from omegaconf import DictConfig
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import os
import csv
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns


def score_function_mae(out , trues):
    answer = abs(out - trues)
    sum = answer.sum().item()
    sum = -sum
    return sum

def score_function_mse(out , trues):
    answer = (out - trues) * (out - trues)
    sum = answer.sum().item()
    sum = -sum
    return sum

def plot(out , trues, epoch, print_plots, print_plot_every_epochs):
    if (print_plots == False):
        return
    """if (epoch % print_plot_every_epochs != 0):
        return"""
    now = datetime.now()
    temp_out = out.clone().detach() 
    temp_trues =trues.clone().detach()
    temp_out = temp_out/5
    for i in range(len(trues)):
        out_i = int(temp_out[i])*5/5
        if out_i == 20:
            out_i = 19
        temp_out[i] = out_i
        trues_i = int(temp_trues[i] / 5)
        if trues_i == 20:
            trues_i = 19
        temp_trues[i] = trues_i
    #tensor_trues = torch.from_numpy(temp_trues)
    temp_y = abs(temp_trues - temp_out)
    confusion = torch.zeros(20, 20)
    for i in range(len(out)):
        out_i = int(temp_out[i])
        trues_i = int(temp_y[i])
        confusion[out_i][trues_i] += 1

    for i in range(20):
        confusion[i] = confusion[i] / confusion[i].sum()


    confusion = torch.rot90(confusion , 1, [0, 1])
    zeros = torch.zeros(1,20)
    confusion = torch.cat((zeros, confusion),0)
    confusion = torch.cat((confusion, zeros),0)
    # Normalize by dividing every row by its sum

    confusion[21] = confusion[21] / confusion[21].sum()
    confusion[0] = confusion[0] / confusion[0].sum()
 

    fig = plt.figure(figsize=(17,14))
    # ax= plt.subplot()
    ax = fig.add_subplot(111)
    sns.heatmap(confusion, annot=True, ax = ax, linewidths=2.5, center=0, fmt="0.2f"); #annot=True to annotate cells

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('diff')
    ax.set_title('Confusion Matrix')

    plt.savefig('output/confusion matrix_dropout_deeper'+str(epoch)+' length ='+str(len(out))+" "+str(now)+'.png')
    fig.show


def compute_score_with_logits(logits: Tensor, labels: Tensor) -> Tensor:
    """
    Calculate multiclass accuracy with logits (one class also works)
    :param logits: tensor with logits from the model
    :param labels: tensor holds all the labels
    :return: score for each sample
    """
    logits = torch.max(logits, 1)[1].data  # argmax

    logits_one_hots = torch.zeros(*labels.size())
    if torch.cuda.is_available():
        logits_one_hots = logits_one_hots.cuda()
    logits_one_hots.scatter_(1, logits.view(-1, 1), 1)

    scores = (logits_one_hots * labels)

    return scores


def get_zeroed_metrics_dict() -> Dict:
    """
    :return: dictionary to store all relevant metrics for training
    """
    return {'train_loss': 0, 'train_score': 0, 'total_norm': 0, 'count_norm': 0}


class TrainParams:
    """
    This class holds all train parameters.
    Add here variable in case configuration file is modified.
    """
    num_epochs: int
    lr: float
    lr_decay: float
    lr_gamma: float
    lr_step_size: int
    grad_clip: float
    save_model: bool

    def __init__(self, **kwargs):
        """
        :param kwargs: configuration file
        """
        self.num_epochs = kwargs['num_epochs']

        self.lr = kwargs['lr']['lr_value']
        #self.lr_decay = kwargs['lr']['lr_decay']
        self.lr_gamma = kwargs['lr']['lr_gamma']
        self.lr_step_size = kwargs['lr']['lr_step_size']

        self.grad_clip = kwargs['grad_clip']
        self.save_model = kwargs['save_model']
        self.print_plots = kwargs['print_plots']
        self.print_plot_every_epochs = kwargs['print_plot_every_epochs']


def get_train_params(cfg: DictConfig) -> TrainParams:
    """
    Return a TrainParams instance for a given configuration file
    :param cfg: configuration file
    :return:
    """
    return TrainParams(**cfg['train'])
