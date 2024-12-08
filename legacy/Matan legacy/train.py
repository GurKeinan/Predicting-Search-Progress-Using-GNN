import time
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import train_utils
from torch.utils.data import DataLoader
from utils.types import Scores, Metrics
from utils.train_utils import TrainParams, score_function_mae, score_function_mse, plot
from utils.train_logger import TrainLogger
import os
import csv
import numpy as np
import pandas as pd
import torch.nn.functional as F
import src.pytorch_utils as ptu
import sklearn
from sklearn import preprocessing
from torch.autograd import Variable
import torchviz
import matplotlib.pyplot as plt
from datetime import datetime
import torchvision.transforms
import seaborn as sns


def get_metrics(best_eval_score: float, eval_score: float, train_loss: float) -> Metrics:
    """
    Example of metrics dictionary to be reported to tensorboard. Change it to your metrics
    :param best_eval_score:
    :param eval_score:
    :param train_loss:
    :return:
    """
    return {'Metrics/BestAccuracy': best_eval_score,
            'Metrics/LastAccuracy': eval_score,
            'Metrics/LastLoss': train_loss}


def train(model: nn.Module, train_loader: DataLoader, eval_loader: DataLoader, train_params: TrainParams,
          logger: TrainLogger, metric) -> Metrics:
    """
    Training procedure. Change each part if needed (optimizer, loss, etc.)
    :param model:
    :param train_loader:
    :param eval_loader:
    :param train_params:
    :param logger:
    :return:
    """
    metrics = train_utils.get_zeroed_metrics_dict()
    best_eval_score = None

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)

    # Create learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_params.lr_step_size,
                                                gamma=train_params.lr_gamma)
    loss_func = nn.MSELoss()
    for epoch in tqdm(range(train_params.num_epochs)):
        t = time.time()
        metrics = train_utils.get_zeroed_metrics_dict()

        for i, (x, y) in enumerate(train_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            y_hat = model(x)
            y_hat= y_hat.view(-1)

            loss = loss_func(y_hat, y)

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            metrics['total_norm'] += nn.utils.clip_grad_norm_(model.parameters(), train_params.grad_clip)
            metrics['count_norm'] += 1

            batch_score = score_function_mae(y_hat, y.data)
            metrics['train_score'] += batch_score

            metrics['train_loss'] += loss.item() * x.size(0)

        # Learning rate scheduler step
        scheduler.step()

        # Calculate metrics
        metrics['train_loss'] /= len(train_loader.dataset)

        metrics['train_score'] /= len(train_loader.dataset)

        norm = metrics['total_norm'] / metrics['count_norm']

        model.train(False)
        metrics['eval_score'], metrics['eval_loss'] = evaluate(model, eval_loader, epoch, train_params.print_plots, train_params.print_plot_every_epochs, best_eval_score,metric)
        model.train(True)

        epoch_time = time.time() - t
        logger.write_epoch_statistics(epoch, epoch_time, norm, metrics['train_loss'],
                                      metrics['train_score'], metrics['eval_score'])

        scalars = {'Accuracy/Train': metrics['train_score'],
                   'Accuracy/Validation': metrics['train_loss'],
                   'Loss/Train': metrics['eval_score'],
                   'Loss/Validation': metrics['eval_loss']}

        logger.report_scalars(scalars, epoch)

        if best_eval_score == None or metrics['eval_score'] > best_eval_score:
            best_eval_score = metrics['eval_score']
            if train_params.save_model:
                logger.save_model(model, epoch, optimizer)

    return get_metrics(best_eval_score, metrics['eval_score'], metrics['train_loss'])


@torch.no_grad()
def evaluate(model: nn.Module, dataloader: DataLoader, epoch: int, print_plots: bool, print_plot_every_epochs: int, best_eval_score: float, metric: str) -> Scores:
    """
    Evaluate a model without gradient calculation
    :param model: instance of a model
    :param dataloader: dataloader to evaluate the model on
    :return: tuple of (accuracy, loss) values
    """
    score = 0
    loss = 0
    out = None
    trues = None

    loss_func = nn.MSELoss()
    for i, (x, y) in enumerate(dataloader):
        if torch.cuda.is_available():
            x = x.cuda()
            y = y.cuda()

        y_hat = model(x)
        y_hat= y_hat.view(-1)
        loss += loss_func(y_hat, y)
        if (metric == "MSE"):
            score += score_function_mse(y_hat, y.data)
        else:
            score += score_function_mae(y_hat, y.data)
        if (i==0):
            out = y_hat
            trues = y
        else:
            out = torch.cat((out, y_hat),0)
            trues = torch.cat((trues , y),0)
    
    
    loss /= len(dataloader.dataset)
    score /= len(dataloader.dataset)
    if (best_eval_score == None or score > best_eval_score):
        plot(out , trues, epoch, print_plots, print_plot_every_epochs)

    return score, loss
