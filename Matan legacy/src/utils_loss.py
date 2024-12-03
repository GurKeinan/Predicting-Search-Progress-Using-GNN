import torch
import torch.nn as nn


def loss_decision_func(self, device, batch, prints=True):
    input , y = batch
    out = self.model.forward(input.to(device), prints)
    print('out', out.shape) if prints else None
    print('y', y.shape) if prints else None
    loss = self.criterion(out.to(device), y.to(device).long())

    return loss, out, y

"""
def metrics_batch(target, output):
    # obtain output class
    pred = output.argmax(dim=1, keepdim=True)
    
    # compare output class with target class
    corrects=pred.eq(target.view_as(pred)).sum().item()
    return corrects
"""

def out_decision_func(self, out):
    pred = out.argmax(dim=1, keepdim=True)
    return pred