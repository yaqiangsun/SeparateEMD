from logging import log
import math

import torch
# from torch.autograd.grad_mode import F
import torch.nn as nn

class CrossEntropyLoss(nn.Module):

    def __init__(self
                 ):
        super(CrossEntropyLoss, self).__init__()
        self.nll_loss = torch.nn.NLLLoss()

    def forward(self, logits, target, *args, **kwargs):
        reduction="mean"

        prob = torch.softmax(logits,dim=1)

        logits = torch.log(prob)
        loss = self.nll_loss(logits,target)
        
        if reduction == "mean": return loss.mean()
        else: return loss.sum()