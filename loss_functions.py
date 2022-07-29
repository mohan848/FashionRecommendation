# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:31:30 2022

@author: mohan
"""

import torch.nn as nn
# custom loss function for multi-head multi-category classification
def loss_fn(outputs, targets):
    o1, o2, o3 = outputs
    t1, t2, t3 = targets
    l1 = nn.CrossEntropyLoss()(o1, t1)
    l2 = nn.CrossEntropyLoss()(o2, t2)
    l3 = nn.CrossEntropyLoss()(o3, t3)
    return (l1 + l2 + l3) / 3