import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_var


def mixmatch(batch, alpha):
    input, labels_x, guess_labels_x = batch
    
    l = np.random.beta(alpha, alpha)        
    l = max(l, 1-l)

    idx = torch.randperm(input.size(0))
    
    input_a, input_b = input, input[idx]
    target_a, target_b = labels_x, labels_x[idx]
    mixed_input = l * input_a + (1 - l) * input_b
    mixed_target = l * target_a + (1 - l) * target_b
    
    guess_target_a, guess_target_b = guess_labels_x, guess_labels_x[idx]
    mixed_guess_target = l * guess_target_a + (1-l) * guess_target_b
    return mixed_input, mixed_target, mixed_guess_target


## boost labeled data
def boost_labeled_data(batch_data, model1, model2, num_classes, T=0.5):
    inputs1, inputs2, target, w_x = batch_data
    inputs1_var, inputs2_var = to_var(inputs1, requires_grad=False), to_var(inputs2, requires_grad=False)
    batch_size = inputs1_var.size()[0]
    labels_x = torch.zeros(batch_size, num_classes).scatter_(1, target.view(-1,1), 1)
    
    outputs1 = model1(inputs1_var)
    outputs2 = model1(inputs2_var)

    px = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1)) / 2
    px = w_x.float().view(-1, 1).cuda()*labels_x.cuda() + (1-w_x).float().view(-1, 1).cuda()*px
    ptx = px**(1/T) # temparature sharpening

    target = ptx / ptx.sum(dim=1, keepdim=True) # normalize

    target = target.detach()


    outputs_21 = model2(inputs1_var)
    outputs_22 = model2(inputs2_var)
    guess_x = (torch.softmax(outputs1, dim=1) + torch.softmax(outputs2, dim=1) + torch.softmax(outputs_21, dim=1) + torch.softmax(outputs_22, dim=1)) / 4


    guess_x = guess_x**(1/T)
    guess_t_x = guess_x/guess_x.sum(dim=1, keepdim=True)
    guess_t_x = guess_t_x.detach()

    
    return inputs1, inputs2, target, guess_t_x


## boost unlabeled data
def boost_unlabeled_data(batch_data, model1, model2, num_classes, T=0.5):
    inputs1, inputs2, targets_u, w_u = batch_data
    unlabeled_batch_size = inputs1.size()[0]
    labels_u = torch.zeros(unlabeled_batch_size, num_classes).scatter_(1, targets_u.view(-1,1), 1)
    
    inputs1_var, inputs2_var = to_var(inputs1, requires_grad=False), to_var(inputs2, requires_grad=False)
    
    outputs_u11 = model1(inputs1_var)
    outputs_u12 = model1(inputs2_var)
    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1)) / 2
    pu = w_u.float().view(-1, 1).cuda()*labels_u.cuda() + (1-w_u).float().view(-1, 1).cuda()*pu
    ptu = pu**(1/T) # temparature sharpening
    
    targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
    targets_u = targets_u.detach()
    
    ### guess unlabled data
    outputs_u21 = model2(inputs1_var)
    outputs_u22 = model2(inputs2_var)
    guess_u = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) + torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1)) / 4
    guess_u = guess_u**(1/T)
    guess_t_u = guess_u/guess_u.sum(dim=1, keepdim=True)
    guess_t_u = guess_t_u.detach()
    return inputs1, inputs2, targets_u, guess_t_u


## process training data
def augment_and_refinement_training_data(labeled_data, unlabeled_data, model1, model2, num_classes, T=0.5):
    with torch.no_grad():
        inputs1_x, inputs2_x, targets_x, guess_t_x = boost_labeled_data(labeled_data, model1, model2, num_classes, T)
        inputs1_u, inputs2_u, targets_u, guess_t_u = boost_unlabeled_data(unlabeled_data, model1, model2, num_classes, T)
        inputs = torch.cat([inputs1_x, inputs2_x, inputs1_u, inputs2_u], dim=0)
        target = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
        guess_target = torch.cat([guess_t_x, guess_t_x, guess_t_u, guess_t_u], dim=0)
        
    return inputs, target, guess_target


def convert_train_data(labeled_data, unlabeled_data, model1, model2, num_classes, alpha):
    
    batch = augment_and_refinement_training_data(labeled_data, unlabeled_data, model1, model2, num_classes)
    ## do mixmatch
    batch = mixmatch(batch, alpha)
    return batch 


## process meta data
def augment_and_refinement_meta_data(batch, model1, model2, num_classes, T = 0.5):
    with torch.no_grad():
        inputs1, inputs2, target, guess_t_x = boost_labeled_data(batch, model1, model2, num_classes, T)
        inputs = torch.cat([inputs1, inputs2], dim=0)
        target = torch.cat([target, target], dim=0)
        guess_target = torch.cat([guess_t_x, guess_t_x], dim=0)
        
    return inputs, target, guess_target


def convert_meta_data(batch, model1, model2, num_classes, alpha):
    batch = augment_and_refinement_meta_data(batch, model1, model2, num_classes)
    batch = mixmatch(batch, alpha)
    return batch 


