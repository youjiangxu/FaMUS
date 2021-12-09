# -*- coding: utf-8 -*-

import argparse
import os
import time
import random
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable


import dataloader_red_mini_imagenet as mini_imagenet_dataloader
from build_models import build_training, build_model, build_grad_models
from utils import AverageMeter, accuracy, resume_model, save_model, to_var
from grad_operator_layer import _compute_gated_grad
from data_utils import mixmatch, convert_train_data, convert_meta_data


parser = argparse.ArgumentParser(description='PyTorch WideResNet Training')
parser.add_argument('--num_meta', type=int, default=1000)
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
                    help='initial learning rate')
parser.add_argument('--meta_lr', default=1e-3, type=float,
                    help='initial learning rate for vnet')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, type=bool, help='nesterov momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--widen-factor', default=10, type=int,
                    help='widen factor (default: 10)')
parser.add_argument('--layers', default=28, type=int,
                    help='total number of layers (default: 28)')
parser.add_argument('--name', default='F-WideResNet-28-10-diff-valid', type=str,
                    help='name of experiment')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--step1', type=int, default=200)
parser.add_argument('--step2', type=int, default=250)
parser.add_argument('--warmup', default=True, action='store_false', help='to remove warmup for models; default: True')
parser.add_argument('--warmup_epochs', type=int, default=5)
parser.add_argument('--p_threshold', default=0.5, type=float, help='clean probability threshold')
parser.add_argument('--meta_threshold', default=0.5, type=float, help='clean probability threshold for meta set')
parser.add_argument('--alpha', default=4, type=float, help='parameter for Beta')
parser.add_argument('--network', default="PreActResNet18", type=str, help='specify model architecture/network')
parser.add_argument('--regularization', default=True, action='store_false', help='remove regularization in total loss; default: True')
parser.add_argument('--lambda_u', default=50, type=float, help='weight for unsupervised loss')
parser.add_argument('--rampup_length', default=16, type=int)

## add famus options
parser.add_argument('--top_k', type=int, default=15)## 6+7+2
parser.add_argument('--num_act', type=float, default=4)## 6+7+2
parser.add_argument('--mse_factor', type=float, default=1e-1)
parser.add_argument('--act_factor', type=float, default=1e-1)
parser.add_argument('--go_lr', type=float, default=1e-1)

## configure dataloader
parser.add_argument('--split', type=str, default='red_noise_nl_0.4')
## configure output
parser.add_argument('--out_root', type=str, default='./output')

parser.set_defaults(augment=True)


best_acc = 0

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))

def cross_entropy(input, target, reduction=False, use_pytorch=False):
    if not use_pytorch:
        return F.cross_entropy(input, target, reduce=reduction)
    if reduction:
        return -torch.mean(torch.sum(F.log_softmax(input, dim=1) * target.float(), dim=1))
    else:
        return -torch.sum(F.log_softmax(input, dim=1) * target.float(), dim=1)


def get_lambda_u(e_i, rampup_length=16):
    current = np.clip(e_i*1.0/rampup_length, 0.0, 1.0)
    return args.lambda_u*float(current)
    
def mse(logit, target, reduction='none' ):
    
    probs = torch.softmax(logit, dim=1)
    if reduction=='none':
        return torch.mean((probs-target)**2, dim=1)
    else:
        return torch.mean((probs-target)**2)

    
def regularization(logits):
    prior = torch.ones(args.num_classes)/args.num_classes
    prior = prior.cuda()        
    pred_mean = torch.softmax(logits, dim=1).mean(0)
    penalty = torch.sum(prior*torch.log(prior/pred_mean))
    return penalty

   

def norm_weight(weights):
    norm = torch.sum(weights)
    if norm != 0:
        normed_weights = weights / norm
    else:
        normed_weights = weights
    return normed_weights
    
    
def adjust_learning_rate(optimizer, e_i):
    lr = args.lr * ((0.1 ** int(e_i >= args.step1)) * (0.1 ** int(e_i >= args.step2)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
        
def warmup(model, optimizer, data_loader, test_loader, criterion):
    for w_i in range(args.warmup_epochs):
        for iters, batch in enumerate(data_loader):

            model.train()
            input, target, _ = batch
            input_var = to_var(input, requires_grad=False)
            target_var = to_var(target, requires_grad=False)
            optimizer.zero_grad()
            outputs = model(input_var)
            loss = F.cross_entropy(outputs, target_var)      
            loss.backward()  
            optimizer.step() 
        print('Warmup Epoch {} '.format(w_i)) 
        test(model, test_loader, criterion)


def meta_step(model, optimizer_a, vnet, optimizer_c, train_batch, valid_batch, e_i, grad_models, grad_optimizers):
    # adjust learning rate
    adjust_learning_rate(optimizer_a, e_i)
    # parse input
    input_train, target_train, guess_target_train = train_batch
    input_var = to_var(input_train, requires_grad=False)
    target_var = to_var(target_train, requires_grad=False)
    guess_target_var = to_var(guess_target_train, requires_grad=False)


    # build meta model
    meta_model = build_model(args)
    meta_model.load_state_dict(model.state_dict())

    # virtual forward
    y_f_hat = meta_model(input_var)

    cost = cross_entropy(y_f_hat, target_var, reduction=False, use_pytorch=True)

    cost_v = torch.reshape(cost, (len(cost), 1))


    mse_loss = mse(y_f_hat, guess_target_var, reduction='none')

    v_lambda = vnet(cost_v.data)

    batch_size = v_lambda.size()[0]
    
    v_lambda = v_lambda.view(-1)
    v_lambda = norm_weight(v_lambda)

    cur_lambda_u = get_lambda_u(e_i, rampup_length=args.rampup_length)
    ## labeled + unlabeled
    l_f_meta = 2 * torch.sum(v_lambda[0:batch_size//2] * cost[0:batch_size//2]) + 2 * cur_lambda_u * torch.sum(v_lambda[batch_size//2::] * mse_loss[batch_size//2::])
 
    if args.regularization:
        l_r_meta = regularization(y_f_hat)
        #print(l_r_meta)
        l_f_meta = l_f_meta + l_r_meta 

    # virtual backward & update
    meta_model.zero_grad()
    grads = torch.autograd.grad(l_f_meta,(meta_model.params()),create_graph=True, allow_unused=True)
    meta_lr = optimizer_a.param_groups[0]['lr']
    # compute gradient gates
    new_grads, act_loss  = _compute_gated_grad(grads, grad_models, args.top_k, args.num_act)
    meta_model.update_params(lr_inner=meta_lr,source_params=new_grads)
    
    # parse pseudo-clean input
    input_validation, target_validation, guess_target_validation = valid_batch
    guess_target_validation_var = to_var(guess_target_validation, requires_grad=False)


    input_validation_var = to_var(input_validation, requires_grad=False)
    target_validation_var = to_var(target_validation, requires_grad=False)

    # meta forward; 
    valid_y_f_hat = meta_model(input_validation_var, detach=True)
    valid_loss = cross_entropy(valid_y_f_hat, target_validation_var, reduction=True, use_pytorch=True)


    if args.regularization:
        valid_l_r = regularization(valid_y_f_hat)
        valid_loss = valid_loss + valid_l_r
   
    mse_loss = 0.
    valid_params = list(meta_model.params())[-2::]
    
    
    # meta backward & update
    valid_grads = torch.autograd.grad(valid_loss, tuple(valid_params), create_graph=True)
    
    
    ## compute last gradient loss
    for train_grad, valid_grad in zip(grads[-2::], valid_grads):
        if len(train_grad.size()) >= 2:
            dim0 = train_grad.size()[0]
            grad_target = valid_grad.detach()
            g_mean, g_std = torch.mean(grad_target.view(dim0, -1), dim=-1), torch.std(grad_target.view(dim0, -1), dim=-1)
            pg_mean, pg_std = torch.mean(train_grad.view(dim0, -1), dim=-1), torch.std(train_grad.view(dim0, -1), dim=-1)
            mse_loss += (args.mse_factor * torch.mean((pg_mean - g_mean)**2 + (pg_std - g_std)**2))
        else:
            grad_target = valid_grad.detach()
            g_mean, g_std = torch.mean(grad_target), torch.std(grad_target)
            pg_mean, pg_std = torch.mean(train_grad), torch.std(train_grad)
            mse_loss += (args.mse_factor * torch.mean((pg_mean - g_mean)**2 + (pg_std - g_std)**2))
    valid_loss += mse_loss
    
    ## add or not
    valid_loss += (args.act_factor * act_loss)
 
    optimizer_c.zero_grad()
    for go in grad_optimizers:
        go.zero_grad()
    valid_loss.backward()
    optimizer_c.step()
    for go in grad_optimizers:
        go.step()
    del grads, valid_grads
    
    return valid_loss, 0.


    
def model_step(model, optimizer_a, vnet, train_batch, e_i):
    # actual forward & backward & update
    input_train, target_train, guess_target_train = train_batch
    input_var = to_var(input_train, requires_grad=False)
    target_var = to_var(target_train, requires_grad=False)
    guess_target_var = to_var(guess_target_train, requires_grad=False)

    y_f = model(input_var)
    cost_w = cross_entropy(y_f, target_var, reduction=False, use_pytorch=True)
    cost_v = torch.reshape(cost_w, (len(cost_w), 1))


    with torch.no_grad():
        w_new = vnet(cost_v)


    mse_loss = mse(y_f, guess_target_var, reduction='none' )
    w_new = w_new.view(-1)#
    batch_size = w_new.size()[0]

    w_new = norm_weight(w_new)
    cur_lambda_u = get_lambda_u(e_i, rampup_length=args.rampup_length)
    l_f = 2*torch.sum(w_new[0:batch_size//2] * cost_w[0:batch_size//2]) + 2*cur_lambda_u * torch.sum(w_new[batch_size//2::] * mse_loss[batch_size//2::])

    if args.regularization:
        l_r = regularization(y_f)
        l_f = l_f + l_r


    optimizer_a.zero_grad()
    l_f.backward()
    optimizer_a.step()
    
    return 0., l_f


def training_step(train_batch, model_1, optimizer_a, vnet1, optimizer_c, valid_batch, e_i, grad_models_1, grad_optimizers_1):
    # virtual train & meta train
    valid_loss, prec_meta = meta_step(model_1, optimizer_a, vnet1, optimizer_c, train_batch, valid_batch, e_i, grad_models_1, grad_optimizers_1)
    # actual train
    prec_train, l_f = model_step(model_1, optimizer_a, vnet1, train_batch, e_i)

    return l_f, valid_loss, prec_train, prec_meta


def test(model, test_loader, criterion):
    losses_test = AverageMeter()
    top1_test = AverageMeter()
    model.eval()
    for i, (input_test, target_test) in enumerate(test_loader):
        input_test_var = to_var(input_test, requires_grad=False)
        target_test_var = to_var(target_test, requires_grad=False)

                    # compute output
        with torch.no_grad():
            output_test = model(input_test_var)
        loss_test = criterion(output_test, target_test_var)
        prec_test = accuracy(output_test.data, target_test_var.data, topk=(1,))[0]

        losses_test.update(loss_test.data.item(), input_test_var.size(0))
        top1_test.update(prec_test.item(), input_test_var.size(0))

    print(' \t* Prec@1 {top1.avg:.3f}'.format(top1=top1_test))
    return top1_test


def test_model_ensembel(model_1, model_2, test_loader, criterion):

    losses_test = AverageMeter()
    top1_test = AverageMeter()
    model_1.eval()
    model_2.eval()
    for i, (input_test, target_test) in enumerate(test_loader):
        input_test_var = to_var(input_test, requires_grad=False)
        target_test_var = to_var(target_test, requires_grad=False)

        with torch.no_grad():
            output_test_1 = model_1(input_test_var)
            output_test_2 = model_2(input_test_var)
        output_test = (output_test_1 + output_test_2)*0.5
        loss_test = criterion(output_test, target_test_var)
        prec_test = accuracy(output_test.data, target_test_var.data, topk=(1,))[0]

        losses_test.update(loss_test.data.item(), input_test_var.size(0))
        top1_test.update(prec_test.item(), input_test_var.size(0))

    print(' \t* Prec@1 {top1.avg:.3f}'.format(top1=top1_test))
    return top1_test


## fit 2 component GMM 
def eval_train(model, dataloader, all_loss):
    model.eval()
    losses = torch.zeros(dataloader.dataset.__len__())
    with torch.no_grad():
        for batch_idx, (inputs, targets, index) in enumerate(dataloader):
            inputs, targets = inputs.cuda(), targets.cuda() 
            input_var = to_var(inputs, requires_grad=False)
            target_var = to_var(targets, requires_grad=False)

            outputs = model(inputs) 
            loss = F.cross_entropy(outputs, target_var, reduce = False)
            for b in range(inputs.size(0)):
                losses[index[b]]=loss[b]        
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)

    input_loss = losses.reshape(-1,1)

    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]         
    return prob, all_loss


## split pseudo clean label set
def split_pseudo_clean_set(model, train_loader, eval_loader, all_loss, p_threshold, meta_threshold):
    prob, all_loss = eval_train(model, eval_loader, all_loss)
    
    # generate meta set
    meta_pred = (prob > meta_threshold)
    train_meta_loader  = train_loader.run('labeled', meta_pred, prob)
    
    # generate train set
    pred = (prob > p_threshold)
    
    ## prevent from bad cases
    if pred.sum() < 64 or pred.sum() > 45000:
         pred = (prob > np.median(prob))
            
    labeled_trainloader, unlabeled_trainloader = train_loader.run('train', pred, prob)
    
    return labeled_trainloader, unlabeled_trainloader, train_meta_loader, all_loss
        
def train_epoch(epoch, total_epochs, model_a, model_b, labeled_loader, unlabeled_loader, meta_loader, optimizer_a, vnet_a, optimizer_c, grad_models, grad_model_optimizers):
    loss_stats = []
    ## train model
    print('Training Epoch: [%d/%d]'% (epoch, total_epochs))
    model_a.train()
    model_b.eval() 

    total_iters = max(len(labeled_loader), len(unlabeled_loader))

    iter_labeled_loader = iter(labeled_loader)
    iter_unlabeled_loader = iter(unlabeled_loader)
    iter_meta_loader = iter(meta_loader)

    for iters in tqdm(range(total_iters)):

        try:
            labeled_data = next(iter_labeled_loader)
        except StopIteration as e:
            iter_labeled_loader = iter(labeled_loader)
            labeled_data = next(iter_labeled_loader)

        try:
            unlabeled_data = next(iter_unlabeled_loader)
        except StopIteration as e:
            iter_unlabeled_loader = iter(unlabeled_loader)
            unlabeled_data = next(iter_unlabeled_loader)

        try:
            valid_batch = next(iter_meta_loader)
        except StopIteration as e:
            iter_meta_loader = iter(meta_loader)
            valid_batch = next(iter_meta_loader)

        train_batch = convert_train_data(labeled_data, unlabeled_data, model_a, model_b, args.num_classes, args.alpha)

        valid_batch = convert_meta_data(valid_batch, model_a, model_b, args.num_classes, args.alpha)
        l_f, valid_loss, prec_train, prec_meta = training_step(train_batch, model_a, optimizer_a, vnet_a, optimizer_c, valid_batch, epoch, grad_models, grad_model_optimizers)
        
        
        loss_stats.append(l_f.item())
    print('\t Loss: %.4f\t MetaLoss:%.4f\t Prec@1 %.2f\t Prec_meta@1 %.2f \tLr %.4f ' % (
                  np.mean(loss_stats), valid_loss, prec_train, prec_meta, optimizer_a.param_groups[0]['lr']))


def main():
    global args, best_acc
    best_acc = -1
    args = parser.parse_args()
    print()
 
    args.num_classes = 100
    args.data_root = './dataset/mini-imagenet/'
    
    args.train_split_file = '{}/split/{}'.format(args.data_root, args.split)
    args.train_path = '{}/training_s32/{}'.format(args.data_root, args.split)
    args.test_split_file = '{}/split/clean_validation'.format(args.data_root)
    args.test_path = '{}/validation_s32/'.format(args.data_root)
    
    
    save_dir='{}/{}'.format(args.out_root, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    acc_log=open('{}/{}_acc.txt'.format(args.out_root, args.name),'w')
 
    train_loader = mini_imagenet_dataloader.red_mini_imagenet_dataloader(args.train_split_file, batch_size=args.batch_size, num_workers=6, root_dir=args.train_path)
    test_loader = mini_imagenet_dataloader.red_mini_imagenet_dataloader(args.test_split_file, batch_size=args.batch_size, num_workers=2, root_dir=args.test_path)

    
    warmup_loader = train_loader.run('warmup')
    test_loader = test_loader.run('test')
    # create model1
    model_1, optimizer_a_1, vnet_1, optimizer_c_1 = build_training(args)
    grad_models_1, grad_optimizers_1 = build_grad_models(args, model_1)

    # create model2
    model_2, optimizer_a_2, vnet_2, optimizer_c_2 = build_training(args)
    grad_models_2, grad_optimizers_2 = build_grad_models(args, model_2)

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    all_loss = [[],[]] # save the history of losses from two networks

    if args.warmup:
        print('warm up for model 1 ...')
        warmup(model_1, optimizer_a_1, warmup_loader, test_loader, criterion)
        warmup(model_2, optimizer_a_2, warmup_loader, test_loader, criterion)

    eval_loader = train_loader.run('eval_train')
    
    for e_i in range(args.epochs):

        if e_i % 50 == 0:
            save_model(model_1, vnet_1, e_i, "1", args.out_root, args.name)
            save_model(model_2, vnet_2, e_i, "2", args.out_root, args.name)
        if e_i % 1 == 0:
            print('update GMM model')
            ## generate training dataset & exchange
            labeled_trainloader2, unlabeled_trainloader2, train_meta_loader2, all_loss[0] = split_pseudo_clean_set(model_1, train_loader, eval_loader, all_loss[0], args.p_threshold, args.meta_threshold)
            labeled_trainloader1, unlabeled_trainloader1, train_meta_loader1, all_loss[1] = split_pseudo_clean_set(model_2, train_loader, eval_loader, all_loss[1], args.p_threshold, args.meta_threshold)
            
        # exchange
        ## train model 1
        train_epoch(e_i, args.epochs, model_1, model_2, labeled_trainloader1, unlabeled_trainloader1, train_meta_loader1, optimizer_a_1, vnet_1, optimizer_c_1, grad_models_1, grad_optimizers_1)
        
        ## train model 2
        train_epoch(e_i, args.epochs, model_2, model_1, labeled_trainloader2, unlabeled_trainloader2, train_meta_loader2, optimizer_a_2, vnet_2, optimizer_c_2, grad_models_2, grad_optimizers_2)
        
        ## test model 1
        acc_1 = test(model_1, test_loader, criterion)
        
        ## test model 2
        acc_2 = test(model_2, test_loader, criterion)
        
        acc_ensemble = test_model_ensembel(model_1, model_2, test_loader, criterion)
        
        if acc_ensemble.avg > best_acc:
            best_acc = acc_ensemble.avg
        acc_log.write('epoch: {} model1: {top1.avg:.3f}\n'.format(e_i, top1=acc_1))
        acc_log.write('epoch: {} model2: {top1.avg:.3f}\n'.format(e_i, top1=acc_2))
        acc_log.write('epoch: {} model_ensemble: {top1.avg:.3f}, best_acc: {best_acc:.3f}\n'.format(e_i, top1=acc_ensemble, best_acc=best_acc))
        acc_log.flush()

        
if __name__ == '__main__':
    main()
