
import math
import torch
from torch import nn
from torch.nn import functional as F
from wideresnet import MetaModule, MetaLinear, MetaBasicBlock, MetaConv2d, MetaNetworkBlock, MetaBatchNorm2d, MetaBatchNorm1d
from gumbel_softmax import gumbel_softmax


class GradGumbelSoftmax(MetaModule):
    def __init__(self, input, hidden, input_norm=False):
        super(GradGumbelSoftmax, self).__init__()
        #self.bn = MetaBatchNorm1d(input)
        self.linear1 = MetaLinear(input, hidden)
        self.relu1 = nn.PReLU()
        self.linear2 = MetaLinear(hidden, hidden)
        self.relu2 = nn.PReLU()

        self.act = MetaLinear(hidden, 2)
        self.input_norm = input_norm
        #self.linear_beta = MetaLinearZero(hidden, input)
    def forward(self, x):
        if self.input_norm:
            x_mean, x_std = x.mean(), x.std()
            x = (x-x_mean)/(x_std+1e-9)
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        return gumbel_softmax(self.act(x))


def grad_function(grad, grad_model):
    grad_size = grad.size()
    if len(grad_size) == 4:
        reduced_grad = torch.sum(grad, dim=[1, 2, 3]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    elif len(grad_size) == 2:
        reduced_grad = torch.sum(grad, dim=[1]).view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    else:
        reduced_grad = grad.view(-1, grad_size[0])
        grad_act = grad_model(reduced_grad.detach())
        grad_act = grad_act[:, 1].view(-1)
    return grad_act


def _compute_gated_grad(grads, grad_models, num_opt, num_act):
#     num_opt = args.top_k
    new_grads = []
    acts = []
    gates = []
    for grad in grads[0:-num_opt]:
        new_grads.append(grad.detach())
    for g_id, grad in enumerate(grads[-num_opt:-2]):
        grad_act = grad_function(grad, grad_models[g_id])
        if grad_act > 0.5:
            new_grads.append(grad_act * grad)
        else:
            new_grads.append((1-grad_act) * grad.detach())
    acts.append(grad_act)
    for grad in grads[-2::]:
        new_grads.append(grad)
    act_loss = (torch.sum(torch.cat(acts)) - num_act)**2
    return new_grads, act_loss


