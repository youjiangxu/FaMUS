import torch

from grad_operator_layer import GradGumbelSoftmax
from preresnet import PreActResNet18, VNet


def build_training(args):
    model = build_model(args)
     
    optimizer_a = torch.optim.SGD(model.params(), args.lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    vnet = VNet(1, 100, 1)
    vnet = vnet.cuda()

    optimizer_c = torch.optim.SGD(vnet.params(), args.meta_lr,
                                  momentum=args.momentum, nesterov=args.nesterov,
                                  weight_decay=args.weight_decay)
    return model, optimizer_a, vnet, optimizer_c


def build_model(args):
    
    if args.network == 'PreActResNet18': 
        model = PreActResNet18(num_classes=args.num_classes)
    else:
        assert False, 'Error args.network: {}'.format(args.network)
    if torch.cuda.is_available():
        model.cuda()
        torch.backends.cudnn.benchmark = True
    return model

        
def build_thres_model(args, weight_shape):
    hidden_dim = 128
    model = GradGumbelSoftmax(weight_shape[0], hidden_dim, input_norm=True)
    if torch.cuda.is_available():
        model.cuda()
    return model


def build_grad_models(args, model):
    grad_models = []
    grad_optimizers = []
    for param in list(model.params())[-args.top_k:-2]:
        param_shape = param.size()
        _grad_model = build_thres_model(args, param_shape)
        _optimizer = torch.optim.SGD(_grad_model.params(), args.go_lr,
            momentum=args.momentum, nesterov=args.nesterov,
            weight_decay=0)
        grad_models.append(_grad_model)
        grad_optimizers.append(_optimizer)
    return grad_models, grad_optimizers