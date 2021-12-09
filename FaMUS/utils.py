import os
import torch

from torch.autograd import Variable


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def resume_model(model, checkpoint_path, log_content=''):
    if checkpoint_path is not None:
        print('load {} model from {}'.format(log_content, checkpoint_path))
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)


def save_model(model, vnet, ind, model_idx, out_root, prefix):
    save_dir='{}/{}'.format(out_root, prefix)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(model.state_dict(), '{}/i{}_{}'.format(save_dir, ind, 'model_{}.pth'.format(model_idx)))
    torch.save(vnet.state_dict(), '{}/i{}_{}'.format(save_dir, ind, 'vnet_{}.pth'.format(model_idx)))


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


