import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
import torch.nn.init as init


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


class MetaModule(nn.Module):
    # adopted from: Adrien Ecoffet https://github.com/AdrienLE
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params_v2(self, lr_inner, first_order=False, source_params=None, detach=False):
        '''
             ours update
        '''
        if source_params is not None:
            for tgt, grad in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                #grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad.data.detach()
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)


    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        '''
            official implementation
        '''
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                if src is None:
                    continue
                #grad = src
                if first_order:
                    src = to_var(src.detach().data)
                tmp = param_t - lr_inner * src
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()  # https://blog.csdn.net/qq_39709535/article/details/81866686
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class GradMetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)
        ignore.weight.data.zero_()
        ignore.bias.data.copy_(torch.tensor([0.], dtype=torch.float))
        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]



class MetaConv2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Conv2d(*args, **kwargs)

        self.in_channels = ignore.in_channels
        self.out_channels = ignore.out_channels
        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups
        self.kernel_size = ignore.kernel_size

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaConvTranspose2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.ConvTranspose2d(*args, **kwargs)

        self.stride = ignore.stride
        self.padding = ignore.padding
        self.dilation = ignore.dilation
        self.groups = ignore.groups

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))

        if ignore.bias is not None:
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))
        else:
            self.register_buffer('bias', None)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(x, output_size)
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding,
                                  output_padding, self.groups, self.dilation)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]



class MetaBatchNorm1d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm1d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]



class MetaBatchNorm2d(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.BatchNorm2d(*args, **kwargs)

        self.num_features = ignore.num_features
        self.eps = ignore.eps
        self.momentum = ignore.momentum
        self.affine = ignore.affine
        self.track_running_stats = ignore.track_running_stats

        if self.affine:
            self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
            self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(self.num_features))
            self.register_buffer('running_var', torch.ones(self.num_features))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

    def forward(self, x):
        return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias,
                            self.training or not self.track_running_stats, self.momentum, self.eps)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]


class MetaBasicBlock(MetaModule):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(MetaBasicBlock, self).__init__()

        self.bn1 = MetaBatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = MetaConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = MetaBatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = MetaConv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and MetaConv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class MetaNetworkBlock(MetaModule):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(MetaNetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(MetaModule):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()
    def forward_detach(self, x, detach_point='final'):
        out = self.conv1(x)
        if detach_point == 'conv1':
            out = out.detach()

        out = self.block1(out)
        if detach_point == 'block1':
            out = out.detach()

        out = self.block2(out)
        if detach_point == 'block2':
            out = out.detach()

        out = self.block3(out)
        if detach_point == 'block3':
            out = out.detach()

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        if detach_point == 'final':
            out = out.detach()
        return self.fc(out)

    def forward_feat(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out, self.fc(out)


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)



class SpotTuneWideResNet(MetaModule):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(SpotTuneWideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.num_block = n

        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()
    def forward_block(self, x, res_block, policy):
        num_blocks = len(res_block)
        for idx in range(num_blocks):
            _x = res_block[idx](x)
            with torch.no_grad():
                _x_wo_grad = res_block[idx](x)
            po = policy[:, idx].view(-1, 1, 1, 1)
            x = torch.where(po.float()==1.0, po.float()*_x, ((1-po).float()*_x_wo_grad).detach())
        return x
    def forward_feat(self, x, policy=None):
        out = self.conv1(x)

        if policy is not None:
            out = self.forward_block(out, self.block1, policy[:, 0:0+self.num_block])
            out = self.forward_block(out, self.block2, policy[:, 0+self.num_block:0+self.num_block*2])
            out = self.forward_block(out, self.block3, policy[:, 0+self.num_block*2:0+self.num_block*3])
        else:
            out = self.block1(out)
            out = self.block2(out)
            out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out, self.fc(out)


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


from gumbel_softmax import gumbel_softmax
class ThresResNet(MetaModule):
    def __init__(self, depth, num_classes=160, widen_factor=1, dropRate=0.0):
        super(ThresResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = torch.mean(out, 0, keepdim=True)
        out = self.fc(out)
        
        out = out.view(-1, 2)
        return gumbel_softmax(out)
class Thres5LinkResNet(MetaModule):
    def __init__(self, depth, num_classes=80, widen_factor=1, dropRate=0.0):
        super(Thres5LinkResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = MetaBasicBlock
        # 1st conv before any network block
        self.conv1 = MetaConv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = MetaNetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = MetaNetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = MetaNetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = MetaBatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = MetaLinear(nChannels[3], num_classes)
        self.fc2 = MetaLinear(nChannels[3], num_classes)
        self.fc3 = MetaLinear(nChannels[3], num_classes)
        self.fc4 = MetaLinear(nChannels[3], num_classes)
        self.fc5 = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = torch.mean(out, 0, keepdim=True)
        out1 = gumbel_softmax(self.fc1(out))
        out2 = gumbel_softmax(self.fc2(out))
        out3 = gumbel_softmax(self.fc3(out))
        out4 = gumbel_softmax(self.fc4(out))
        out5 = gumbel_softmax(self.fc5(out))
        res = torch.cat([out1, out2, out3, out4, out5], dim=0)
         
        return torch.max(res, dim=0)[0]


class ThresModule(MetaModule):
    def __init__(self, depth, num_classes=160, widen_factor=1, dropRate=0.0):
        super(ThresModule, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        self.fc = MetaLinear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, MetaConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, MetaBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, MetaLinear):
                m.bias.data.zero_()


    def forward(self, x):
        out = torch.mean(x, 0, keepdim=True)
        out = self.fc(out)
        
        out = out.view(-1, 2)
        return gumbel_softmax(out)


class GradModule(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradModule, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class GradModuleV2(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradModuleV2, self).__init__()
        self.alpha = nn.Sequential(
            MetaLinear(input, hidden),
            nn.ReLU(inplace=True),
            MetaLinear(hidden, output)
            )
        self.grad_feat = nn.Sequential(
            MetaLinear(input, hidden),
            nn.ReLU(inplace=True),
            MetaLinear(hidden, output),
            nn.Sigmoid()
            )

    def forward(self, x):
        weights = self.alpha(x)
        feats = self.grad_feat(x)
        return weights, feats





class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)

class GradVNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradVNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = GradMetaLinear(hidden, output)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)


class ComplexVNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(ComplexVNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, hidden, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.linear3 = MetaLinear(hidden, output, bias=True)



    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        out = self.linear3(x)
        return F.sigmoid(out)



class NormVNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(NormVNet, self).__init__()
        self.norm0 = MetaBatchNorm1d(input)
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)



    def forward(self, x):
        x = self.norm0(x)
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)

#class NormVNet(MetaModule):
#    def __init__(self, input, hidden, output):
#        super(NormVNet, self).__init__()
#        self.norm0 = MetaBatchNorm1d(input)
#        self.linear1 = MetaLinear(input, hidden)
#        self.norm1 = MetaBatchNorm1d(hidden)
#        self.relu = nn.ReLU(inplace=True)
#        self.linear2 = MetaLinear(hidden, output)
#
#
#
#    def forward(self, x):
#        x = self.norm0(x)
#        x = self.linear1(x)
#        x = self.norm1(x)
#        x = self.relu(x)
#        out = self.linear2(x)
#        return F.sigmoid(out)
class GradAmplifierLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        output = x
        return output
    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 100.
        return grad_input


class GradDiscriminatorNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradDiscriminatorNet, self).__init__()
        self.dis = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        out = self.dis(x)
        return torch.sigmoid(out)


class GradDiscriminatorPReLUNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradDiscriminatorPReLUNet, self).__init__()
        self.dis = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        out = self.dis(x)
        return torch.sigmoid(out)


class GradComplexDiscriminatorPReLUNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradComplexDiscriminatorPReLUNet, self).__init__()
        self.dis = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            MetaLinear(hidden, hidden),
            nn.PReLU(),
            MetaLinear(hidden, hidden),
            nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            #nn.ReLU(inplace=True),
            nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        out = self.dis(x)
        return torch.sigmoid(out)





class Mine(MetaModule):
    def __init__(self, input, hidden, output):
        super(Mine, self).__init__()
        self.net = torch.nn.Sequential(
            MetaLinear(input, hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        out = self.net(x)
        return out




class GradEncodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradEncodeNet, self).__init__()
        self.cor_encode = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )
        self.incor_encode = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        cor_feat = self.cor_encode(x)
        incor_feat = self.incor_encode(x)
        return cor_feat, incor_feat

class GradTransformNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradTransformNet, self).__init__()
        self.cor_encode = torch.nn.Sequential(
            MetaLinear(input, hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        cor_feat = self.cor_encode(x)
        return cor_feat 


class TargetGradEncodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(TargetGradEncodeNet, self).__init__()
        self.cor_encode = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        cor_feat = self.cor_encode(x)
        return cor_feat

class GradDecodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradDecodeNet, self).__init__()
        self.cor_encode = torch.nn.Sequential(
            MetaLinear(input, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, hidden),
            #MetaBatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            #nn.PReLU(),
            MetaLinear(hidden, output)
            )

    def forward(self, x):
        out = self.cor_encode(x)
        return out


class SuperLinearLayer(nn.Module):
    def __init__(self, input, hidden, output):
        super(SuperLinearLayer, self).__init__()
        gamma = 0.2
        self.linear1 = nn.Linear(input, hidden, bias=True)
        self.linear2 = nn.Linear(hidden, output, bias=False)
        self.linear1.weight.data.fill_(0.)
        self.linear1.bias.data.fill_(1.0)
        self.linear2.weight.data.fill_(gamma / output)
            

    def forward(self, x):
         
        x = self.linear1(x)
        out = self.linear2(x)
        return out

class HyperParamGradEncodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(HyperParamGradEncodeNet, self).__init__()
        init_gamma = 0.1
        #self.encode_cor_sl_1 = SuperLinearLayer(input, hidden, weight_init=0.0, bias_init=1.0)
        #self.encode_cor_sl_2 = SuperLinearLayer(hidden, output, bias=False, weight_init=init_gamma/output, bias_init=0.)
        self.encode_cor_sl = SuperLinearLayer(input, hidden, output)
        #self.encode_incor_sl_1 = SuperLinearLayer(input, hidden, weight_init=0.0, bias_init=1.0)
        #self.encode_incor_sl_2 = SuperLinearLayer(hidden, output, bias=False, weight_init=init_gamma/output, bias_init=0.)
        self.encode_incor_sl = SuperLinearLayer(input, hidden, output)

    def forward(self, x):
        p1 = self.encode_cor_sl(x)
        cor_feat = p1 * x

        ipi = self.encode_incor_sl(x)
        incor_feat = ipi * x
        return cor_feat, incor_feat


class HyperParamTargetGradEncodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(HyperParamTargetGradEncodeNet, self).__init__()
        init_gamma = 0.1
        #self.encode_sl_1 = SuperLinearLayer(input, hidden, weight_init=0.0, bias_init=1.0)
        #self.encode_sl_2 = SuperLinearLayer(hidden, output, bias=False, weight_init=init_gamma/output, bias_init=0.)
        self.encode_sl = SuperLinearLayer(input, hidden, output)

    def forward(self, x):
        #p1 = self.encode_sl_2(self.encode_sl_1(x))
        p1 = self.encode_sl(x)

        return p1 * x


class HyperParamGradDecodeNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(HyperParamGradDecodeNet, self).__init__()
        init_gamma = 0.1
        #self.decode_cor_sl_1 = SuperLinearLayer(input, hidden, weight_init=0.0, bias_init=1.0)
        #self.decode_cor_sl_2 = SuperLinearLayer(hidden, output, bias=False, weight_init=init_gamma/output, bias_init=0.)
        self.decode_cor_sl = SuperLinearLayer(input, hidden, output) 
        #self.decode_incor_sl_1 = SuperLinearLayer(input, hidden, weight_init=0.0, bias_init=1.0)
        #self.decode_incor_sl_2 = SuperLinearLayer(hidden, output, bias=False, weight_init=init_gamma/output, bias_init=0.)
        self.decode_incor_sl = SuperLinearLayer(input, hidden, output)


    def forward(self, x1, x2):
        ##p1 = self.decode_cor_sl_2(self.decode_cor_sl_1(x1))
        #p2 = self.decode_incor_sl_2(self.decode_incor_sl_1(x2))
        p1 = self.decode_cor_sl(x1)
        p2 = self.decode_incor_sl(x2)
        out = p1 * x1 + p2 * x2

        return out

class GradReconstructNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(GradReconstructNet, self).__init__()
        self.linear1_x1 = nn.Linear(input, hidden) 
        #self.norm_1 = MetaBatchNorm1d(hidden)
        self.linear1_x2 = nn.Linear(input, hidden) 
        #self.norm_2 = MetaBatchNorm1d(hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2*hidden, hidden)
        #self.norm_3 = MetaBatchNorm1d(hidden)
        self.linear3 = nn.Linear(hidden, output)


    def forward(self, x1, x2):
        #x1 = self.relu(self.norm_1(self.linear1_x1(x1)))
        #x2 = self.relu(self.norm_2(self.linear1_x2(x2)))
        #x3 = self.relu(self.norm_3(self.linear2(torch.cat([x1 + x2], dim=1))))
        x1 = self.relu(self.linear1_x1(x1))
        x2 = self.relu(self.linear1_x2(x2))
        x3 = self.relu(self.linear2(torch.cat([x1 + x2], dim=1)))

        out = self.linear3(x3)
        return out








class VNetFeat(MetaModule):
    def __init__(self, input, hidden, output, feat_input=100):
        super(VNetFeat, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(2*hidden, output)

        self.feat_linear1 = MetaLinear(feat_input, hidden)




    def forward(self, x, feat):
        x = self.linear1(x)
        x = self.relu(x)
        feat = self.feat_linear1(feat)
        feat = self.relu(feat)
        x = torch.cat([x, feat], dim=1)
        out = self.linear2(x)
        return F.sigmoid(out)
