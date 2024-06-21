import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torch.autograd import Variable
class Learned_Dw_Conv(nn.Module):
    global_progress = 0.0

    def __init__(self, in_channels, out_channels, fiter_kernel, stride, padding, dropout_rate, k,cardinality=32):
        super(Learned_Dw_Conv, self).__init__()

        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k
        self.group=self.out_channels//self.cardinality
        self.in_channel_per_group=in_channels//self.group
        self.dwconv = nn.Conv2d(in_channels, out_channels, fiter_kernel, stride, padding, bias=False,groups=self.group)

        self.delta_prune = (int)(self.in_channel_per_group*(self.cardinality-k)*0.25)

        self.tmp = self.group*(self.in_channel_per_group*self.cardinality- 4*self.delta_prune)
        self.pwconv = nn.Conv2d(self.tmp, out_channels, 1, 1, bias=False)

        self.dwconv2 = nn.Conv2d(self.tmp, self.tmp, fiter_kernel, stride, padding, groups=self.tmp, bias=False)
        self.register_buffer('index', torch.LongTensor(self.tmp))
        self.register_buffer('_mask_dw', torch.ones(self.dwconv.weight.size()))
        self.register_buffer('_count', torch.zeros(1))
        #self.pwconv.weight.requires_grad = False
        #self.dwconv2.weight.requires_grad = False
    def _check_drop(self):
        progress = Learned_Dw_Conv.global_progress
        if progress == 0:
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress<45 :
            self.dwconv2.weight.data.zero_()
            self.pwconv.weight.data.zero_()
        if progress>45 :
            self.dwconv.weight.data.zero_()
            ### Check for dropping
        if progress == 11 or progress == 23 or progress == 34 or progress == 45: # or progress==150 :
            # if progress == 1 or progress == 2 or progress == 3 or progress == 4:
            if progress<=45:
                self._dropping_group(self.delta_prune)
         #   else:
         #       if self.in_channel_per_group==8:
        #            self._dropping_group(16)
         #       else:
        #           self._dropping_group(32)
        return
    def _dropping_group(self,delta):
        if Learned_Dw_Conv.global_progress <= 45:
            weight=self.dwconv.weight*self.mask_dw
            weight=weight.view(self.group,self.cardinality,self.in_channel_per_group,3,3).abs().sum([3,4])
            for i in range(self.group):
                weight_tmp=weight[i,:,:].view(-1)
                di=weight_tmp.sort()[1][self.count:self.count+delta]
                for d in di.data:
                    out_ = d // self.in_channel_per_group
                    in_ = d % self.in_channel_per_group
                    self._mask_dw[i*self.cardinality+out_, in_, :, :].fill_(0)
            self.count = self.count + delta
            #print(self.in_channel_per_group)
            #print(self.delta_prune)
            #print(self.count)
        index=0
        if Learned_Dw_Conv.global_progress == 45:
            self.pwconv.weight.data.zero_()
            for i in range(self.group):
                for j in range(self.cardinality):
                    for k in range(self.in_channel_per_group):
                        if self._mask_dw[i*self.cardinality+j,k,0,0]==1:
                            self.index[index]=i*self.in_channel_per_group+k
                            self.dwconv2.weight.data[index,:,:,:]=self.dwconv.weight.data[i*self.cardinality+j,k,:,:].view(1,3,3)
                            self.pwconv.weight.data[i*self.cardinality+j,index,:,:].fill_(1)
                            index=index+1
            assert index==self.tmp
            self.dwconv.weight.data.zero_()
    def forward(self, x):
        progress = Learned_Dw_Conv.global_progress
        self._check_drop()
        if self.dropout_rate > 0:
            x = self.drop(x)
        ### Masked output

        if progress < 45:
            weight = self.dwconv.weight * self.mask_dw
            return F.conv2d(x,weight, None, self.dwconv.stride,
                            1, self.dwconv.dilation, self.group)
        else:
            x = torch.index_select(x, 1, Variable(self.index))
            x = self.dwconv2(x)
            self.pwconv.weight.data = self.pwconv.weight.data  # *self.mask_pw
            x = F.conv2d(x, self.pwconv.weight, None, self.pwconv.stride,
                         0, self.pwconv.dilation, 1)
            return x

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def mask_dw(self):
        return Variable(self._mask_dw)

    @property
    def mask_pw(self):
        return Variable(self._mask_pw)
    @property
    def ldw_loss(self):
        if Learned_Dw_Conv.global_progress >= 45:
            return 0
        weight = self.dwconv.weight * self.mask_dw
        weight_1=weight.abs().sum(-1).sum(-1).view(self.group,self.cardinality,self.in_channel_per_group)
        weight=weight.abs().sum([2,3]).view(self.group,-1)
        mask=torch.ge(weight,torch.topk(weight,2*self.in_channel_per_group,1,sorted=True)[0][:,2*self.in_channel_per_group-1]
                      .view(self.group,1).expand_as(weight)).view(self.group,self.cardinality,self.in_channel_per_group)\
            .sum(1).view(self.group,1,self.in_channel_per_group)
        mask = torch.exp((mask.float() - 1.5 * self.k) / (10)) - 1
        mask=mask.expand_as(weight_1)
        weight=(weight_1.pow(2)*mask).sum(1).clamp(min=1e-6).sum(-1).sum(-1)
        return weight

##Usage
#        self.add_module('conv2', Learned_Dw_Conv(bn_size * growth_rate,growth_rate,3,1,1,0,2,cardinality=32)),
