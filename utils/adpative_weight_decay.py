import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from time import sleep


class AdaptiveWeightDecay(nn.Module):

    def __init__(self, net: nn.Module, loss_function: nn.Module, opt: Optimizer, c1=2, c2=0.5, m1=2, m2=1.1):
        super().__init__()
        self.alpha = 1e-8
        self.net = net
        self.loss_function = loss_function
        self.opt = opt
        self.c1, self.c2, self.m1, self.m2 = c1, c2, m1, m2
        self.grad = 0.  # torch.zeros(self.vectorize_parameters().size())

    def vectorize_parameters(self):
        p_list = list(self.net.parameters())
        p_v_list = list()
        g_v_list = list()
        for p in p_list:
            if len(p.size()) > 1:
                p_v_list.append(p.view(-1))
                if p.grad is not None:
                    g_v_list.append(p.grad.view(-1))
        w_v = torch.cat(p_v_list, dim=0)
        if len(g_v_list) > 0:
            self.grad = torch.cat(g_v_list, dim=0)
        return w_v

    def update_alpha(self, v, verbose=True):
        w = self.vectorize_parameters()
        e_w = torch.norm(self.grad) / (self.alpha * torch.norm(w))
        e_w = e_w.detach().cpu().numpy()
        if v > self.m1:
            self.alpha = self.alpha * np.max([self.c1, e_w])
        elif v < self.m2:
            self.alpha = self.alpha * np.min([self.c2, e_w])
        self.alpha = np.clip(self.alpha, 1e-8, 10.)
        if verbose:
            print('New Value of Alpha is {0:01f}'.format(self.alpha))

    def compute_loss(self, data_loader: DataLoader, dataset_tag):
        total_loss = 0
        counter = 0
        with torch.no_grad():
            for data in data_loader:
                x = data[0].cuda()
                y = data[1].cuda()
                outputs = self.net(x)
                total_loss += self.loss_function(outputs, y)
                counter += 1
        tqdm.write('Loss of the network on the %s data is: %f' % (dataset_tag,
                                                                  total_loss / counter))
        return total_loss / counter

    def forward(self, network_output, target):
        loss = self.loss_function(network_output, target)
        w_norm = torch.norm(self.vectorize_parameters())
        return loss + self.alpha * w_norm, loss

    def fit(self, train_loader: DataLoader, test_loader: DataLoader, max_epoch):
        for epoch in range(max_epoch):
            losses = 0.
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=100)
            for i, batch in pbar:
                self.opt.zero_grad()
                output = self.net(batch[0].float().cuda())
                loss_reg, loss = self(output, batch[1].long().cuda())
                loss_reg.backward()
                self.opt.step()
                losses += loss.detach().cpu().numpy()
                pbar.set_description("Loss Value is %f" % (losses / (i + 1)))

            sleep(0.25)
            losses /= len(train_loader)
            tqdm.write('Loss of the network on the %s data is: %f' % ('Train', losses))
            test_loss = self.compute_loss(test_loader, 'Test')
            v = test_loss / losses
            self.update_alpha(v, True)
            sleep(0.25)