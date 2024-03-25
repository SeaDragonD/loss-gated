import torch, math, numpy
import torch.nn as nn
import torch.nn.functional as F
from tools import * 

# AAM softmax loss same to that in voxceleb_trainer
class LossFunction(nn.Module):
    def __init__(self, num_class, n_out = 192, m = 0.2, s = 30):
        super(LossFunction, self).__init__()
        self.pre_head =nn.Sequential(
        nn.Linear(n_out, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, num_class),
        )

        # self.m = m
        # self.s = s
        # self.weight = torch.nn.Parameter(torch.FloatTensor(n_class, n_out), requires_grad=True)
        # self.ce = nn.CrossEntropyLoss(reduction = 'none')
        # nn.init.xavier_normal_(self.weight, gain=1)
        # self.cos_m = math.cos(self.m)
        # self.sin_m = math.sin(self.m)
        # self.th = math.cos(math.pi - self.m)
        # self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, x):
        # output =  self.pre_head(x)

        # label = label.repeat(1, output.shape[1]).squeeze().cuda()
        # output = torch.cat(torch.unbind(output, dim=1), dim=0)
        #
        # closs = self.criterion(output, label)
        # # prec1 = accuracy_sup(output, label)

        return self.pre_head(x)

        # cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # sine = torch.sqrt((1.0 - torch.mul(cosine, cosine)).clamp(0, 1))
        # phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        # one_hot = torch.zeros_like(cosine)
        # one_hot.scatter_(1, label.view(-1, 1), 1)
        # output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        # output = output * self.s
        #
        # ce = self.ce(output, label)
        # # LGL
        # # mask = ce <= gate # Find the sample that loss smaller that gate
        # nselect = sum(mask).detach() # Count the num
        # loss = torch.sum(ce * mask, dim = -1) / nselect # Compute the loss for the selected data only
        # prec1 = accuracy(output.detach(), label * mask.detach(), topk=(1,))[0] * x.size()[0] # Compute the training acc for these selected data only
        # return loss, prec1, nselect