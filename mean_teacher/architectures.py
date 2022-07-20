import pdb
import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from .utils import export, parameter_count

import torchvision.models as models

@export
def res18(pretrained_facedb='default', attention=False, num_classes=8, **kwargs):
    #assert not pretrained
    model = Res18Feature(pretrained=pretrained_facedb, drop_rate=0, attention=attention, num_classes=num_classes)
    return model


class Res18Feature(nn.Module):
    def __init__(self, pretrained='default', num_classes=8, drop_rate=0, attention=False):
        super(Res18Feature, self).__init__()
        self.drop_rate = drop_rate
        self.attention = attention
        print("num classes is ", num_classes)

        resnet = models.resnet18(True)
        if pretrained == 'ms-celeb':
            #we initialize from a pretrained model on msceleb
            checkpoint = torch.load('/mnt/Data/tohar/resnet18_msceleb.pth')
            resnet.load_state_dict(checkpoint['state_dict'], strict=True)
            print(" ==== initialising from ms-celeb")

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # after avgpool 512x1

        fc_in_dim = list(resnet.children())[-1].in_features  # original fc layer's in dimention 512
        self.fc = nn.Linear(fc_in_dim, num_classes)  # new fc layer 512x7

        if attention:
            self.alpha = nn.Sequential(nn.Linear(fc_in_dim, 1), nn.Sigmoid())


    def forward(self, x, return_w=False, return_features=False, mode_val=False, drop_rate=False):
        x = self.features(x)
        feat = x

        if self.drop_rate > 0 or drop_rate:
            x = nn.Dropout(self.drop_rate)(x)
        x = x.view(x.size(0), -1)
        dict={}

        if mode_val:
            return self.fc(x)

        if self.attention:
            logits = self.fc(x)
            attention_weights = self.alpha(x)
            out = attention_weights * logits
            if return_w:
                dict['attention_weights'] = attention_weights
                dict['pure_logits'] = logits

        else:
            out = self.fc(x)
        if return_features:
            dict['feat'] = feat

        if return_features or return_w:
            return dict, out
        else:
            return out
