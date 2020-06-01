import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder (nn.Module):
    def __init__(self,in_feats,out_feats_1=16,out_feats_2=8):
        super(Encoder, self).__init__()
        self.encoder1 = nn.Linear(in_feats,out_feats_1).double()
        self.encoder2 = nn.Linear(out_feats_1,out_feats_2).double()
        self.decoder1 = nn.Linear(out_feats_2,out_feats_1).double()
        self.decoder2 = nn.Linear(out_feats_1,in_feats).double()

    def forward(self,x):
        x = self.encoder1(x)
        x = F.relu(x)
        y = x
        x = self.encoder2(x)
        x = F.relu(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        return x,y


