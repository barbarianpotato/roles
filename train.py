import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder
from nmf.nmf import nmf
import numpy

class autoencode ():
    def __init__(self,data,device,epochs,method):
        self.adj, self.nodes = data
        self.adj = self.adj.to(device)
        self.net = Encoder(self.nodes)
        self.net.to(device)
        self.epochs = epochs
        self.feature_mat = torch.FloatTensor()
        self.method = method

    def train(self):
        for epoch in range(0,self.epochs):
            optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-2)
            t = time.time()
            self.net.train()
            val , self.feature_mat = self.net(self.adj)
            loss = nn.MSELoss()(self.adj.to_dense().reshape(-1),val.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if self.method == 'a':
                print ("Epoch: {:05d} |  Loss: {:.5f} | Time: {:.4f}".format(epoch+1,loss.item(),time.time()-t))
            elif self.method == 'r':
                if epoch==self.epochs-1:
                    print ("At epoch: {:05d} |  Loss: {:.5f} | Time: {:.4f}".format(epoch+1,loss.item(),time.time()-t))

        
    def feature_matrix(self):
        return self.feature_mat
    
    def feature_matrix_numpy(self):
        return self.feature_mat.cpu().detach().numpy()

class roleextraction ():
    def __init__(self, nodefeaturematrix):
        self.nodefeaturematrix = nodefeaturematrix
        self.node_role_mat = numpy.array([])
        self.role_feat_mat = numpy.array([])
    
    def train(self):
        self.node_role_mat,self.role_feat_mat =  nmf(self.nodefeaturematrix)
    
    def node_role_matrix(self):
        return self.node_role_mat
    
    def role_feat_matrix(self):
        return self.role_feat_mat
    

def train (dataset,device,method):
    encoder = autoencode(dataset,device,20,method)
    encoder.train()
    print ("Applying non-negative matrix factorization.")
    nmf = roleextraction(encoder.feature_matrix_numpy())
    nmf.train()
    return encoder.feature_matrix_numpy(), nmf.node_role_matrix()



