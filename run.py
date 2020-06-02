import torch
import sys
from train import train
from metrics import AUC_ROC,save,ROB
from preprocess import import_data
import networkx as nx


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("Computing device: ", device)
dataset = sys.argv[2]
method = sys.argv[1]
data,adj = import_data('./datasets/'+ dataset +'.gml')
print ("Nodes: ",data[1])

if method=='a':
    print ("Computing precision & AUC-ROC.\n")
    feat_mat,role_mat = train(data,device,method)
    auc = AUC_ROC(feat_mat,role_mat,adj)
    print ("Calculating similarity between nodes...")
    auc.similarity()
    auc.flatten()
    precision = auc.precision()
    roc_score = auc.roc_score()
    print ("PREC : {:.6f}  |  AUC-ROC : {:.6f}".format(precision,roc_score))
    save(dataset,"precision : {}\nroc_score: {}\n".format(precision,roc_score),method = 'auc_prec')
elif method=='r':
    print ("Computing robustness.\n")
    rob = ROB(adj,device,dataset)
    rob.calculate()
    x,y = rob.plot(dataset)
    save(dataset,"AUC_ROC : {}\nRatio: {}\n".format(x,y),method = 'robust')
else:
    print ("\nInvalid method selected.\n")
    print ("Methods available :\n1. Precision & AUC-ROC\n2. Robustness\nFor (1.) type 'a' & (2.) type 'r' in the command.\n")

















