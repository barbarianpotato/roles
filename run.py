import torch
import sys
from train import autoencode,roleextraction,train
from metrics import AUC_ROC,save
from preprocess import import_data
import networkx as nx


torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("Computing device: ", device)
dataset = sys.argv[1]
data,adj = import_data('./datasets/'+ dataset +'.gml')
print ("Nodes: ",data[1])
feat_mat,role_mat = train(data,device)
auc = AUC_ROC(feat_mat,role_mat,adj)
print ("Calculating similarity between nodes...")
auc.similarity()
auc.flatten()
precision = auc.precision()
roc_score = auc.roc_score()
print ("PREC : {:.6f}  |  AUC-ROC : {:.6f}".format(precision,roc_score))
save(dataset,"precision : {}\nroc_score: {}\n".format(precision,roc_score),method = 'auc_prec')



















