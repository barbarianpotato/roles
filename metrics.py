import numpy
from  sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy import spatial
import networkx as nx
import math


class AUC_ROC ():
    def __init__(self,feature_mat,role_mat,adj):
        self.y_score = numpy.array([])
        self.y_true = numpy.array([])
        self.feature_mat = feature_mat
        self.role_mat = role_mat
        self.sim_mat = numpy.zeros((len(self.feature_mat),len(self.feature_mat)))
        self.adj = adj
    
    def flatten (self):
        self.y_true = self.adj.flatten()
        self.y_score = self.sim_mat.flatten()

    def roc_score (self):
        return roc_auc_score(self.y_true,self.y_score)

    def precision (self):
        return average_precision_score(self.y_true,self.y_score)

    def similarity(self):
        sim_mat_feat = []
        sim_mat_roles = []
        mod_square_feat = []
        mod_square_role = []
        sim_mat_feat = numpy.zeros((len(self.feature_mat),len(self.feature_mat)))
        sim_mat_roles = numpy.zeros((len(self.feature_mat),len(self.feature_mat)))
        for k in range(len(self.feature_mat)):
            mod_square_feat.append(math.sqrt(numpy.sum(numpy.square(self.feature_mat[k]))))
            mod_square_role.append(math.sqrt(numpy.sum(numpy.square(self.role_mat[k]))))
        for k in range (len(self.feature_mat)):
            for j in range (k,len(self.feature_mat)):
                sim_mat_feat[k][j] = numpy.sum(numpy.multiply(self.feature_mat[k],self.feature_mat[j])) / (mod_square_feat[k]*mod_square_feat[j])
                sim_mat_roles[k][j] = numpy.sum(numpy.multiply(self.role_mat[k],self.role_mat[j])) / (mod_square_role[k]*mod_square_role[j])
                # sim_mat_feat[k].append(1-spatial.distance.cosine(self.feature_mat[k], self.feature_mat[j]))
                # sim_mat_roles[k].append(1-spatial.distance.cosine(self.role_mat[k], self.role_mat[j]))
                self.sim_mat[k][j]=((sim_mat_feat[k][j]+sim_mat_roles[k][j])/2)
                self.sim_mat[j][k]=((sim_mat_feat[k][j]+sim_mat_roles[k][j])/2)
        return self.sim_mat



def save (dataset, args, method):
    file = open("./output/{}/{}.txt".format(method,dataset), "w")
    file.write(args)
    file.close()

    
    





    
