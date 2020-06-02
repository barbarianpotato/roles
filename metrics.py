import numpy
from  sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy import spatial
import networkx as nx
import math
from preprocess import normalize
from sklearn.model_selection import KFold
from train import train
import matplotlib.pyplot as plt


class AUC_ROC ():
    def __init__(self,feature_mat,role_mat,adj):
        self.y_score = numpy.array([])
        self.y_true = numpy.array([])
        self.feature_mat = feature_mat
        self.role_mat = role_mat
        self.sim_mat = numpy.zeros((len(self.feature_mat),len(self.feature_mat)))
        self.adj = adj
        self.method = 'a'
    
    def flatten(self):
        self.y_true = self.adj.flatten()
        self.y_score = self.sim_mat.flatten()

    def roc_score(self):
        return roc_auc_score(self.y_true,self.y_score)

    def precision(self):
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
                self.sim_mat[k][j]=((sim_mat_feat[k][j]+sim_mat_roles[k][j])/2)
                self.sim_mat[j][k]=((sim_mat_feat[k][j]+sim_mat_roles[k][j])/2)
        return self.sim_mat


class ROB ():
    def __init__(self,adj,device,dataset):
        self.adj = adj
        self.x_values = numpy.array([])
        self.y_values = numpy.array([])
        self.graph = nx.Graph(adj)
        self.edges = numpy.array(self.graph.edges())
        self.nodes = numpy.array(self.graph.nodes())
        self.non_edges = numpy.array(list(nx.non_edges(self.graph)))
        self.device = device
        self.dataset = dataset
        self.method = 'r'
        numpy.random.shuffle(self.non_edges)
        numpy.random.shuffle(self.edges)

    def calculate(self):
        for i in numpy.arange(0.6,0.8,0.2):
            edges_new = numpy.array(self.edges,copy=True)
            numpy.random.shuffle(edges_new)
            edges_new = edges_new[:int(i*(len(edges_new)))]
            self.x_values = numpy.append(self.x_values,self.kfold (edges_new , self.nodes))
            self.y_values = numpy.append(self.y_values,i-1)
            self._print(i-1,self.x_values[-1])
        for i in numpy.arange(0,1.2,0.2):
            edges_new = numpy.array(self.edges,copy=True)
            edges_new = numpy.concatenate((edges_new,self.non_edges[0:int(len(edges_new)*i)]),axis=0)
            self.x_values = numpy.append(self.x_values,self.kfold (edges_new , self.nodes))
            self.y_values = numpy.append(self.y_values,i)
            self._print(i,self.x_values[-1])

    def kfold(self,edges,nodes):
        auc = 0
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(edges)
        for train_index, test_index in kf.split(edges):
                edges_train, edges_test = edges[train_index], edges[test_index]
                edges_train = list(tuple(map(tuple, edges_train)))
                graph = nx.Graph()
                graph.add_nodes_from(nodes)
                graph.add_edges_from(edges_train)
                non_exist_edges = list(nx.non_edges(graph))
                data = normalize(nx.adjacency_matrix(graph),graph.degree())
                feat_mat,role_mat = train(data,self.device,self.method)
                so = AUC_ROC(feat_mat,role_mat,self.adj)
                sim_mat = so.similarity()
                non_exist_edges = self.non_exist_sim(non_exist_edges,sim_mat)
                auc = auc + self.AUC(list(tuple(map(tuple, edges_test))),sim_mat,non_exist_edges)
        return (auc/10)

    def AUC(self,test_data,sim_mat,non_edges):
        great = 0
        equal = 0
        result = 0
        for m in range (len(test_data)):
            sim = sim_mat[test_data[m][0]][test_data[m][1]]
            for c in range (len(non_edges)):
                if sim > non_edges[c][2] :
                    great = great+1
                if sim == non_edges[c][2] :
                    equal = equal + 1
            result = (result + (great+ (0.5*equal)))/(len(test_data)*len(non_edges))
        return result

    def non_exist_sim(self,non_edges , result_mat):
        for k in range (len(non_edges)):
            non_edges[k]= list(non_edges[k])
            non_edges[k].append(result_mat[non_edges[k][0]][non_edges[k][1]])
        non_edges.sort(key=lambda x: x[2],reverse=True)
        return non_edges

    def plot(self,dataset):
        fig = plt.figure()
        fig.suptitle(dataset)
        plt.plot (self.y_values,self.x_values)
        fig.savefig('./output/robust/{}.jpg'.format(dataset),dpi=300)
        return self.x_values,self.y_values
    
    def _print (self,i,value):
        print ("AUC-ROC @ (ratio = {:.1f}): {:.4f}\n".format(i,value))
        


def save(dataset, args, method):
    file = open("./output/{}/{}.txt".format(method,dataset), "w")
    file.write(args)
    file.close()

    
    

    
