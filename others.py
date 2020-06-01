import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from scipy import spatial
from train import deep_test
from sklearn.model_selection import KFold
import matplotlib
import matplotlib.pyplot as plt
import csv
import time
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from preprocess import process
import pickle

np.random.seed(0)
def normalize (n):
    for i in n:
        max = np.amax(i)
        if max > 0 :
            for j in range(len(i)):
                i[j] = i[j] / max
    return  n

def similarity (feature_mat):
    result_mat = []
    for k in range (len(feature_mat)):
        result_mat.append([])
        for j in range (len(feature_mat)):
            result_mat[k].append(1-spatial.distance.cosine(feature_mat[k], feature_mat[j]))
    return result_mat

def non_existing_similarity (non_edges , result_mat):
    for k in range (len(non_edges)):
        non_edges[k]= list(non_edges[k])
        non_edges[k].append(result_mat[non_edges[k][0]][non_edges[k][1]])
    non_edges.sort(key=lambda x: x[2],reverse=True)
    return non_edges

def precision (testdata, non_edges, length):
    non_edges=non_edges[0:length]
    for k in range (len(non_edges)):
        non_edges[k].pop()
        non_edges[k]=tuple(non_edges[k])
    prec=len(set(testdata).intersection(set(non_edges)))/len(set(testdata))
    return prec

def AUC (test_data,feature_mat,non_edges):
    great = 0
    equal = 0
    result = 0
    for m in range (len(test_data)):
        sim = 1 - spatial.distance.cosine(feature_mat[test_data[m][0]], feature_mat[test_data[m][1]])
        for c in range (len(non_edges)):
            if sim > non_edges[c][2] :
                great = great+1
            if sim == non_edges[c][2] :
                equal = equal + 1
        result = (result + (great+ (0.5*equal)))/(len(test_data)*len(non_edges))
    return result



def k_fold_AUC (adj):
    auc = 0
    X = np.array(list(nx.Graph(adj).edges))
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    nodes = list (range(len(adj)))
    for train_index, test_index in kf.split(X):
        print ("lol")
        X_train, X_test = X[train_index], X[test_index]
        X_train = list(tuple(map(tuple, X_train)))
        Graph = nx.Graph()
        Graph.add_nodes_from(nodes)
        Graph.add_edges_from(X_train)
        non_existing_edges = list(nx.non_edges(Graph))
        train_graph = nx.adjacency_matrix(Graph).todense()
        feature_mat=deep_test(train_graph,[16,8],5000,1)
        feature_mat=normalize(feature_mat)
        result_mat = similarity(feature_mat)
        non_existing_edges = non_existing_similarity(non_existing_edges,result_mat)
        # print (non_existing_edges)
        auc = auc + AUC(list(tuple(map(tuple, X_test))),feature_mat,non_existing_edges)
    print ("AUC : ", auc/10)


def k_fold_robust (X,nodes):
    auc = 0
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            X_train = list(tuple(map(tuple, X_train)))
            Graph = nx.Graph()
            Graph.add_nodes_from(nodes)
            Graph.add_edges_from(X_train)
            non_existing_edges = list(nx.non_edges(Graph))
            train_graph = nx.adjacency_matrix(Graph).todense()
            feature_mat=deep_test(train_graph,[16,8],5000,1)
            feature_mat=normalize(feature_mat)
            result_mat = similarity(feature_mat)
            non_existing_edges = non_existing_similarity(non_existing_edges,result_mat)
            auc = auc + AUC(list(tuple(map(tuple, X_test))),feature_mat,non_existing_edges)
    return (auc/10)


def robustness (adj,ds):
    y=[]
    x=[]
    Graph = nx.Graph(adj)
    edges = np.array(list(Graph.edges))
    nodes = list (range(len(adj)))
    np.random.shuffle(edges)
    et = edges
    nonedges = np.array(list(nx.non_edges(Graph)))
    np.random.shuffle(nonedges)
    for i in np.arange(0.6,0.8,0.2):
        etnew = np.array(et,copy=True)
        np.random.shuffle(etnew)
        etnew = etnew[:int(i*(len(etnew)))]
        x.append(k_fold_robust (etnew , nodes))
        y.append(i-1)
    for i in np.arange(0,1.2,0.2):
        etnew = np.array(et,copy=True)
        etnew = np.concatenate((etnew,nonedges[0:int(len(etnew)*i)]),axis=0)
        x.append(k_fold_robust (etnew , nodes))
        y.append(i)
    # print (x)
    # print (y)
    c = zip(y,x)
    with open (str(ds)+'result.csv',"a") as fp:
        wr = csv.writer(fp,dialect='excel')
        wr.writerow (['Ratio', 'AUC'])
        for i in c :
            wr.writerow(i)
    return x


def cost_graph (adj):
    cost = []
    features =[]
    for i in range(1,50):
        features.append(i)
        cost.append(deep_test(adj,[i,8],5000,0))
    plt.plot(features,cost)
    plt.show()
    
def save_csv (x,y):
    c = zip(y,x)
    with open (str(time.time())+'result.csv',"a") as fp:
        wr = csv.writer(fp,dialect='excel')
        wr.writerow (['Ratio', 'AUC'])
        for i in c :
            wr.writerow(i)


def avg_prec (edges,pred):
    edges = np.array(edges)
    edges_flat = edges.flatten()
    pred_flat = pred.flatten()
    auc = roc_auc_score(edges_flat,pred_flat)
    prec = average_precision_score(edges_flat,pred_flat)
    # print (auc)
    # print (prec)
    return (auc , prec)

def deep_prec (adj,adj2):
    Graph = nx.Graph(adj)
    edges = np.array(list(Graph.edges))
    nodes = list (range(len(adj)))
    g = nx.Graph()
    g.add_nodes_from(nodes)
    g.add_edges_from(edges)
    train_graph = nx.adjacency_matrix(g).todense()
    feature_mat=deep_test(train_graph,[16,8],5000,1)
    feature_mat=normalize(feature_mat)
    result_mat = similarity(feature_mat)
    auc = avg_prec(adj2,np.array(result_mat))
    return auc


def plotgraph (result,xaxis,name):
    graph = []
    fig = plt.figure()
    fig.suptitle(name)
    for i in result.keys():
        a ,= plt.plot (xaxis,result[i],c=np.random.rand(3,),marker='o',label=i)
        graph.append(a)
        pickle.dump(result[i],open('./plots/{}_{}.txt'.format(i,name),'w'))
    plt.title(name)
    plt.legend(handles=graph)
    fig.savefig('./plots/{}.jpg'.format(name),dpi=300)