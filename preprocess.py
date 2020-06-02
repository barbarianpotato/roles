import torch
import networkx as nx 
import numpy as np

def import_data(dataset):
    h=nx.read_gml(dataset,label='id')
    h.remove_edges_from(nx.selfloop_edges(h))
    adj= nx.adjacency_matrix(h)
    n=adj.shape[0]
    adj=adj.todense()
    x=h.degree()
    degree_matrix = np.identity(n)
    for i in range(1,n):
        if x[i] != 0:
            degree_matrix[i-1,i-1] = 1/(x[i]**0.5)
    adj = adj+np.identity(n)
    adj_norm = np.matmul(adj,degree_matrix)
    adj_norm = np.matmul(degree_matrix,adj)
    return [torch.from_numpy(adj_norm).to_sparse(),n],np.array(adj)

def normalize (adj,h):
    n=adj.shape[0]
    adj=adj.todense()
    x=h
    degree_matrix = np.identity(n)
    for i in range(1,n):
        if x[i] != 0:
            degree_matrix[i-1,i-1] = 1/(x[i]**0.5)
    adj = adj+np.identity(n)
    adj_norm = np.matmul(adj,degree_matrix)
    adj_norm = np.matmul(degree_matrix,adj)
    return torch.from_numpy(adj_norm).to_sparse(),n
