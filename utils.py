import torch
import networkx as nx
import numpy as np
import scipy.sparse as sp
import pandas as pd
from math import ceil
import glob
import unidecode 
from datetime import date, timedelta
from sklearn import preprocessing
import os


    
def read_meta_datasets(window):
    """
    数据读取
    """
    os.chdir("../Covid_Pred/data")
    meta_labs = []
    meta_labs_inner = []
    meta_graphs = []
    meta_graphs_inner = []
    meta_features = []
    meta_features_inner = []
    meta_y = []
    meta_y_inner = []

    os.chdir("Florida")
    labels = pd.read_csv("florida_labels.csv")
    labels = labels.set_index("City")

    sdate = date(2020, 7, 1)
    edate = date(2020, 9, 30)
    delta = edate - sdate
    dates = [sdate + timedelta(days=i) for i in range(delta.days+1)]
    dates = [str(date) for date in dates]
    
    
    Gs, Gs_meta = generate_graphs_tmp(dates,"FL")

    labels_meta = labels.loc[list(Gs_meta[0].nodes()),:]
    labels_meta = labels_meta.loc[:,dates]

    labels = labels.loc[list(Gs[0].nodes()),:]
    labels = labels.loc[:,dates]


     
    meta_labs.append(labels)
    meta_labs_inner.append(labels_meta)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    gs_adj_meta = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs_meta]

    meta_graphs.append(gs_adj)
    meta_graphs_inner.append(gs_adj_meta)

    features = generate_new_features(Gs ,labels ,dates ,window )
    features_meta = generate_new_features(Gs_meta, labels_meta,dates, window)

    meta_features.append(features)
    meta_features_inner.append(features_meta)

    y = list()
    y_inner = list()
    for i,G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node,dates[i]])

    for i,G in enumerate(Gs_meta):
        y_inner.append(list())
        for node in G.nodes():
            y_inner[i].append(labels_meta.loc[node,dates[i]])

    meta_y.append(y)
    meta_y_inner.append(y_inner)

    Gs, Gs_meta = generate_graphs_tmp(dates, "IT")

    labels_meta = labels.loc[list(Gs_meta[0].nodes()), :]
    labels_meta = labels_meta.loc[:, dates]

    labels = labels.loc[list(Gs[0].nodes()), :]
    labels = labels.loc[:, dates]

    meta_labs.append(labels)
    meta_labs_inner.append(labels_meta)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    gs_adj_meta = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs_meta]

    meta_graphs.append(gs_adj)
    meta_graphs_inner.append(gs_adj_meta)

    features = generate_new_features(Gs, labels, dates, window)
    features_meta = generate_new_features(Gs_meta, labels_meta, dates, window)

    meta_features.append(features)
    meta_features_inner.append(features_meta)

    y = list()
    y_inner = list()
    for i, G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node, dates[i]])

    for i, G in enumerate(Gs_meta):
        y_inner.append(list())
        for node in G.nodes():
            y_inner[i].append(labels_meta.loc[node, dates[i]])

    meta_y.append(y)
    meta_y_inner.append(y_inner)

    Gs, Gs_meta = generate_graphs_tmp(dates, "EN")

    labels_meta = labels.loc[list(Gs_meta[0].nodes()), :]
    labels_meta = labels_meta.loc[:, dates]

    labels = labels.loc[list(Gs[0].nodes()), :]
    labels = labels.loc[:, dates]

    meta_labs.append(labels)
    meta_labs_inner.append(labels_meta)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    gs_adj_meta = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs_meta]

    meta_graphs.append(gs_adj)
    meta_graphs_inner.append(gs_adj_meta)

    features = generate_new_features(Gs, labels, dates, window)
    features_meta = generate_new_features(Gs_meta, labels_meta, dates, window)

    meta_features.append(features)
    meta_features_inner.append(features_meta)

    y = list()
    y_inner = list()
    for i, G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node, dates[i]])

    for i, G in enumerate(Gs_meta):
        y_inner.append(list())
        for node in G.nodes():
            y_inner[i].append(labels_meta.loc[node, dates[i]])

    meta_y.append(y)
    meta_y_inner.append(y_inner)

    Gs, Gs_meta = generate_graphs_tmp(dates, "PA")

    labels_meta = labels.loc[list(Gs_meta[0].nodes()), :]
    labels_meta = labels_meta.loc[:, dates]

    labels = labels.loc[list(Gs[0].nodes()), :]
    labels = labels.loc[:, dates]

    meta_labs.append(labels)
    meta_labs_inner.append(labels_meta)
    gs_adj = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs]

    gs_adj_meta = [nx.adjacency_matrix(kgs).toarray().T for kgs in Gs_meta]

    meta_graphs.append(gs_adj)
    meta_graphs_inner.append(gs_adj_meta)

    features = generate_new_features(Gs, labels, dates, window)
    features_meta = generate_new_features(Gs_meta, labels_meta, dates, window)

    meta_features.append(features)
    meta_features_inner.append(features_meta)

    y = list()
    y_inner = list()
    for i, G in enumerate(Gs):
        y.append(list())
        for node in G.nodes():
            y[i].append(labels.loc[node, dates[i]])

    for i, G in enumerate(Gs_meta):
        y_inner.append(list())
        for node in G.nodes():
            y_inner[i].append(labels_meta.loc[node, dates[i]])

    meta_y.append(y)
    meta_y_inner.append(y_inner)



    # os.chdir("../Covid_Pred")

    return meta_labs, meta_labs_inner ,meta_graphs, meta_graphs_inner,meta_features, meta_features_inner,meta_y, meta_y_inner
    
    

def generate_graphs_tmp(dates,country):
    """
    将城市节点添加到图内
    """
    Gs = []
    Gs_meta = []
    for date in dates:
        d = pd.read_csv("graphs/"+"Graph-geoid-all"+"-"+date+".csv")
        d_meta = d.loc[d['City_o'] == d['City_d']]
        # print(f'Graph for inner: {d_meta}')
        G = nx.DiGraph()
        Gm = nx.DiGraph()
        nodes = set(d.iloc[:,0].unique()).union(set(d.iloc[:,1].unique()))
        nodes_meta = set(d_meta.iloc[:,0].unique()).union(set(d_meta.iloc[:,1].unique()))
        G.add_nodes_from(nodes)
        Gm.add_nodes_from(nodes_meta)



        for row in d.iterrows():
            G.add_edge(row[1][0], row[1][1], weight=row[1][2])
        for row in d_meta.iterrows():
            Gm.add_edge(row[1][0], row[1][1], weight=row[1][2])
        Gs.append(G)
        Gs_meta.append(Gm)

    return Gs, Gs_meta



def generate_new_features(Gs, labels, dates, window=3, scaled=False):
    """
    生成节点特征
    Features[1]包含与y[1]对应的特征
    例如，如果窗口=7 特征[7]=day0:day6 y[7]=day7
    """
    features = list()
    
    labs = labels.copy()
    nodes = Gs[0].nodes()
  
    for idx,G in enumerate(Gs):
        #  特征=人口、地点、d个过去的案例、热点区域
        
        H = np.zeros([G.number_of_nodes(),window]) 
        me = labs.loc[:, dates[:(idx)]].mean(1)
        sd = labs.loc[:, dates[:(idx)]].std(1)+1

        for i,node in enumerate(G.nodes()):
            # 过去案例   
            if(idx < window):
                if(scaled):
                    H[i,(window-idx):(window)] = (labs.loc[node, dates[0:(idx)]] - me[node])/ sd[node]
                else:
                    print(labs.loc[node, dates[0:(idx)]])
                    H[i,(window-idx):(window)] = labs.loc[node, dates[0:(idx)]]

            elif idx >= window:
                if(scaled):
                    H[i,0:(window)] =  (labs.loc[node, dates[(idx-window):(idx)]] - me[node])/ sd[node]
                else:
                    H[i,0:(window)] = labs.loc[node, dates[(idx-window):(idx)]]
        features.append(H)
        
    return features



def generate_new_batches(Gs, features, y, idx, graph_window, shift, batch_size, device, test_sample):
    """
    为MPNN的图生成批次
    """

    N = len(idx)
    n_nodes = Gs[0].shape[0]
  
    adj_lst = list()
    features_lst = list()
    y_lst = list()

    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*graph_window*n_nodes
        step = n_nodes*graph_window

        adj_tmp = list()
        features_tmp = np.zeros((n_nodes_batch, features[0].shape[1]))

        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)

        #为每个批次填写输入
        for e1,j in enumerate(range(i, min(i+batch_size, N) )):
            val = idx[j]

            # Feature[10] 包含了 y[10] 的前七个案例
            for e2,k in enumerate(range(val-graph_window+1,val+1)):
                
                adj_tmp.append(Gs[k-1].T)  
                features_tmp[(e1*step+e2*n_nodes):(e1*step+(e2+1)*n_nodes),:] = features[k]
            
            
            if(test_sample>0):
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                    
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
                        
            else:
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
        
        adj_tmp = sp.block_diag(adj_tmp)
        adj_lst.append(sparse_mx_to_torch_sparse_tensor(adj_tmp).to(device))
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append(torch.FloatTensor(y_tmp).to(device))

    return adj_lst, features_lst, y_lst



def generate_batches_lstm(n_nodes, y, idx, window, shift, batch_size, device,test_sample):
    """
    为LSTM的图生成批次
    """
    N = len(idx)
    features_lst = list()
    y_lst = list()
    adj_fake = list()
    
    for i in range(0, N, batch_size):
        n_nodes_batch = (min(i+batch_size, N)-i)*n_nodes*1
        step = n_nodes*1

        adj_tmp = list()
        features_tmp = np.zeros((window, n_nodes_batch))
        
        y_tmp = np.zeros((min(i+batch_size, N)-i)*n_nodes)
        
        for e1,j in enumerate(range(i, min(i+batch_size, N))):
            val = idx[j]
            
            # 从val-window到val-1保留过去的信息
            for e2,k in enumerate(range(val-window,val)):
               
                if(k==0): 
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.zeros([n_nodes])
                else:
                    features_tmp[e2, (e1*step):(e1*step+n_nodes)] = np.array(y[k])

            if(test_sample>0):
                if(val+shift<test_sample):
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]
                else:
                    y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val]
                        
            else:
         
                y_tmp[(n_nodes*e1):(n_nodes*(e1+1))] = y[val+shift]       
         
        adj_fake.append(0)
        
        features_lst.append(torch.FloatTensor(features_tmp).to(device))
        y_lst.append( torch.FloatTensor(y_tmp).to(device))
        
    return adj_fake, features_lst, y_lst



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy稀疏矩阵转换为Torch稀疏张量
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



class AverageMeter(object):
    """
    计算并存储平均值和当前值
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        



def preprocess_gradients(x):
    p = 10
    indicator = (x.abs() >= np.exp(-p)).to(torch.float32)

    # preproc1
    x_proc1 = indicator * torch.log(x.abs() + 1e-8) / p + (1 - indicator) * -1
    # preproc2
    x_proc2 = indicator * torch.sign(x) + (1 - indicator) * np.exp(p) * x
    return torch.stack((x_proc1, x_proc2), 1)