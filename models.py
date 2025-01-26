import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import networkx as nx
import numpy as np
import scipy.sparse as sp
from statsmodels.tsa.arima_model import ARIMA


from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

def arima(ahead,start_exp,n_samples,labels):
    var = []
    for idx in range(ahead):
        var.append([])

    error= np.zeros(ahead)
    count = 0
    for test_sample in range(start_exp,n_samples-ahead):#
        print(test_sample)
        count+=1
        err = 0
        for j in range(labels.shape[0]):
            ds = labels.iloc[j,:test_sample-1].reset_index()

            if(sum(ds.iloc[:,1])==0):
                yhat = [0]*(ahead)
            else:
                try:
                    fit2 = ARIMA(ds.iloc[:,1].values, (2, 0, 2)).fit()
                except:
                    fit2 = ARIMA(ds.iloc[:,1].values, (1, 0, 0)).fit()
                #yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-1) ))
                yhat = abs(fit2.predict(start = test_sample , end = (test_sample+ahead-2) ))
            y_me = labels.iloc[j,test_sample:test_sample+ahead]
            e =  abs(yhat - y_me.values)
            err += e
            error += e

        for idx in range(ahead):
            var[idx].append(err[idx])
    return error, var


class MPNet(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout, nfeat_inner):
        super(MPNet, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        self.nhid = nhid
        self.nfeat = nfeat

        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)

        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)

        self.lstm1 = nn.LSTM(2*nhid, nhid, 1)
        self.lstm2 = nn.LSTM(nhid, nhid, 1)

        self.fc1 = nn.Linear(2*nhid+window*nfeat+1, nhid) #bring back the input X to the fully connnected layer
        self.fc3 = nn.Linear(nhid, nout)
        self.fc4 = nn.Linear(nout, nout)



        # self.Inner_Inter = nn.Linear(nhid, inner_hid)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, adj, x, inner_out):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()

        skip = x.view(-1, self.window, self.n_nodes, self.nfeat) # batch_size
        # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1, self.window, self.nfeat) # batch_size * number of nodes

        x = self.relu(self.conv1(x, adj, edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)

        x = self.relu(self.conv2(x, adj, edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)

        x = torch.cat(lst, dim=1)

        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))# window * (number of cases * cities(nodes))

        x, (hn1, cn1) = self.lstm1(x)
        out2, (hn2, cn2) = self.lstm2(x)

        x = torch.cat([hn1[0, :, :], hn2[0, :, :]], dim=1)
        inner_out = torch.unsqueeze(inner_out, dim=1)
        x = torch.cat([x, inner_out], dim=1)
        skip = skip.reshape(skip.size(0), -1)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.fc1(x))

        x = self.dropout(x)

        x = self.relu(self.fc3(x)).squeeze()

        x = x.view(-1)

        return x

            
            
class MPNN_LSTM(nn.Module):
    def __init__(self, nfeat, nhid, nout, n_nodes, window, dropout):
        super(MPNN_LSTM, self).__init__()
        self.window = window
        self.n_nodes = n_nodes
        #self.batch_size = batch_size
        self.nhid = nhid
        self.nfeat = nfeat
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.rnn1 = nn.LSTM(2*nhid, nhid, 1)
        self.rnn2 = nn.LSTM(nhid, nhid, 1)
        
        #self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc1 = nn.Linear(2*nhid+window*nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nout)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        
    def forward(self, adj, x, inner_out=0):
        lst = list()
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       # print(x.shape)
        skip = x.view(-1,self.window,self.n_nodes,self.nfeat)#self.batch_size
       # print(skip.shape)
        skip = torch.transpose(skip, 1, 2).reshape(-1,self.window,self.nfeat)#self.batch_size*self.n_nodes
        
        x = self.relu(self.conv1(x, adj,edge_weight=weight))
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = torch.cat(lst, dim=1)
        
        #print(x.shape)
        x = x.view(-1, self.window, self.n_nodes, x.size(1))
        #print(x.shape)
        #print(x.shape)
        x = torch.transpose(x, 0, 1)
        x = x.contiguous().view(self.window, -1, x.size(3))#self.batch_size*self.n_nodes
        
        #print(x.shape)
        x, (hn1, cn1) = self.rnn1(x)
        
        
        out2, (hn2,  cn2) = self.rnn2(x)
        
        #print(self.rnn2._all_weights)
        x = torch.cat([hn1[0,:,:],hn2[0,:,:]], dim=1)
        #print(skip.shape)
        #print(x.shape)
        #skip = skip.view(skip.size(0),-1)
        skip = skip.reshape(skip.size(0),-1)
        #print(x.shape)
        #print(skip.shape)
                
        x = torch.cat([x,skip], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x)).squeeze()
        x = x.view(-1)
        
        
        return x
 



class TGNN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(TGNN, self).__init__()
        #self.n_nodes = n_nodes
    
        #self.batch_size = batch_size
        self.nhid = nhid
        
        
        self.conv1 = GCNConv(nfeat, nhid)
        self.conv2 = GCNConv(nhid, nhid) 
        self.bn1 = nn.BatchNorm1d(nhid)
        self.bn2 = nn.BatchNorm1d(nhid)
        
        self.fc1 = nn.Linear(nfeat+2*nhid, nhid )
        self.fc2 = nn.Linear(nhid, nout)
        #self.bn3 = nn.BatchNorm1d(nhid)
        #self.bn4 = nn.BatchNorm1d(nhid)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        #nn.init.zeros_(self.conv1.weight)
        #nn.init.zeros_(self.conv2.weight)
        #nn.init.zeros_(self.fc1.weight)
        #nn.init.zeros_(self.fc2.weight)
        
        
    def forward(self, adj, x,inner_out=0):
        lst = list()
        #print(x.shape)
        #print(adj.shape)
        weight = adj.coalesce().values()
        adj = adj.coalesce().indices()
       
        #lst.append(ident)
        
        #x = x[:,mob_feats]
        #x = xt.index_select(1, mob_feats)
        lst.append(x)
        
        x = self.relu(self.conv1(x,adj,edge_weight=weight))
        #print(x)
        #print(x.shape)
        x = self.bn1(x)
        x = self.dropout(x)
        lst.append(x)
        
        x = self.relu(self.conv2(x, adj,edge_weight=weight))
        #print(x.shape)
        x = self.bn2(x)
        x = self.dropout(x)
        lst.append(x)
        
        
        x = torch.cat(lst, dim=1)
                                   
        x = self.relu(self.fc1(x))
        #x = self.bn3(x)
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x)).squeeze() # 
        #x = self.bn4(x)
        
        x = x.view(-1)
        
        return x

    
    
    
class LSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout,batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers= 2
        
        self.nfeat = nfeat 
        self.recur = recur
        self.batch_size = batch_size
        self.lstm = nn.LSTM(nfeat, self.nhid, num_layers=self.nb_layers)
    
        self.linear = nn.Linear(nhid, self.nout)
        # self.cell = ( nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))
      
        #self.hidden_cell = (torch.zeros(2,self.batch_size,self.nhid).to(device),torch.zeros(2,self.batch_size,self.nhid).to(device))
        #nn.Parameter(nn.init.xavier_uniform(torch.Tensor(self.nb_layers, self.batch_size, self.nhid).type(torch.FloatTensor).cuda()),requires_grad=True))
        
        
    def forward(self, adj, features,inner_out=0):
        #adj is 0 here
        #print(features.shape)
        features = features.view(self.window,-1, self.n_nodes)#.view(-1, self.window, self.n_nodes, self.nfeat)
        #print(features.shape)
        
        # if(self.recur):
        #     #print(features.shape)
        #     #self.hidden_cell =
        #     try:
        #         lstm_out, (hc,self.cell) = self.lstm(features,(torch.zeros(self.nb_layers,self.batch_size,self.nhid).cuda(),self.cell))
        #         # = (hc,cn)
        #     except:
        #         #hc = self.hidden_cell[0][:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
        #         hc = torch.zeros(self.nb_layers,features.shape[1],self.nhid).cuda()
        #         cn = self.cell[:,0:features.shape[1],:].contiguous().view(2,features.shape[1],self.nhid)
        #         lstm_out, (hc,cn) = self.lstm(features,(hc,cn))
        # else:
        #------------------
        lstm_out, (hc,cn) = self.lstm(features)#, self.hidden_cell)#self.hidden_cell
            
        predictions = self.linear(lstm_out)#.view(self.window,-1,self.n_nodes)#.view(self.batch_size,self.nhid))#)
        #print(predictions.shape)
        return predictions[-1].view(-1)


class BiLSTM(nn.Module):
    def __init__(self, nfeat, nhid, n_nodes, window, dropout, batch_size, recur):
        super().__init__()
        self.nhid = nhid
        self.n_nodes = n_nodes
        self.nout = n_nodes
        self.window = window
        self.nb_layers = 2  # Number of LSTM layers

        self.nfeat = nfeat
        self.recur = recur
        self.batch_size = batch_size

        # Define bidirectional LSTM
        self.lstm = nn.LSTM(
            nfeat,
            self.nhid,
            num_layers=self.nb_layers,
            bidirectional=True
        )

        self.linear = nn.Linear(2 * nhid, self.nout)

    def forward(self, adj, features, inner_out=0):

        features = features.view(self.window, -1, self.n_nodes)


        lstm_out, (hc, cn) = self.lstm(features)


        predictions = self.linear(lstm_out[-1])  # Use the last time step's output

        # Return predictions reshaped to match the expected output
        return predictions.view(-1)




