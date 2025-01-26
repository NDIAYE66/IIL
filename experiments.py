#!/usr/bin/env python
# coding: utf-8
import copy
import os
import time
import argparse
import networkx as nx
import numpy as np
import torch.nn as nn


import torch
import torch.nn.functional as F
import torch.optim as optim

from math import ceil

import itertools
import pandas as pd


from utils import generate_new_features, generate_new_batches, AverageMeter, generate_batches_lstm, read_meta_datasets, \
    preprocess_gradients
from models import MPNN_LSTM, LSTM, MPNN, MPNet, arima, BiLSTM
from meta_LSTM import Inner_learner, Metalearner
        

    
def train(epoch, adj, features, y, inner_out):
    optimizer.zero_grad()
    output = model(adj, features, inner_out)
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train



def test(adj, features, y, inner_out):
    output = model(adj, features, inner_out)
    loss_test = F.mse_loss(output, y)
    return output, loss_test


def train_inner_learner(inner_learner_w_grad, metaLearner, adj_train_inner, features_train_inner, y_train_inner, args, n_train_batches):
    cI = metaLearner.metalstm.cI.data
    hs = [None]

    best_loss = 1e8
    stop = False
    for i in range(args.inner_epochs):
        for batch in range(n_train_batches):
            # for x, y in zip(meta_feat_train[batch], meta_y_train[batch]):
            # get the loss/grad

            inner_learner_w_grad.copy_flat_params(cI)
            # print(features_train_inner[batch].shape)
            output = inner_learner_w_grad(adj_train_inner[batch],
                                          features_train_inner[batch])  # put the right input.
            # print(f'Train output{output.shape}')
            # print(f"y shape {y_train_inner[batch].shape}")
            # print(f'Train y_train{y_train_inner.shape}')
            loss = F.mse_loss(output, y_train_inner[batch])
            # print(f"Inner learner loss: {loss}")

            inner_learner_w_grad.zero_grad()
            loss.backward()
            # optimizer.step()

            # Either divided by batch_size or by the lenght of the training input
            grad = torch.cat([p.grad.data.view(-1) / args.batch_size for p in inner_learner_w_grad.parameters()], 0)

            # process grad and loss and metalearner
            grad_prep = preprocess_gradients(grad)  # [n_learner_params, 2]
            loss_prep = preprocess_gradients(loss.data.unsqueeze(0))  # [1, 2]
            metalearner_input = [loss_prep, grad_prep, grad.unsqueeze(1)]
            cI, h = metaLearner(metalearner_input, hs[-1])
            hs.append(h)
        # if (loss < best_loss):
        #     best_loss = loss
        #     torch.save({
        #         'state_dict':inner_learner_w_grad.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, 'model_best.pth.tar')

        if (i % 50 == 0):
            print(f'epoch: {i} loss: {loss} ')


    return cI

def meta_test(inner_learner_w_grad, inner_learner_wo_grad, metaLearner, gs_adj_inner, features_inner, y_inner, idx_train, idx_val, shift, device, test_sample, args, n_train_batch):
    adj_train_inner, features_train_inner, y_train_inner = generate_new_batches(gs_adj_inner, features_inner, y_inner,
                                                                                idx_train,
                                                                                args.graph_window, shift,
                                                                                args.batch_size, device, test_sample)
    adj_val_inner, features_val_inner, y_val_inner = generate_new_batches(gs_adj_inner, features_inner, y_inner,
                                                                          idx_val,
                                                                          args.graph_window, shift, args.batch_size,
                                                                          device, test_sample)

    adj_test_inner, features_test_inner, y_test_inner = generate_new_batches(gs_adj_inner, features_inner, y_inner,
                                                                             [test_sample],
                                                                             args.graph_window, shift,
                                                                             args.batch_size, device, test_sample)

    # train the learner with metalearner
    inner_learner_w_grad.reset_batch_stats()
    inner_learner_wo_grad.reset_batch_stats()
    inner_learner_w_grad.train()
    inner_learner_wo_grad.eval()
    cI = train_inner_learner(inner_learner_w_grad, metaLearner, adj_val_inner, features_val_inner, y_val_inner, args,
                             n_train_batch)
    inner_learner_wo_grad.transfer_params(inner_learner_w_grad, cI)
    output = inner_learner_wo_grad(adj_test_inner, features_test_inner)
    loss = F.mse_loss(output, y_test_inner)


    return output

def inner_test(inner_wo_grad, adj_test_inner, features_test_inner, y_test_inner):

    inner_wo_grad.reset_batch_stats()
    inner_wo_grad.eval()
    output = inner_wo_grad(adj_test_inner[0], features_test_inner[0])
    loss = F.mse_loss(output, y_test_inner[0])

    return output, loss











if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epoch')
    parser.add_argument('--inner_epochs', type=int, default=300,
                        help='Number of epoch of the inner learner')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help='Learning rate for the Meta LSTM.')
    parser.add_argument('--hidden', type=int, default=64,
                        help='Hidden layer')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch_size')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout')
    parser.add_argument('--window', type=int, default=7,
                        help='window')
    parser.add_argument('--graph-window', type=int, default=7,
                        help='MPNN_LSTM window')
    parser.add_argument('--recur',  default=False,
                        help='True or False.')
    parser.add_argument('--early-stop', type=int, default=100,
                        help='Early stop')
    parser.add_argument('--start-exp', type=int, default=2,
                        help='First day of Prediction')
    parser.add_argument('--ahead', type=int, default=1,
                        help='Prediction ahead')
    parser.add_argument('--sep', type=int, default=10,
                        help='Separation of training set and validation set')
    parser.add_argument('--grad_clip', type=float,
                        default=2.0, help='Clip gradients larger than this number.')
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
    
    
    meta_labs,meta_labs_inner, meta_graphs, meta_graphs_inner, meta_features,meta_features_inner, meta_y, meta_y_inner = read_meta_datasets(args.window)
    
    State = "FL"
    idx = 0

    labels = meta_labs[idx]
    labels_inner = meta_labs_inner[idx]
    gs_adj = meta_graphs[idx]
    gs_adj_inner = meta_graphs_inner[idx]
    features = meta_features[idx]
    features_inner = meta_features_inner[idx]
    y = meta_y[idx]
    y_inner = meta_y_inner[idx]
    n_samples = len(gs_adj) - 71
    n_samples_inner = len(gs_adj_inner)
    nfeat = meta_features[0][0].shape[1]
    nfeat_inner = meta_features_inner[0][0].shape[1]

    n_nodes = gs_adj[0].shape[0]
    n_nodes_inner = gs_adj_inner[0].shape[0]
    print(n_nodes)
    print(n_nodes_inner)
    #print(f'Inner feat {nfeat_inner}')
    #print(f'Inter feat {nfeat}')
    if not os.path.exists('../Covid_Pred/results'):
        os.makedirs('../Covid_Pred/results')
    fw = open("../Covid_Pred/results/results_" + State + ".csv", "a")

    for args.model in ["AVG-WINDOW", "ARIMA", "Gompert","AVG_WINDOW","LSTM","BiLSTM","TGNN"]:  # "MPNN",,"MPNet"


        # if(args.model=="PROPHET"):
        #
        #     error, var = prophet(args.ahead,args.start_exp,n_samples,labels)
        #     count = len(range(args.start_exp,n_samples-args.ahead))
        #     for idx,e in enumerate(error):
        #         #fw.write(args.model+","+str(shift)+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")
        #         fw.write("PROPHET,"+str(idx)+",{:.5f}".format(e/(count*n_nodes))+",{:.5f}".format(np.std(var[idx]))+"\n")
        #     continue

        if (args.model == "ARIMA"):

            error, var = arima(args.ahead, args.start_exp, n_samples, labels)
            count = len(range(args.start_exp, n_samples - args.ahead))

            for idx, e in enumerate(error):
                fw.write("ARIMA," + str(idx) + ",{:.5f}".format(e / (count * n_nodes)) + ",{:.5f}".format(
                    np.std(var[idx])) + "\n")
            continue

        for shift in list(range(0, args.ahead)):

            result = []
            exp = 0

            for test_sample in range(args.start_exp, n_samples - shift):  #
                exp += 1
                print(test_sample)

                # 定义数据的分割
                idx_train = list(range(args.window - 1, test_sample - args.sep))

                idx_val = list(range(test_sample - args.sep, test_sample, 2))

                idx_train = idx_train + list(range(test_sample - args.sep + 1, test_sample, 2))

                # 模型

                if (args.model == "Gompert"):
                    win_lab = labels.iloc[:, test_sample - 1]
                    # print(win_lab[1])
                    targets_lab = labels.iloc[:, test_sample + shift]  #:(test_sample+1)]
                    error = np.sum(abs(win_lab - targets_lab)) / n_nodes  # /avg)
                    if (not np.isnan(error)):
                        result.append(error)
                    else:
                        exp -= 1
                    continue

                if (args.model == "AVG_WINDOW"):
                    win_lab = labels.iloc[:, (test_sample - args.window):test_sample]
                    targets_lab = labels.iloc[:, test_sample + shift]  #:
                    error = np.sum(abs(win_lab.mean(1) - targets_lab)) / n_nodes
                    if (not np.isnan(error)):
                        result.append(error)
                    else:
                        exp -= 1
                    continue

                if (args.model == "LSTM"):
                    lstm_features = 1 * n_nodes
                    adj_train, features_train, y_train = generate_batches_lstm(n_nodes, y, idx_train, args.window,
                                                                               shift, args.batch_size, device,
                                                                               test_sample)
                    adj_val, features_val, y_val = generate_batches_lstm(n_nodes, y, idx_train, args.window, shift,
                                                                         args.batch_size, device, test_sample)
                    adj_test, features_test, y_test = generate_batches_lstm(n_nodes, y, [test_sample], args.window,
                                                                            shift, args.batch_size, device, test_sample)


                elif (args.model == "MPNN_LSTM"):
                    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train,
                                                                              args.graph_window, shift, args.batch_size,
                                                                              device, test_sample)
                    adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, args.graph_window,
                                                                        shift, args.batch_size, device, test_sample)
                    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [test_sample],
                                                                           args.graph_window, shift, args.batch_size,
                                                                           device, test_sample)

                elif (args.model == "MPNet"):
                    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train,
                                                                              args.graph_window, shift,
                                                                              args.batch_size, device, test_sample)
                    adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val,
                                                                        args.graph_window, shift, args.batch_size,
                                                                        device, test_sample)
                    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [test_sample],
                                                                           args.graph_window, shift,
                                                                           args.batch_size, device, test_sample)
                    print(f"Generate Inner data: ")
                    adj_train_inner, features_train_inner, y_train_inner = generate_new_batches(gs_adj_inner,
                                                                                                features_inner, y_inner,
                                                                                                idx_train,
                                                                                                args.graph_window,
                                                                                                shift,
                                                                                                args.batch_size, device,
                                                                                                test_sample)
                    adj_val_inner, features_val_inner, y_val_inner = generate_new_batches(gs_adj_inner, features_inner,
                                                                                          y_inner, idx_val,
                                                                                          args.graph_window, shift,
                                                                                          args.batch_size,
                                                                                          device, test_sample)

                    adj_test_inner, features_test_inner, y_test_inner = generate_new_batches(gs_adj_inner,
                                                                                             features_inner, y_inner,
                                                                                             [test_sample],
                                                                                             args.graph_window, shift,
                                                                                             args.batch_size, device,
                                                                                             test_sample)

                    inner_w_grad = Inner_learner(c_feat=nfeat_inner, c_hid=args.hidden, num_c=n_nodes_inner,
                                                 dropout=args.dropout, window=args.window).to(device)
                    inner_wo_grad = copy.deepcopy(inner_w_grad)
                    metaLearner = Metalearner(c_feat=4, c_hid=64,
                                              n_learner_params=inner_w_grad.get_flat_params().size(0)).to(device)
                    metaLearner.metalstm.init_cI(inner_w_grad.get_flat_params())
                    meta_optim = torch.optim.Adam(metaLearner.parameters(), args.meta_lr)


                else:
                    adj_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, 1, shift,
                                                                              args.batch_size, device, test_sample)
                    adj_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, 1, shift,
                                                                        args.batch_size, device, test_sample)
                    adj_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [test_sample], 1, shift,
                                                                           args.batch_size, device, -1)

                n_train_batches = ceil(len(idx_train) / args.batch_size)
                n_val_batches = 1
                n_test_batches = 1

                # 训练模块
                # 模型和优化器
                stop = False  #
                while (not stop):  #
                    if (args.model == "LSTM"):
                        print(f'LSTM model train')
                        model = LSTM(nfeat=lstm_features, nhid=args.hidden, n_nodes=n_nodes, window=args.window,
                                     dropout=args.dropout, batch_size=args.batch_size, recur=args.recur).to(device)

                    if (args.model == "BiLSTM"):
                        print(f'BiLSTM model train')
                        model = BiLSTM(nfeat=lstm_features, nhid=args.hidden, n_nodes=n_nodes, window=args.window,
                                     dropout=args.dropout, batch_size=args.batch_size, recur=args.recur).to(device)

                    elif (args.model == "MPNN"):
                        print(f'MPNN model train')
                        model = MPNN(nfeat=nfeat, nhid=args.hidden, nout=1, dropout=args.dropout).to(device)

                    elif (args.model == "MPNN_LSTM"):
                        print(f'MPNN_LSTM model train')
                        model = MPNN_LSTM(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes,
                                          window=args.graph_window, dropout=args.dropout).to(device)

                    elif (args.model == "MPNet"):
                        # print(f'MPNet model train')

                        model = MPNet(nfeat=nfeat, nhid=args.hidden, nout=1, n_nodes=n_nodes, window=args.graph_window,
                                      dropout=args.dropout, nfeat_inner=nfeat_inner).to(device)
                        cI = train_inner_learner(inner_learner_w_grad=inner_w_grad, metaLearner=metaLearner,
                                                 adj_train_inner=adj_train_inner,
                                                 features_train_inner=features_train_inner,
                                                 y_train_inner=y_train_inner, args=args,
                                                 n_train_batches=n_train_batches)

                    optimizer = optim.Adam(model.parameters(), lr=args.lr)
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

                    # 训练
                    best_val_acc = 1e8
                    val_among_epochs = []
                    train_among_epochs = []
                    stop = False

                    print('Inner city module training')


                    for epoch in range(args.epochs):
                        start = time.time()

                        model.train()
                        train_loss = AverageMeter()
                        inner_train_out = 0
                        if (args.model == "MPNet"):
                            inner_w_grad.reset_batch_stats()
                            inner_wo_grad.reset_batch_stats()
                            inner_w_grad.train()
                            inner_wo_grad.eval()

                            inner_wo_grad.transfer_params(inner_w_grad, cI)

                            for batch in range(n_train_batches):
                                output = inner_wo_grad(adj_val_inner[0], features_val_inner[0])
                                loss_inner = F.mse_loss(output, y_val_inner[0])

                                meta_optim.zero_grad()
                                loss_inner.backward()
                                # nn.utils().clip_grad_norm(metaLearner.parameters(), args.grad_clip)
                                meta_optim.step()

                            inner_train_out, loss_train_inner = inner_test(inner_wo_grad, adj_train_inner,
                                                                           features_train_inner,
                                                                           y_train_inner)

                            tensor = torch.zeros((n_nodes - n_nodes_inner) * 5)

                            inner_train_out = torch.cat((inner_train_out, tensor), dim=-1)

                            print(f'Inner module train loss: {loss_train_inner}')

                        # 训练一次
                        for batch in range(n_train_batches):
                            print(f"Inter city module training")
                            output, loss = train(epoch, adj_train[batch], features_train[batch], y_train[batch],
                                                 inner_train_out)
                            train_loss.update(loss.data.item(), output.size(0))

                        # 对验证集进行评估
                        model.eval()

                        # for i in range(n_val_batches):
                        inner_out_val = 0
                        if(args.model == "MPNet"):
                            tensor = torch.zeros((n_nodes - n_nodes_inner) * 5)
                            inner_out_val, loss_val_inner = inner_test(inner_wo_grad, adj_val_inner, features_val_inner,
                                                       y_val_inner)
                            inner_out_val = torch.cat((inner_out_val, tensor), dim=-1)
                            print(f'Inner val loss: {loss_val_inner}')
                        output, val_loss = test(adj_val[0], features_val[0], y_val[0], inner_out_val)
                        val_loss = float(val_loss.detach().cpu().numpy())

                        # 输出结果
                        if (epoch % 50 == 0):
                            # print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg), "time=", "{:.5f}".format(time.time() - start))
                            print("Epoch:", '%03d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss.avg),
                                  "val_loss=", "{:.5f}".format(val_loss), "time=", "{:.5f}".format(time.time() - start))

                        train_among_epochs.append(train_loss.avg)
                        val_among_epochs.append(val_loss)

                        # print(int(val_loss.detach().cpu().numpy()))

                        if (epoch < 30 and epoch > 10):
                            if (len(set([round(val_e) for val_e in val_among_epochs[-20:]])) == 1):
                                # stuck= True
                                stop = False
                                break

                        if (epoch > args.early_stop):
                            if (len(set([round(val_e) for val_e in val_among_epochs[-50:]])) == 1):  #
                                print("break")
                                # stop = True
                                break

                        stop = True

                        # 记住最佳精度并保存检查点
                        if val_loss < best_val_acc:
                            best_val_acc = val_loss
                            torch.save({
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                            }, 'model_best.pth.tar')

                        scheduler.step(val_loss)

                print("validation")
                # print(best_val_acc)
                # 测试
                test_loss = AverageMeter()

                # print("Loading checkpoint!")
                checkpoint = torch.load('model_best.pth.tar')
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                model.eval()

                # error= 0
                # for batch in range(n_test_batches):
                inner_out = 0
                if(args.model == "MPNet"):
                    tensor = torch.zeros((n_nodes - n_nodes_inner))
                    inner_out, loss_test_inner = inner_test(inner_wo_grad, adj_test_inner, features_test_inner,
                                                            y_test_inner)
                    inner_out = torch.cat((inner_out, tensor), dim=-1)
                    print(f"Inner test loss: {loss_test_inner}")
                output, loss = test(adj_test[0], features_test[0], y_test[0], inner_out)

                if (args.model == "LSTM"):
                    o = output.view(-1).cpu().detach().numpy()
                    l = y_test[0].view(-1).cpu().numpy()
                else:
                    o = output.cpu().detach().numpy()
                    l = y_test[0].cpu().numpy()

                # 每个区域的平均误差
                error = np.sum(abs(o - l)) / n_nodes

                # 输出结果
                print("test error=", "{:.5f}".format(error))
                result.append(error)

            print("{:.5f}".format(np.mean(result)) + ",{:.5f}".format(np.std(result)) + ",{:.5f}".format(
                np.sum(labels.iloc[:, args.start_exp:test_sample].mean(1))))

            fw.write(str(args.model) + "," + str(shift) + ",{:.5f}".format(np.mean(result)) + ",{:.5f}".format(
                np.std(result)) + "\n")
            # fw.write(hypers+",{:.5f}".format(np.mean(result))+",{:.5f}".format(np.std(result))+"\n")

fw.close()



