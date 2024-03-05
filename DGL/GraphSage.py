from enum import unique
from time import process_time_ns
import torch as torch
import dgl
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score

from torch.utils.tensorboard import SummaryWriter 
import os.path  
import json
import os

outFileFolder,acc_test,index_arr='','',[]

import torch

print(torch.cuda.is_available())
# define GraphSAGE
from dgl.nn.pytorch import SAGEConv
class GraphSAGE(nn.Module):
    def __init__(self, 
                 in_feats,
                 n_hidden, # hidden size也可以是一个list
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator):
        
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layer = nn.ModuleList()
        self.layer.append(SAGEConv(in_feats, n_hidden, aggregator))
        for i in range(1, n_layers - 1):
            self.layer.append(SAGEConv(n_hidden, n_hidden, aggregator))#kb   n_hidden[i-1],n_hidden[i]
        self.layer.append(SAGEConv(n_hidden, n_classes, aggregator))#kb   n_hidden[-1]
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
    
    def forward(self, blocks, feas):
        h = feas
        for i, (layer, block) in enumerate(zip(self.layer, blocks)):
            h = layer(block, h)
            if i != self.n_layers - 1:
                h = self.activation(h)
                h = self.dropout(h)
        
        return h
    def inference(self, my_net, val_nid, batch_s, num_worker, device):
        
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(self.n_layers)
        dataloader = dgl.dataloading.NodeDataLoader(
                    my_net,
                    val_nid,
                    sampler,
                    batch_size = batch_s,
                    shuffle=True,
                    drop_last=False,
                    num_workers=num_worker
                )
        
        ret = torch.zeros(my_net.num_nodes(), self.n_classes)
        # dgl.dataloading.pytorch.NodeDataLoader的3个参数
        # block_sampler：采样器，这个采样器是预先定义的，比如可以用dgl.dataloading.MultiLayerNeighborSampler([15, 10, 5])
        # 表示第一层为每个节点采样15个邻居，第二层采样10个邻居，第三层采样5个邻居
        # 采样器也有一些其它函数，可以参考官网文档
        for input_nodes, output_nodes, blocks in dataloader:
            h = blocks[0].srcdata['features'].to(device)
            for i, (layer, block) in enumerate(zip(self.layer, blocks)):
                block = block.int().to(device)
                h = layer(block, h)
                if i != self.n_layers - 1 :
                    h = self.activation(h)
                    h = self.dropout(h)
            ret[output_nodes] = h.cpu()
        return ret.to('cuda:0')

def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes.
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

def evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device):
    
    model.eval()
    with torch.no_grad():
        label_pred = model.inference(my_net, val_nid,  batch_s, num_worker, device)#
    model.train()
    label_val_pred=label_pred[val_mask]#
    label_valmask=labels[val_mask]#
    sum_pred_true=(torch.argmax(label_val_pred, dim=1)==label_valmask).float().sum()#
    acc=sum_pred_true/len(label_pred[val_mask])
    return acc,label_pred


import itertools

def Run(data, train_val_data, args, sample_size, learning_rate, device_num):
    
    if device_num >= 0:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    in_feats, n_classes, my_net, fea_para = data
    epochs,hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker ,weight_decay = args
    
    train_mask, val_mask, test_mask, train_nid, val_nid , test_nid= train_val_data

    nfeat = my_net.ndata['features']
    labels = my_net.ndata['label']
    sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_size)
    
    dataloader =dgl.dataloading.DataLoader(# dgl.dataloading.NodeDataLoader(
        my_net,
        train_nid,
        sampler,
        batch_size = batch_s,
        shuffle=True,
        drop_last=False,
        num_workers=num_worker#
    )
    model = GraphSAGE(in_feats, hidden_size, n_classes, n_layers, activation, dropout, aggregator)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=weight_decay)
    # begin train
    model.train()
    
    loss_fun = nn.CrossEntropyLoss()
    loss_fun.to(device)
    x_epoch,losses_train,losses_val,accs_train,accs_val=[],[],[],[],[]
    train_acc,train_label_pred =None,None
    for epoch in range(epochs):
        loss=0
        for batch, (input_nodes, output_nodes, block) in enumerate(dataloader):
            batch_feature, batch_label = load_subtensor(nfeat, labels, output_nodes, input_nodes, device)
            block = [block_.int().to(device) for block_ in block]
            model_pred = model(block, batch_feature)
            loss = loss_fun(model_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            val_acc,val_label_pred = evaluate(model, my_net, labels, val_nid, val_mask, batch_s, num_worker, device)
            val_loss = loss_fun(val_label_pred[val_nid], labels[val_nid])
            train_acc,train_label_pred = evaluate(model, my_net, labels, train_nid, train_mask, batch_s, num_worker, device)
            train_loss=loss_fun(train_label_pred[train_nid],labels[train_nid])
            print('Epoch %d | Train Loss: %0.4f | Train ACC: %.4f |Val Loss:%.4f | Val ACC: %.4f' % (epoch, train_loss.item(), train_acc.item(),val_loss.item(), val_acc.item()))
            x_epoch.append(epoch)
            accs_train.append(train_acc.cpu().item())
            accs_val.append(val_acc.cpu().item())
            losses_train.append(train_loss.cpu().item())
            losses_val.append(val_loss.cpu().item())
            # writer.add_scalar('acc/train', train_acc, global_step=epoch, walltime=None)
            # writer.add_scalar('acc/val', val_acc, global_step=epoch, walltime=None)
    # writer.close()

    acc_test_,test_label_pred_proba = evaluate(model, my_net, labels, test_nid, test_mask, batch_s, num_worker, device)
    print('Test ACC: %.4f' % (acc_test_.item()))
    acc_test  =str(round(acc_test_.item(),4))
    curves_metrics=[x_epoch,accs_train,accs_val,losses_train,losses_val]
    return model,curves_metrics,acc_test,F.softmax(test_label_pred_proba),str(round(train_acc.item(),4)) ,F.softmax(train_label_pred),F.softmax(val_label_pred) 

#split train validation data ,the remain are test data
def splitTrainValTestByCountRandom(y,lst_Num_singleCls_train,lst_Num_singleCls_val):
    SampleNum=len(y)    
    index_arr=[i for i in range(SampleNum)]#
    import random
    random.shuffle(index_arr)#
    numclass=len(np.unique(y))
    train_mask = torch.zeros(SampleNum).to(torch.bool)
    val_mask = torch.zeros(SampleNum).to(torch.bool)
    test_mask = torch.zeros(SampleNum).to(torch.bool)
    labels_Num_Train_dic={}
    labels_Num_Val_dic={}
    for i in range(numclass):
        labels_Num_Train_dic[i]=lst_Num_singleCls_train[i] #labels_Num_Train_dic[i]
        labels_Num_Val_dic[i]=lst_Num_singleCls_val[i] 
    for i in range(len(index_arr)): 
        index_i=index_arr[i]#
        y_i=y[index_i]
        if  labels_Num_Train_dic[y_i]>0:
            train_mask[index_i]=True
            labels_Num_Train_dic[y_i]=labels_Num_Train_dic[y_i]-1
        elif labels_Num_Val_dic[y_i]>0:
            val_mask[index_i]=True
            labels_Num_Val_dic[y_i]=labels_Num_Val_dic[y_i]-1
        else:
            test_mask[index_i]=True
    return train_mask,val_mask,test_mask,index_arr
    
#split train validation data ,the remain are test data
def splitTrainValTestByCount(y,lst_Num_singleCls_train,lst_Num_singleCls_val):
    SampleNum=len(y)    
    numclass=len(np.unique(y))
    train_mask = torch.zeros(SampleNum).to(torch.bool)
    val_mask = torch.zeros(SampleNum).to(torch.bool)
    test_mask = torch.zeros(SampleNum).to(torch.bool)
    labels_Num_Train_dic={}
    labels_Num_Val_dic={}
    for i in range(numclass):
        labels_Num_Train_dic[i]=lst_Num_singleCls_train[i] #
        labels_Num_Val_dic[i]=lst_Num_singleCls_val[i] 
    for i in range(SampleNum): 
        y_i=y[i]
        if  labels_Num_Train_dic[y_i]>0:
            train_mask[i]=True
            labels_Num_Train_dic[y_i]=labels_Num_Train_dic[y_i]-1
        elif labels_Num_Val_dic[y_i]>0:
            val_mask[i]=True
            labels_Num_Val_dic[y_i]=labels_Num_Val_dic[y_i]-1
        else:
            test_mask[i]=True
    return train_mask,val_mask,test_mask
 
#load data and contruct graph and slpit training validation test data
def LoaddataBuilding0324(Adjtxt_path,xfeaturetxt_path,ylabeltxt_path,FeaLst_item,lst_Num_singleCls_train,lst_Num_singleCls_val,
                         train_masktxt_path,val_masktxt_path,test_masktxt_path,oldTagID,newTagID):
    random_index_arr=None
    #adj
    Adjtxt=np.loadtxt(Adjtxt_path,dtype= np.float32,delimiter=' ',encoding='utf-8')
    Adjtxt=Adjtxt.squeeze()
    src=Adjtxt[0:,0:1]
    dst=Adjtxt[0:,1:2]
    src=src.squeeze()
    dst=dst.squeeze()

    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    my_net=dgl.graph((u, v))#DGLGraph((u, v))
    my_net=my_net.to('cuda:0')

    # node feature
    fea_np= np.loadtxt(xfeaturetxt_path,dtype= np.float32,delimiter=None,encoding='utf-8')
    fea_np_1=fea_np[:,FeaLst_item[0][0]:FeaLst_item[0][1]].tolist()
    fea_np_2=fea_np[:,FeaLst_item[1][0]:FeaLst_item[1][1]].tolist()
    fea_np_merge=[]
    for i in range(len(fea_np_1)):
        fea_np_merge.append(fea_np_1[i]+fea_np_2[i])
    fea_np=np.array(fea_np_merge,dtype='float32')#fea_np[:,FeaLst_item[0]:FeaLst_item[1]]
    in_feats=len(fea_np[0])
    # fea_= nn.Embedding(my_net.num_nodes, in_feats)
    fea_np=torch.tensor(fea_np).to('cuda:0')
    my_net.ndata['features']=fea_np
    # node label
    label= np.loadtxt(ylabeltxt_path,dtype= np.int32,delimiter=' ',encoding='utf-8')
    train_mask, val_mask,test_mask,train_maskSA1_2, val_maskSA1_2,test_maskSA1_2=None,None,None,None,None,None
    if os.path.isfile(train_masktxt_path) and os.path.isfile(val_masktxt_path) and os.path.isfile(test_masktxt_path):
        train_maskSA1_2=torch.tensor(np.load(train_masktxt_path))
        val_maskSA1_2=torch.tensor(np.load(val_masktxt_path))
        test_maskSA1_2=torch.tensor(np.load(test_masktxt_path))
        # As to many times experiments need to be conoduct,we split the training validatioin test data at one time and save the split result.
        # next time experiment will use the same training_mask validation_mask test_mask.
        train_TagIDSA1_2=oldTagID[train_maskSA1_2]
        val_TagIDSA1_2=oldTagID[val_maskSA1_2]
        SampleNum=len(label)
        train_mask = torch.zeros(SampleNum).to(torch.bool)
        val_mask = torch.zeros(SampleNum).to(torch.bool)
        test_mask = torch.zeros(SampleNum).to(torch.bool)
        assert SampleNum==len(newTagID)
        for tagID_i in range(len(newTagID)):
            tagID=newTagID[tagID_i]
            if tagID in train_TagIDSA1_2:
                train_mask[tagID_i]=True
            elif tagID in val_TagIDSA1_2:
                val_mask[tagID_i]=True
            else:
                test_mask[tagID_i]=True
    else:
        train_mask, val_mask,test_mask,random_index_arr=splitTrainValTestByCountRandom(label,lst_Num_singleCls_train,lst_Num_singleCls_val)#
    my_net.ndata['label']=torch.tensor(label).to('cuda:0')
    n_classes=len(np.unique(label))
    
    train_mask=train_mask.to('cuda:0')
    train_nid =train_mask.nonzero()
    train_nid=train_nid.squeeze()
    val_mask=val_mask.to('cuda:0')
    val_nid = val_mask.nonzero()
    val_nid = val_nid.squeeze()
    test_mask=test_mask.to('cuda:0')
    test_nid = test_mask.nonzero()
    test_nid = test_nid.squeeze()

    label_s=label.tolist()
    my_net.ndata['label']=torch.tensor(label_s).to('cuda:0')
    data = in_feats, n_classes, my_net, fea_np
    train_val_data=train_mask,  val_mask, test_mask,train_nid, val_nid, test_nid 
    
    return data, train_val_data,random_index_arr

#load data of NanShanGuangMing
def LoaddataBuilding0324NanShanGuangMing(Adjtxt_path,xfeaturetxt_path,ylabeltxt_path,FeaLst_item):
    random_index_arr=None
    #adj
    Adjtxt=np.loadtxt(Adjtxt_path,dtype= np.float32,delimiter=' ',encoding='utf-8')
    Adjtxt=Adjtxt.squeeze()
    src=Adjtxt[0:,0:1]
    dst=Adjtxt[0:,1:2]
    src=src.squeeze()
    dst=dst.squeeze()

    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    my_net=dgl.graph((u, v))#DGLGraph((u, v))
    my_net=my_net.to('cuda:0')

    # node feature
    fea_np= np.loadtxt(xfeaturetxt_path,dtype= np.float32,delimiter=None,encoding='utf-8')
    fea_np_1=fea_np[:,FeaLst_item[0][0]:FeaLst_item[0][1]].tolist()
    fea_np_2=fea_np[:,FeaLst_item[1][0]:FeaLst_item[1][1]].tolist()
    fea_np_merge=[]
    for i in range(len(fea_np_1)):
        fea_np_merge.append(fea_np_1[i]+fea_np_2[i])
    fea_np=np.array(fea_np_merge,dtype='float32')#fea_np[:,FeaLst_item[0]:FeaLst_item[1]]
    in_feats=len(fea_np[0])
    # fea_= nn.Embedding(my_net.num_nodes, in_feats)
    fea_np=torch.tensor(fea_np).to('cuda:0')
    my_net.ndata['features']=fea_np
    # node label
    label= np.loadtxt(ylabeltxt_path,dtype= np.int32,delimiter=' ',encoding='utf-8')
    my_net.ndata['label']=torch.tensor(label).to('cuda:0')
    n_classes=len(np.unique(label))
    label_s=label.tolist()
    my_net.ndata['label']=torch.tensor(label_s).to('cuda:0')
    data = in_feats, n_classes, my_net, fea_np
    return data

def get_mask(idx, length):
    mask = np.zeros(length)
    mask[idx] = 1
    return np.array(mask, dtype=np.float64)

def split_dataset(n_samples):
    val_indices = np.random.choice(list(range(n_samples)), size=int(n_samples * 0.2), replace=False)
    left = set(range(n_samples)) - set(val_indices)
    test_indices = np.random.choice(list(left), size=int(n_samples * 0.2), replace=False)
    train_indices = list(left - set(test_indices))

    train_mask = get_mask(train_indices, n_samples)
    eval_mask = get_mask(val_indices, n_samples)
    test_mask = get_mask(test_indices, n_samples)

    return train_mask, eval_mask, test_mask

#draw curve of acc
def exp_moving_avg(x, decay=0.9):
    shadow = x[0]
    a = [shadow]
    for v in x[1:]:
        shadow -= (1-decay) * (shadow-v)
        a.append(shadow)
    return a


# draw confusion matrix
def plot_confusion_matrix_percent(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues,confusionMatrixPngPath='./confusionMatrixPng.png'):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("混淆矩阵：")
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
        # print(cm)
    else:
        print()
        # print('混淆矩阵：')
        # print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    plt.savefig(confusionMatrixPngPath)
    plt.close()

# #load the data of NanShan after trained the graphsage using FuTian data
# Adjtxt_path=r'.\data&code\data\relGraphMatrixTransform_NanShan.txt'
# xfeaturetxt_path=r'.\data&code\data\featureMatrix_NanShan.txt'
# ylabeltxt_path=r'.\data&code\data\yClassMatrix_NanShan.txt'
# ylabel=np.loadtxt(ylabeltxt_path,dtype= np.int32,delimiter=' ',encoding='utf-8')
# TagID=np.loadtxt(r'.\data&code\data\TagIDTransform_NanShan.txt',dtype= np.str_,encoding='utf-8')

# #
# FeaLst=[
#         [[0,180],[0,0]],#      
# ]

# for FeaLst_i in range(len(FeaLst)):
#     data = LoaddataBuilding0324NanShanGuangMing(Adjtxt_path,xfeaturetxt_path,ylabeltxt_path,FeaLst[FeaLst_i])
#     model = torch.load(r'.\data&code\data\SaveModel.pth')
#     model.eval()  
#     with torch.no_grad():
#         my_net=data[2]
#         label=my_net.ndata['label']
#         all_mask=torch.tensor(np.full(label.shape,True)).to('cuda:0')
#         all_id=all_mask.nonzero().squeeze()                            
#         all_acc,all_label_pred_proba = evaluate(model, my_net, label, all_id, all_mask, 280, 0, 0)
#         all_label_pred=torch.argmax(all_label_pred_proba, dim=1)   
        
#         #输出文件的命名（预测所有数据）
#         outFileFolder=r'.\data&code\results'
#         outFileName_BasedParam=''#'Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctall"+str(round(all_acc.item(),4))
#         #输出txt（所有数据的预测结果 ）
#         all_label_pred=all_label_pred.cpu().tolist()
#         label_lst=label.cpu().tolist()
#         all_label_pred_proba=all_label_pred_proba.cpu().tolist()
#         result=[]#由TagID、y、pred、pred_probability组成
#         for prob_i in range(len(all_label_pred_proba)):
#             preResult_i=[TagID[prob_i],label_lst[prob_i],all_label_pred[prob_i]]
#             for prob_j in range(len(all_label_pred_proba[prob_i])):
#                 preResult_i.append(all_label_pred_proba[prob_i][prob_j])
#             result.append(preResult_i)
#         np.savetxt(outFileFolder+'/GraghSAGE predict__NanShan'+outFileName_BasedParam+'.txt',result,fmt='%s')
#         #混淆矩阵（所有数据的预测结果 ）
#         cm= confusion_matrix(label_lst,all_label_pred)
#         attack_types = ['Industrial','Commercial', 'Resident', 'Public', 'Education', 'Mixed', 'UrbanVillage']
#         plot_confusion_matrix_percent(cm, classes=attack_types, normalize=False, title='Normalized confusion matrix  all_acc '+str(round(all_acc.item(),4)),confusionMatrixPngPath=outFileFolder+"/CM_NanShan"+outFileName_BasedParam+".png")
    
        
# # load the data of GuangMing after trained the graphsage using FuTian data
# Adjtxt_path=r'.\data&code\data\relGraphMatrixTransform_GuangMing.txt'
# xfeaturetxt_path=r'.\data&code\data\featureMatrix_GuangMing.txt'
# ylabeltxt_path=r'.\data&code\data\yClassMatrix_GuangMing.txt'
# ylabel=np.loadtxt(ylabeltxt_path,dtype= np.int32,delimiter=' ',encoding='utf-8')
# TagID=np.loadtxt(r'.\data&code\data\TagIDTransform_GuangMing.txt',dtype= np.str_,encoding='utf-8')

# #
# FeaLst=[
#         [[0,180],[0,0]],#
# ]
# for FeaLst_i in range(len(FeaLst)):
#     data = LoaddataBuilding0324NanShanGuangMing(Adjtxt_path,xfeaturetxt_path,ylabeltxt_path,FeaLst[FeaLst_i])
#     model = torch.load(r'.\data&code\data\SaveModel.pth')
#     model.eval() 
#     with torch.no_grad():
#         my_net=data[2]
#         label=my_net.ndata['label']
#         all_mask=torch.tensor(np.full(label.shape,True)).to('cuda:0')
#         all_id=all_mask.nonzero().squeeze()                            
#         all_acc,all_label_pred_proba = evaluate(model, my_net, label, all_id, all_mask, 280, 0, 0)
#         all_label_pred=torch.argmax(all_label_pred_proba, dim=1)   
        
#         #output path name
#         outFileFolder=r'.\data&code\results'
#         outFileName_BasedParam=''#'Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctall"+str(round(all_acc.item(),4))
#         #
#         all_label_pred=all_label_pred.cpu().tolist()
#         label_lst=label.cpu().tolist()
#         all_label_pred_proba=all_label_pred_proba.cpu().tolist()
#         result=[]#
#         for prob_i in range(len(all_label_pred_proba)):
#             preResult_i=[TagID[prob_i],label_lst[prob_i],all_label_pred[prob_i]]
#             for prob_j in range(len(all_label_pred_proba[prob_i])):
#                 preResult_i.append(all_label_pred_proba[prob_i][prob_j])
#             result.append(preResult_i)
#         np.savetxt(outFileFolder+'/GraghSAGE predict__NanShan'+outFileName_BasedParam+'.txt',result,fmt='%s')
#         cm= confusion_matrix(label_lst,all_label_pred)
#         attack_types = ['Industrial','Commercial', 'Resident', 'Public', 'Education', 'Mixed', 'UrbanVillage']
#         plot_confusion_matrix_percent(cm, classes=attack_types, normalize=False, title='Normalized confusion matrix  all_acc '+str(round(all_acc.item(),4)),confusionMatrixPngPath=outFileFolder+"/CM_NanShan"+outFileName_BasedParam+".png")
    
# Program entry:load FuTian data
Adjtxt_path=r'.\data&code\data\relGraphMatrixTransform.txt'
xfeaturetxt_path=r'.\data&code\data\featureMatrix.txt'
ylabeltxt_path=r'.\data&code\data\yClassMatrix.txt'
ylabel=np.loadtxt(ylabeltxt_path,dtype= np.int32,delimiter=' ',encoding='utf-8')
oldTagID=np.loadtxt(r'.\data&code\data\oldTagID.txt',dtype= np.str_,encoding='utf-8')
TagID=np.loadtxt(r'.\data&code\data\TagID.txt',dtype= np.str_,encoding='utf-8')
outFileFolder2=r'.\data&code\data'
train_masktxt_path=outFileFolder2+'/'+'train mask_.npy'
val_masktxt_path=outFileFolder2+'/'+'val mask_.npy'
test_masktxt_path=outFileFolder2+'/'+'test mask_.npy'
#set paragram
FeaLst=[
        [[0,180],[0,0]],#
]
learning_rate_Lst=[0.002]
batch_s_Lst=[280]#[64,96]
hidden_size_lst=[256]
weight_decay_lst=[1e-3]#[1e-3,5e-3]
sample_size_lst =[[3,3]]#,[4,4,4]
boxplt_acc_val_LR,boxplt_acc_val_batch,boxplt_acc_val_hidden,boxplt_acc_val_sample_size={},{},{},{}
boxplt_acc_test_LR,boxplt_acc_test_batch,boxplt_acc_test_hidden,boxplt_acc_test_sample_size={},{},{},{}
sample_size_keys=[]
for i_samplesize in range(len(sample_size_lst)):
    item_s=''
    for j_samplesize in range(len(sample_size_lst[i_samplesize])):
        item_s=item_s+str(sample_size_lst[i_samplesize][j_samplesize])+'_'
    sample_size_keys.append(item_s)
    
#spli training validation test data
lst_Num_singleCls_train,lst_Num_singleCls_val=[100,300,300,300,300,300,300],[50,100,100,100,100,100,100]
for FeaLst_i in range(len(FeaLst)):
    data, train_val_data,random_index_arr = LoaddataBuilding0324(Adjtxt_path,xfeaturetxt_path,ylabeltxt_path,FeaLst[FeaLst_i],lst_Num_singleCls_train,lst_Num_singleCls_val,
                                                                 train_masktxt_path,val_masktxt_path,test_masktxt_path,oldTagID,TagID)
    for lr_i in range(len(learning_rate_Lst)):
        for batch_s_i in range(len(batch_s_Lst)):
            for hidden_size_i in range(len(hidden_size_lst)):
                for weight_decay_i in range(len(weight_decay_lst)):
                    for sample_size_i in range(len(sample_size_lst)):
                        epochs=40
                        hidden_size = hidden_size_lst[hidden_size_i]#16 #[16,32]  
                        sample_size =sample_size_lst[sample_size_i]#[4,4] 
                        n_layers = 2
                        if len(sample_size)==3:
                            n_layers = 3
                        activation = F.relu
                        dropout = 0.5
                        aggregator = 'mean'
                        batch_s =batch_s_Lst[batch_s_i]# 128  
                        num_worker = 0
                        learning_rate = learning_rate_Lst[lr_i]#0.01
                        device = 0
                        weight_decay=weight_decay_lst[weight_decay_i]
                        args =  epochs,hidden_size, n_layers, activation, dropout, aggregator, batch_s, num_worker  ,weight_decay

                        #traing model
                        trained_model,curves_metrics,acc_test,test_label_pred_proba,acc_train,train_label_pred_proba,val_label_pred_proba = Run(data, train_val_data, args, sample_size, learning_rate, device) #train_val_data的验证、测试集赋值不对1016 train_val_data=train_mask,  test_mask,val_mask, train_nid,  test_nid, val_nid
                        
                        #kb output result as follows
                        my_net=data[2]
                        label=my_net.ndata['label']
                        all_mask=torch.tensor(np.full(label.shape,True)).to('cuda:0')
                        all_id=all_mask.nonzero().squeeze()
                        all_acc,all_label_pred_proba = evaluate(trained_model, my_net, label, all_id, all_mask, batch_s, num_worker, device)
                        all_label_pred=torch.argmax(all_label_pred_proba, dim=1)   
                        outFileFolder=r'.\data&code\results'
                        outFileName_BasedParam='Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctall"+str(round(all_acc.item(),4))
                        acc_loss_pngTitle='GraghSAGE acc and loss lr_'+str(learning_rate)+" dropout_"+str(dropout)+" batch_s_"+str(batch_s)+"\n hidden_"+str(hidden_size)+" gcnnum_"+str(n_layers)+" sample_size_"+str(sample_size)+" aggregator_"+str(aggregator)      +str(aggregator) +"\n weight_decay"+str(weight_decay) +" accall"+str(round(all_acc.item(),4))             

                        all_label_pred=all_label_pred.cpu().tolist()
                        label_lst=label.cpu().tolist()
                        all_label_pred_proba=all_label_pred_proba.cpu().tolist()
                        result=[]#
                        for prob_i in range(len(all_label_pred_proba)):
                            preResult_i=[TagID[prob_i],label_lst[prob_i],all_label_pred[prob_i]]
                            for prob_j in range(len(all_label_pred_proba[prob_i])):
                                preResult_i.append(all_label_pred_proba[prob_i][prob_j])
                            result.append(preResult_i)
                        np.savetxt(outFileFolder+'/GraghSAGE predict_'+outFileName_BasedParam+'.txt',result,fmt='%s')
                        #confusion matrix
                        cm= confusion_matrix(label_lst,all_label_pred)
                        attack_types = ['Industrial','Commercial', 'Resident', 'Public', 'Education', 'Mixed', 'UrbanVillage']
                        plot_confusion_matrix_percent(cm, classes=attack_types, normalize=False, title='Normalized confusion matrix  all_acc '+str(round(all_acc.item(),4)),confusionMatrixPngPath=outFileFolder+"/CM"+outFileName_BasedParam+".png")

                        #kb output training data results
                        train_mask,  val_mask, test_mask,train_nid, val_nid, test_nid= train_val_data
                        # train_acc,train_label_pred_proba = evaluate(trained_model, my_net, label_lst, train_nid , train_mask, batch_s, num_worker, device)
                        train_label_pred_proba=train_label_pred_proba[train_nid]
                        train_label_pred=torch.argmax(train_label_pred_proba, dim=1)   
                        #
                        outFileName_BasedParam='Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctrain"+acc_train
                        acc_loss_pngTitle='GraghSAGE acc and loss lr_'+str(learning_rate)+" dropout_"+str(dropout)+" batch_s_"+str(batch_s)+"\n hidden_"+str(hidden_size)+" gcnnum_"+str(n_layers)+" sample_size_"+str(sample_size)+" aggregator_"+str(aggregator)      +str(aggregator) +"\n weight_decay"+str(weight_decay) +" acctrain"+acc_train           
                        #
                        train_label_pred=train_label_pred.cpu().tolist()
                        train_label_pred_proba=train_label_pred_proba.cpu().tolist()
                        result=[]#
                        TagID_train=TagID[train_nid.cpu().tolist()]
                        train_label=label[train_mask].cpu().tolist()
                        for prob_i in range(len(train_label_pred_proba)):
                            preResult_i=[TagID_train[prob_i],train_label[prob_i],train_label_pred[prob_i]]
                            for prob_j in range(len(train_label_pred_proba[prob_i])):
                                preResult_i.append(train_label_pred_proba[prob_i][prob_j])
                            result.append(preResult_i)
                        np.savetxt(outFileFolder+'/GraghSAGE predict_'+outFileName_BasedParam+'.txt',result,fmt='%s')
                        #confusion matrix
                        cm_train= confusion_matrix(train_label,train_label_pred)
                        plot_confusion_matrix_percent(cm_train, classes=attack_types, normalize=False, title='Normalized confusion matrix  train_acc '+acc_train,confusionMatrixPngPath=outFileFolder+"/CM"+outFileName_BasedParam+".png")

                        #kb 
                        train_mask,  val_mask, test_mask,train_nid, val_nid, test_nid= train_val_data
                        # test_acc,test_label_pred_proba = evaluate(trained_model, my_net, label_lst, test_nid , test_mask, batch_s, num_worker, device)
                        test_label_pred_proba=test_label_pred_proba[test_nid]
                        test_label_pred=torch.argmax(test_label_pred_proba, dim=1)   
                        #
                        outFileName_BasedParam='Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctest"+acc_test
                        acc_loss_pngTitle='GraghSAGE acc and loss lr_'+str(learning_rate)+" dropout_"+str(dropout)+" batch_s_"+str(batch_s)+"\n hidden_"+str(hidden_size)+" gcnnum_"+str(n_layers)+" sample_size_"+str(sample_size)+" aggregator_"+str(aggregator)      +str(aggregator) +"\n weight_decay"+str(weight_decay) +" acctest"+acc_test           
                        #
                        test_label_pred=test_label_pred.cpu().tolist()
                        test_label_pred_proba=test_label_pred_proba.cpu().tolist()
                        result=[]
                        TagID_test=TagID[test_nid.cpu().tolist()]
                        test_label=label[test_mask].cpu().tolist()
                        for prob_i in range(len(test_label_pred_proba)):
                            preResult_i=[TagID_test[prob_i],test_label[prob_i],test_label_pred[prob_i]]
                            for prob_j in range(len(test_label_pred_proba[prob_i])):
                                preResult_i.append(test_label_pred_proba[prob_i][prob_j])
                            result.append(preResult_i)
                        np.savetxt(outFileFolder+'/GraghSAGE predict_'+outFileName_BasedParam+'.txt',result,fmt='%s')
                        #confusion matrix
                        cm_test= confusion_matrix(test_label,test_label_pred)
                        np.save(outFileFolder+'/CMnpy_'+outFileName_BasedParam+'.npy',cm_test)
                        plot_confusion_matrix_percent(cm_test, classes=attack_types, normalize=False, title='Normalized confusion matrix  test_acc '+acc_test,confusionMatrixPngPath=outFileFolder+"/CM"+outFileName_BasedParam+".png")
                        # calculate and output precision、f1 score、ROC-AUC
                        val_label=label[val_mask].cpu().tolist()
                        val_label_pred_proba=val_label_pred_proba[val_nid]
                        val_label_pred=torch.argmax(val_label_pred_proba, dim=1) 
                        val_label_pred=val_label_pred.cpu().tolist()
                        val_label_pred_proba=val_label_pred_proba.cpu().tolist()
                        recall_score_train=  recall_score(train_label,train_label_pred,average='weighted')
                        recall_score_val=recall_score(val_label,val_label_pred,average='weighted')
                        recall_score_test=recall_score(test_label,test_label_pred,average='weighted')
                        precision_train=precision_score(train_label,train_label_pred,average='weighted')#macro
                        precision_val=precision_score(val_label,val_label_pred,average='weighted')
                        precision_test=precision_score(test_label,test_label_pred,average='weighted')
                        f1_train=f1_score(train_label,train_label_pred,average='weighted')
                        f1_val=f1_score(val_label,val_label_pred,average='weighted')
                        f1_test=f1_score(test_label,test_label_pred,average='weighted')
                        #
                        x_epoch,accs_train,accs_val,losses_train,losses_val=curves_metrics
                        out_acc_lst=['accuracy',accs_train[-1] ,accs_val[-1],float(acc_test),'precision',precision_train,precision_val,precision_test,'recall',recall_score_train,recall_score_val,recall_score_test,'f1score',f1_train,f1_val,f1_test]#,'rocauc',roc_auc_train,roc_auc_val,roc_auc_test]
                        outFileName_BasedParam='Fea'+str(FeaLst[FeaLst_i])+str(learning_rate)+" drop_"+str(dropout)+" batch_"+str(batch_s)+" hid_"+str(hidden_size)+" gcn_"+str(n_layers)+" sam_size_"+str(sample_size)+" w_d"+str(weight_decay)+" acctrain"+acc_train
                        np.savetxt(outFileFolder+'\\acc_'+outFileName_BasedParam+'.txt',out_acc_lst,fmt="%s",delimiter=',')#
                        out_predict_lst=[train_label,train_label_pred,train_label_pred_proba,val_label,val_label_pred,val_label_pred_proba,test_label,test_label_pred,test_label_pred_proba]
                        np.save(outFileFolder+'\\out_predict'+outFileName_BasedParam+'.npy',out_predict_lst)
                        out_predict_=np.load(outFileFolder+'\\out_predict'+outFileName_BasedParam+'.npy',allow_pickle=True)
                        # 
                        if learning_rate_Lst[lr_i] in boxplt_acc_val_LR.keys():
                            boxplt_acc_val_LR[learning_rate_Lst[lr_i]].append(accs_val[-1])
                        else:
                            boxplt_acc_val_LR[learning_rate_Lst[lr_i]]=[accs_val[-1]]
                        if batch_s_Lst[batch_s_i] in boxplt_acc_val_batch.keys():
                            boxplt_acc_val_batch[batch_s_Lst[batch_s_i]].append(accs_val[-1])
                        else:
                            boxplt_acc_val_batch[batch_s_Lst[batch_s_i]]=[accs_val[-1]]
                        if hidden_size_lst[hidden_size_i] in boxplt_acc_val_hidden.keys():
                            boxplt_acc_val_hidden[hidden_size_lst[hidden_size_i]].append(accs_val[-1])
                        else:
                            boxplt_acc_val_hidden[hidden_size_lst[hidden_size_i]]=[accs_val[-1]]  
                        if sample_size_keys[sample_size_i] in boxplt_acc_val_sample_size.keys():
                            boxplt_acc_val_sample_size[sample_size_keys[sample_size_i]].append(accs_val[-1])
                        else:
                            boxplt_acc_val_sample_size[sample_size_keys[sample_size_i]]=[accs_val[-1]]     
                        # 
                        if learning_rate_Lst[lr_i] in boxplt_acc_test_LR.keys():
                            boxplt_acc_test_LR[learning_rate_Lst[lr_i]].append(float(acc_test))
                        else:
                            boxplt_acc_test_LR[learning_rate_Lst[lr_i]]=[float(acc_test)]
                        if batch_s_Lst[batch_s_i] in boxplt_acc_test_batch.keys():
                            boxplt_acc_test_batch[batch_s_Lst[batch_s_i]].append(float(acc_test))
                        else:
                            boxplt_acc_test_batch[batch_s_Lst[batch_s_i]]=[float(acc_test)]
                        if hidden_size_lst[hidden_size_i] in boxplt_acc_test_hidden.keys():
                            boxplt_acc_test_hidden[hidden_size_lst[hidden_size_i]].append(float(acc_test))
                        else:
                            boxplt_acc_test_hidden[hidden_size_lst[hidden_size_i]]=[float(acc_test)]  
                        if sample_size_keys[sample_size_i] in boxplt_acc_test_sample_size.keys():
                            boxplt_acc_test_sample_size[sample_size_keys[sample_size_i]].append(float(acc_test))
                        else:
                            boxplt_acc_test_sample_size[sample_size_keys[sample_size_i]]=[float(acc_test)]                       
                        #
                        # TagID_train=TagID[train_nid.cpu().tolist()]
                        # TagID_val=TagID[val_nid.cpu().tolist()]
                        # TagID_filename='TagID_'+str(learning_rate)+" dropout_"+str(dropout)+" batch_s_"+str(batch_s)+" hidden_"+str(hidden_size)+" gcnnum_"+str(n_layers)+" sample_size_"+str(sample_size)+" aggregator_"+str(aggregator) +" weight_decay"+str(weight_decay)+" testall"+acc_test
                        # np.savetxt(outFileFolder+'/'+'train '+TagID_filename+'.txt',TagID_train,fmt='%s')
                        # np.savetxt(outFileFolder+'/'+'val '+TagID_filename+'.txt',TagID_val,fmt='%s')
                        # np.savetxt(outFileFolder+'/'+'test '+TagID_filename+'.txt',TagID_test,fmt='%s')
                        if random_index_arr is not  None:
                            np.save(outFileFolder+'/'+'random_index_arr',random_index_arr)
                            mask_filename='mask_'#+str(learning_rate)+" dropout_"+str(dropout)+" batch_s_"+str(batch_s)+" hidden_"+str(hidden_size)+" gcnnum_"+str(n_layers)+" sample_size_"+str(sample_size)+" aggregator_"+str(aggregator) +" weight_decay"+str(weight_decay)+" testall"+acc_test
                            train_mask=train_mask.cpu().numpy()
                            val_mask=val_mask.cpu().numpy()
                            test_mask=test_mask.cpu().numpy()
                            np.save(outFileFolder+'/'+'train '+mask_filename,train_mask)
                            np.save(outFileFolder+'/'+'val '+mask_filename,val_mask)
                            np.save(outFileFolder+'/'+'test '+mask_filename,test_mask)
                        # np.savetxt(outFileFolder+'/'+'train '+mask_filename+'.txt',train_mask,fmt='%s')
                        # np.savetxt(outFileFolder+'/'+'val '+mask_filename+'.txt',val_mask,fmt='%s')
                        # np.savetxt(outFileFolder+'/'+'test '+mask_filename+'.txt',test_mask,fmt='%s')

                        #
                        fig, ax1 = plt.subplots()
                        ax1.set_xlabel('Epoch')
                        ax1.set_ylabel('Accuracy')#, color=color)
                        ax1.plot(x_epoch, accs_train,color='red')#exp_moving_avg(accs_train),color='red')
                        ax1.plot(x_epoch, accs_val, color='blue')#exp_moving_avg(accs_val), color='blue')
                        ax1.tick_params(axis='y')#, labelcolor=color)
                        ax1.set_title(acc_loss_pngTitle)
                        plt.legend(['Training accuracy ', 'Validation accuracy'])
                        fig.tight_layout()  # otherwise the right y-label is slightly clipped
                        plt.savefig(outFileFolder+"/GraghSAGE acc and loss lr_"+outFileName_BasedParam+".png")
                        # plt.show()
                        plt.close()
                        #
                        import xlwt
                        workbook=xlwt.Workbook(encoding='utf-8')  
                        booksheet1=workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
                        booksheet1.write(0,0,'accs_train')  
                        booksheet1.write(0,1,'accs_val')  
                        booksheet1.write(0,2,'losses_train')  
                        booksheet1.write(0,3,'losses_val')  
                        for res_i in range(len(x_epoch)):
                            booksheet1.write(res_i+1,0,accs_train[res_i])  
                            booksheet1.write(res_i+1,1,accs_val[res_i])  
                            booksheet1.write(res_i+1,2,losses_train[res_i])  
                            booksheet1.write(res_i+1,3,losses_val[res_i])  
                        workbook.save(outFileFolder+'\\_losses_accs.xls')
                        

# # output boxplot result to excel
import xlwt
workbook=xlwt.Workbook(encoding='utf-8')  
booksheet1=workbook.add_sheet('Sheet LR', cell_overwrite_ok=True)
column=0
for key_LR in boxplt_acc_val_LR.keys():
    booksheet1.write(0,column,key_LR)  
    for value_AccLR in range(len(boxplt_acc_val_LR[key_LR])):
        booksheet1.write(value_AccLR+1,column,boxplt_acc_val_LR[key_LR][value_AccLR]) 
    column+=1
booksheet_batch=workbook.add_sheet('Sheet batch', cell_overwrite_ok=True)
column=0
for key_batch in boxplt_acc_val_batch.keys():
    booksheet_batch.write(0,column,key_batch)  
    for value_Acc_batch in range(len(boxplt_acc_val_batch[key_batch])):
        booksheet_batch.write(value_Acc_batch+1,column,boxplt_acc_val_batch[key_batch][value_Acc_batch]) 
    column+=1  
booksheet_hidden=workbook.add_sheet('Sheet hidden', cell_overwrite_ok=True)
column=0
for key_hidden in boxplt_acc_val_hidden.keys():
    booksheet_hidden.write(0,column,key_hidden)  
    for value_Acc_hidden in range(len(boxplt_acc_val_hidden[key_hidden])):
        booksheet_hidden.write(value_Acc_hidden+1,column,boxplt_acc_val_hidden[key_hidden][value_Acc_hidden]) 
    column+=1  
booksheet_sample_size=workbook.add_sheet('Sheet sample_size', cell_overwrite_ok=True)
column=0
for key_sample_size in boxplt_acc_val_sample_size.keys():
    booksheet_sample_size.write(0,column,key_sample_size)  
    for value_Acc_sample_size in range(len(boxplt_acc_val_sample_size[key_sample_size])):
        booksheet_sample_size.write(value_Acc_sample_size+1,column,boxplt_acc_val_sample_size[key_sample_size][value_Acc_sample_size]) 
    column+=1      
workbook.save(outFileFolder+'\\val_Boxplot_sample_size.xls')

workbook=xlwt.Workbook(encoding='utf-8')  
booksheet1=workbook.add_sheet('Sheet LR', cell_overwrite_ok=True)
column=0
for key_LR in boxplt_acc_test_LR.keys():
    booksheet1.write(0,column,key_LR)  
    for value_AccLR in range(len(boxplt_acc_test_LR[key_LR])):
        booksheet1.write(value_AccLR+1,column,boxplt_acc_test_LR[key_LR][value_AccLR]) 
    column+=1
booksheet_batch=workbook.add_sheet('Sheet batch', cell_overwrite_ok=True)
column=0
for key_batch in boxplt_acc_test_batch.keys():
    booksheet_batch.write(0,column,key_batch)  
    for value_Acc_batch in range(len(boxplt_acc_test_batch[key_batch])):
        booksheet_batch.write(value_Acc_batch+1,column,boxplt_acc_test_batch[key_batch][value_Acc_batch]) 
    column+=1  
booksheet_hidden=workbook.add_sheet('Sheet hidden', cell_overwrite_ok=True)
column=0
for key_hidden in boxplt_acc_test_hidden.keys():
    booksheet_hidden.write(0,column,key_hidden)  
    for value_Acc_hidden in range(len(boxplt_acc_test_hidden[key_hidden])):
        booksheet_hidden.write(value_Acc_hidden+1,column,boxplt_acc_test_hidden[key_hidden][value_Acc_hidden]) 
    column+=1  
booksheet_sample_size=workbook.add_sheet('Sheet sample_size', cell_overwrite_ok=True)
column=0
for key_sample_size in boxplt_acc_test_sample_size.keys():
    booksheet_sample_size.write(0,column,key_sample_size)  
    for value_Acc_sample_size in range(len(boxplt_acc_test_sample_size[key_sample_size])):
        booksheet_sample_size.write(value_Acc_sample_size+1,column,boxplt_acc_test_sample_size[key_sample_size][value_Acc_sample_size]) 
    column+=1      
workbook.save(outFileFolder+'\\_test_Boxplot_sample_size.xls')
