import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim as optim
from numpy import load
import torch.nn.functional as F
from torch.autograd import Function
from model import type_ae
from utils import data_loading, masking, normalize_output
from sklearn.metrics import f1_score
import time
from collections import Counter
import pickle
L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)

#L2018=np.load('/home/malo/Stage/Data/données 16 classes/l2018_16.npz',allow_pickle=True)
#L2019=np.load('/home/malo/Stage/Data/données 16 classes/l2019_16.npz',allow_pickle=True)
#L2020=np.load('/home/malo/Stage/Data/données 16 classes/l2020_16.npz',allow_pickle=True)
#R2018=np.load('/home/malo/Stage/Data/données 16 classes/r2018_16.npz',allow_pickle=True)
#R2019=np.load('/home/malo/Stage/Data/données 16 classes/r2019_16.npz',allow_pickle=True)
#R2020=np.load('/home/malo/Stage/Data/données 16 classes/r2020_16.npz',allow_pickle=True)
#T2018=np.load('/home/malo/Stage/Data/données 16 classes/t2018_16.npz',allow_pickle=True)
#T2019=np.load('/home/malo/Stage/Data/données 16 classes/t2019_16.npz',allow_pickle=True)
#T2020=np.load('/home/malo/Stage/Data/données 16 classes/t2020_16.npz',allow_pickle=True)
datas = [L2018,L2019,L2020,T2018,T2019,T2020,R2018,R2019,R2020]
l_f1 = []
for d in datas :
    n_epochs= 100
    data = [[R2018],[d]]
    train_dataloader, test_dataloader, dates, data_shape = data_loading(data,fraction = 0.2)
                                                                        
    config1 = {'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':64}
    config2 = {'emb_size':64,'num_heads':8,'Data_shape':data_shape,'Fix_pos_encode':'tAPE','Rel_pos_encode':'eRPE','dropout':0.2,'dim_ff':64}
    loss_CE = nn.CrossEntropyLoss()
    loss_MSE = nn.MSELoss()
    num_classes = 11
    num_dom = len(data[0]) + 1
    model = type_ae(config1 = config1, config2 = config2, num_classes = num_classes, num_dom = num_dom, dates = dates)

    learning_rate = 0.0001

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    optimizer2 = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for epoch in range(n_epochs):
        model.train()
        print (f"---epoque: {epoch+1}---")
        start = time.time()
        tot_pred = []
        tot_labels = []
        for xm_batch, y_batch, dom_batch in train_dataloader :
            x_batch, m_batch = xm_batch[:,:,:2], xm_batch[:,:,2]
            x_batch.to(device)
            m_batch.to(device)
            y_batch.to(device)
            
            inte = torch.where(m_batch == 1)
        
            x_batch_b, m_batch_b, indices_b = masking(x_batch,m_batch,p=0.2)
            
            optimizer.zero_grad()
            optimizer2.zero_grad()
            cls_dom_adv, cls_class, out = model(x_batch_b, m_batch_b, phase = 'training')
            
            pred_npy = np.argmax(cls_class.cpu().detach().numpy(), axis=1)
        
            i_lab = torch.where(y_batch != -1)
            tot_pred.append( pred_npy[i_lab] )
            tot_labels.append( y_batch[i_lab].cpu().detach().numpy())
            
            
            loss_rec = loss_MSE(out[inte], x_batch[inte])
            
            loss_lab = loss_CE(cls_class[i_lab], y_batch[i_lab]) 
            
            # ajout de losses contratsives supplémentaires. lesquels et comment ? 
            loss_dom_adv = loss_CE(cls_dom_adv, dom_batch)
            loss = loss_lab + loss_dom_adv
            
            loss_rec.backward(retain_graph=True)
            
            loss.backward()
            optimizer.step() 
            optimizer2.step()
        tot_pred = np.concatenate(tot_pred)
        tot_labels = np.concatenate(tot_labels)
        fscore= f1_score(tot_pred, tot_labels, average="weighted")
        print(loss.item(),loss_rec.item(),loss_lab.item())
        print(fscore)
        print(time.time()-start)  
        
    tot_pred = []
    tot_labels = []
    k=0
    s=0
    verif = {}
    model.eval()
    for xm_batch, y_batch in test_dataloader:
        x_batch, m_batch = xm_batch[:,:,:2], xm_batch[:,:,2]
        x_batch.to(device)
        m_batch.to(device)
        y_batch.to(device)
        inte = torch.where(m_batch == 1)
        x_batch_b, m_batch_b, indices_b = masking(x_batch,m_batch,p=0.2)
        cls_dom, cls_class, out = model(x_batch_b, m_batch_b, phase = 'training')
        
        
        
        pred_npy = np.argmax(cls_class.cpu().detach().numpy(), axis=1)
        
        
        tot_pred.append( pred_npy )
        tot_labels.append( y_batch.cpu().detach().numpy())
    verif['original'] = x_batch[1]
    verif['recon'] = out.detach()[1]
    i = torch.where(inte[0]==1)
    verif['indices'] = inte[1][i]
    with open('verification_reconstruction_c','wb') as f :
        pickle.dump(verif,f)
    tot_pred = np.concatenate(tot_pred)
    tot_labels = np.concatenate(tot_labels)



    fscore= f1_score(tot_pred, tot_labels, average="weighted")
    fscore_pc= f1_score(tot_pred, tot_labels, average=None)
    l_f1.append(fscore)
    print(fscore)
    print(fscore_pc)
    print(np.unique(tot_pred))
    print(Counter(tot_pred))
    print(Counter(tot_labels))
with open('ae R2018','wb') as f :
        pickle.dump(l_f1,f)
