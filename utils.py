import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import math
L2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2018_modif.npz',allow_pickle=True)
L2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2019_modif.npz',allow_pickle=True)
L2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/l2020_modif.npz',allow_pickle=True)
R2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2018_modif.npz',allow_pickle=True)
R2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2019_modif.npz',allow_pickle=True)
R2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/r2020_modif.npz',allow_pickle=True)
T2018=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2018_modif.npz',allow_pickle=True)
T2019=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2019_modif.npz',allow_pickle=True)
T2020=np.load('/home/malo/Stage/Data/data modifiées 11 classes/t2020_modif.npz',allow_pickle=True)


def getData(fileNames, max_days):
    data = []
    y = []
    mask = []
    dom = []
    for i, fName in enumerate(fileNames):
        print("LOAD DATA %s "%fName)
        npz = np.load(fName)
        temp_data, temp_y, _, temp_mask = getInfos(npz, max_days)
        data.append(temp_data)
        y.append(temp_y)
        mask.append(temp_mask)
        dom.append(i*np.ones((temp_mask.shape[0])))
    data = np.concatenate(data,axis=0)
    y = np.concatenate(y,axis=0)
    mask = np.concatenate(mask,axis=0)
    dom = np.concatenate(dom,axis=0)
    data, y, mask, dom = shuffle(data, y, mask, dom)
    return data, y, mask, dom


def getInfos(npz, max_days):
    data = npz['X_SAR']
    labels = npz['y']-1
    dates = npz['dates_SAR']
    dates_pos = get_day_count(dates)
    mask = np.zeros((data.shape[0],max_days))
    mask[:,dates_pos]=1.0
    new_data=np.zeros((data.shape[0],max_days,data.shape[2]))
    new_data[:,dates_pos,:] = data
    return new_data, labels, dates, mask

def masking(datas, mask,p) :
  s=mask.size()
  pr=torch.ones(s)*(1-p)
  mask_b = torch.where(mask==1, torch.bernoulli(pr),mask)
  indices = torch.where(mask!=mask_b)
  
  
  datas_b = torch.clone(datas)
  datas_b[indices] = 0
  return datas_b, mask_b, indices

class dropout:
    def __init__(self, p): # p est la probabilité de conservation des donées
        self.p = p
    def augment(self,x,mask):
        
        
        size = [x.shape[0],x.shape[1]]
        
        
       
        suppr = torch.bernoulli(self.p * torch.ones(size)).cuda() # on va conserver les données là où il y a un 1 et supprimer celles où où il y a un 0
        
        
        
        mask = mask.masked_fill(suppr==0,0)
        suppr = suppr.unsqueeze(2)
        suppr = suppr.repeat(1,1,2)
        
        x = x.masked_fill(suppr==0,0)
        
        return x,mask
class identité:
        
        def augment(x,mask):
          return(x,mask)
def get_day_count(dates,ref_day='09-01'):
    # Days elapsed from 'ref_day' of the year in dates[0]
    ref = np.datetime64(f'{dates.astype("datetime64[Y]")[0]}-'+ref_day)
    days_elapsed = (dates - ref).astype('timedelta64[D]').astype(int) #(dates - ref_day).astype('timedelta64[D]').astype(int)#
    return torch.tensor(days_elapsed,dtype=torch.long)

def add_mask(values,mask): # permet d'attacher les mask aux données pour pouvoir faire les batchs sans perdre le mask
    mask=mask.unsqueeze(0).unsqueeze(-1)
    shape=values.shape
    mask=mask.expand(shape[0],-1,-1)
    values=torch.tensor(values,dtype=torch.float32)

    valuesWmask=torch.cat((values,mask),dim=-1)
    return valuesWmask

def comp (data,msk) : #permet de formater les données avec 365 points d'acquisitions
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  j_p=(data['dates_SAR']-ref).astype('timedelta64[D]').astype(int)
  année=list(range(365))

  année = [ref + np.timedelta64(j, 'D') for j in année ]
  mask = []

  for i,jour in enumerate(année):
    if jour not in data['dates_SAR']:

      mask+=[0]
      msk=np.insert(msk,i,0)
      data_r['dates_SAR']=np.insert(data_r['dates_SAR'],i,jour)
      data_r['X_SAR']=np.insert(data_r['X_SAR'],i,[0,0],axis=1)
    else:
      mask+=[1]


  mask=torch.tensor(mask,dtype=torch.float32)
  msk=torch.tensor(msk,dtype=torch.float32)
  return data_r,mask,msk
def suppr (data,ratio):
  data_r={'X_SAR':data['X_SAR'],'y':data['y'],'dates_SAR':data['dates_SAR']}
  ref=data['dates_SAR'][0]
  nbr,seq_len,channels=data['X_SAR'].shape #(nbr,seq_len,channels)
  
  nbr_indice=int(seq_len*ratio)
  indice=list(range(seq_len))
  indice=random.sample(indice,nbr_indice)
  mask=[0 if i in indice else 1 for i in range(seq_len)]
  mask=torch.tensor(mask)

  data_r['X_SAR']=torch.tensor(data_r['X_SAR'])
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].masked_fill(mask==0,0)
  data_r['X_SAR']=data_r['X_SAR'].permute(0,2,1)
  data_r['X_SAR']=data_r['X_SAR'].numpy()
  mask=mask.numpy()
  return data_r,mask

import torch

def normalize_output(output, percentile_low=1, percentile_high=99):
    """
    Normalize the output tensor between 0 and 1 using percentiles.
    
    Args:
    - output (torch.Tensor): The tensor output of the network with shape (batch_size, height, width, channels).
    - percentile_low (float): The lower percentile for normalization (default is 1).
    - percentile_high (float): The upper percentile for normalization (default is 99).
    
    Returns:
    - torch.Tensor: The normalized tensor.
    """
    # Convert output tensor to numpy array
    output_np = output.cpu().detach().numpy()
    
    # Compute percentiles
    p_min = np.percentile(output_np, percentile_low)
    p_max = np.percentile(output_np, percentile_high)
    
    # Normalize
    output_norm = (output_np - p_min) / (p_max - p_min)
    
    # Clip values to be between 0 and 1
    output_norm = np.clip(output_norm, 0, 1)
    
    # Convert back to tensor
    output_normalized = torch.tensor(output_norm, dtype=output.dtype, device=output.device)
    
    return output_normalized





    
    
# preparation train-val-test pas encore de dataloader, # mise au format 365 jours + masque des données
def tvt_split(data): 
  mapping={1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10}
  #mapping = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13,15:14,16:15}

  data,msk=suppr(data,0) # peut être utiliser si l'on souhaite diminuer la quantité de points d'acquisition dans les données
  data,_,mask=comp(data,msk) # rempli les données pour les mettre au fromat 365 j et donne le mask correspondant aux jours où on a mit un 0
  values=data['X_SAR']
  data_shape=data['X_SAR'].shape
  dates=data['dates_SAR']




  labels=data['y']
  labels=[mapping[v] if v in mapping else v for v in labels ]

  max_values = np.percentile(values,99)
  min_values = np.percentile(values,1)
  values_norm=(values-min_values)/(max_values-min_values)
  values_norm[values_norm>1] = 1
  values_norm[values_norm<0] = 0
  values = values_norm                                      # les données sont normalisées
  values=add_mask(values,mask)   
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
  indice = sss.split(values,labels)

  tv_index, test_index = next(indice)

  values_tv=[]
  values_test=[]
  labels_tv=[]
  labels_test=[]
  for i in tv_index :
    values_tv+=[values[i]]
    labels_tv+=[labels[i]]
  for j in test_index :
    values_test+=[values[j]]
    labels_test+=[labels[j]]


  sss2=StratifiedShuffleSplit(n_splits=1,test_size=0.25,random_state=0)
  indice2=sss2.split(values_tv,labels_tv)
  train_index,validation_index = next(indice2)

  values_train=[]
  values_validation=[]
  labels_train=[]
  labels_validation=[]

  for i in train_index :
    values_train+=[values_tv[i]]
    labels_train+=[labels_tv[i]]
  for j in validation_index :
    values_validation += [values_tv[j]]
    labels_validation += [labels_tv[j]]


  values_train=np.array(values_train)
  values_validation=np.array(values_validation)
  values_test=np.array(values_test)
  labels_train=np.array(labels_train)
  labels_validation=np.array(labels_validation)
  labels_test=np.array(labels_test)
  
  data_train = {'X_SAR':values_train, 'y':labels_train, 'dates_SAR':dates}
  data_validation = {'X_SAR': values_validation, 'y':labels_validation, 'dates_SAR':dates}
  data_test = {'X_SAR': values_test,'y':labels_test, 'dates_SAR':dates}



  



  return data_train,data_validation,data_test,dates,data_shape
def selection(data,role, fraction=1, ): # fraction etant le pourcentage de donnée que l'on ouhaite consevrer
      if role == 'source':
        values, labels, dates = data['X_SAR'], data['y'], data['dates_SAR']
        nb_samples = int( fraction * len(labels))
        i_selected = np.random.choice(len(labels),nb_samples,replace=False)
        data_selected =  {'X_SAR':values[i_selected], 'y': labels[i_selected], 'dates_SAR': dates}
        return data_selected
      else :
        values, labels, dates = data['X_SAR'], data['y'], data['dates_SAR']
        nb_samples = int( fraction * len(labels))
        i_selected = np.random.choice(len(labels),nb_samples,replace=False)
        
        data_selected =  {'X_SAR':values[i_selected], 'y': torch.ones(nb_samples)*-1, 'dates_SAR': dates}
        return data_selected
        
   
    
def data_loading(datas,fraction=1):
    
  list_values_train = []
  list_labels_train = []
  list_domain_train = []

  list_values_test = []
  list_labels_test = []
  for i,data in enumerate (datas[0]):
    
    k=i
    data_train,_,_,dates,data_shape = tvt_split(data)
    data_train = selection(data_train, role="source", fraction = fraction)
    value_train,labels_train = data_train['X_SAR'],data_train['y']
    s= np.array(labels_train).shape
    
    list_values_train.append(value_train)
    list_labels_train.append(labels_train)
    list_domain_train.append(np.ones(s[0])*i)
  for data in datas[1] : # les données target sont ajoutées
            
    data_train,_,data_test,dates,data_shape = tvt_split(data)
    data_train = selection(data_train, role="target", fraction = fraction)
    value_train,labels_train = data_train['X_SAR'],data_train['y']
    
    value_test, labels_test = data_test['X_SAR'], data_test['y']
    
    s1 = np.array(labels_train).shape
               
    
    
    
    list_values_train.append(value_train)
    list_labels_train.append(labels_train)
    list_domain_train.append(np.ones(s1[0])*(k+1))
    
    list_values_test.append(value_test)
    list_labels_test.append(labels_test)
    
    
  list_labels_train = np.concatenate(list_labels_train,axis=0)
  list_values_train = np.concatenate(list_values_train,axis=0)
  list_domain_train = np.concatenate(list_domain_train,axis=0)
  
  list_values_train = torch.tensor(list_values_train,dtype=torch.float32)
  list_labels_train = torch.tensor(list_labels_train, dtype=torch.int64)
  list_domain_train = torch.tensor(list_domain_train,dtype=torch.int64)
  
  list_values_test = np.concatenate(list_values_test, axis=0)
  list_labels_test = np.concatenate(list_labels_test, axis=0)
  
  list_values_test = torch.tensor(list_values_test, dtype=torch.float32)
  list_labels_test = torch.tensor(list_labels_test,dtype=torch.int64)
  
 
       
  train_dataset = TensorDataset(list_values_train, list_labels_train, list_domain_train)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64,drop_last=True)
  
  test_dataset = TensorDataset(list_values_test, list_labels_test)
  test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=64,drop_last=True)
  
  return train_dataloader, test_dataloader, dates, data_shape
           
def data_loading2(datas,fraction=1):
    
  list_values_train = []
  list_labels_train = []
  list_domain_train = []

  
  for i,data in enumerate (datas[0]):
    
    k=i
    data_train,_,_,dates,data_shape = tvt_split(data)
    data_train = selection(data_train, role="source", fraction = fraction)
    value_train,labels_train = data_train['X_SAR'],data_train['y']
    s= np.array(labels_train).shape
    
    list_values_train.append(value_train)
    list_labels_train.append(labels_train)
    list_domain_train.append(np.ones(s[0])*i)
  
    
    
  list_labels_train = np.concatenate(list_labels_train,axis=0)
  list_values_train = np.concatenate(list_values_train,axis=0)
  list_domain_train = np.concatenate(list_domain_train,axis=0)
  
  list_values_train = torch.tensor(list_values_train,dtype=torch.float32)
  list_labels_train = torch.tensor(list_labels_train, dtype=torch.int64)
  list_domain_train = torch.tensor(list_domain_train,dtype=torch.int64)
  
  
  
 
       
  train_dataset = TensorDataset(list_values_train, list_labels_train, list_domain_train)
  train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=64,drop_last=True)
  
  
  
  return train_dataloader, dates, data_shape
