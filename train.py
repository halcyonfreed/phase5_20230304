import math
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
from model import Seq2SeqLstm
from lossv1 import RMSE
from preprocess import ngsimDataset
import time
import os
import argparse

# ----1 parameters----
args={
    'input_embedding_size':  32, # 为什么
    'epoch'
    'batch_size':512 #原来是128 太小了
}
parser = argparse.ArgumentParser(description="Process some parameters.")
# parser.add_argument('--train_flag',default=False,action='store_true')  # true=train,false=true
parser.add_argument('--teacher_forcing_rate', default=0.6, type=float, help='Used in the decoder input when training.')

parser.add_argument('--encoder_size', default=64, type=int)
parser.add_argument('--decoder_size', default=128, type=int)
parser.add_argument('--out_length', default=50, type=int)
parser.add_argument('--input_embedding_size', default=64, type=int)
parser.add_argument('--t_h', default=30, type=int, help='length of track history, seconds * sampling rate')
parser.add_argument('--t_f', default=50, type=int, help='length of predicted trajectory, seconds * sampling rate')

parser.add_argument('--learning_rate', default=1e-3, type=float, help='')
parser.add_argument('--epochs', default=10, type=int, help='')
parser.add_argument('--batch_size', default=512, type=int, help='')
args = parser.parse_args()

# ----2 load data----
trainset = ngsimDataset('data/TrainSet.mat')
validset = ngsimDataset('data/ValSet.mat')
# 打包dataloader里面的collate_fn用自定义的！
trainloader = DataLoader(trainset,batch_size=args.batch_size,shuffle=True) #,num_workers=128)
validloader = DataLoader(validset,batch_size=args.batch_size,shuffle=True) #,num_workers=128,collate_fn=validset.collate_fn)

# ----3 train----
model=Seq2SeqLstm(**args.__dict__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-2)
batch_size = args.batch_size
writer=SummaryWriter() # 画loss图
n_epochs,best_loss,step, early_stop_count=args.epochs,math.inf,0,0

for epoch in range(n_epochs):
    train_pbar=tqdm(trainloader,position=0.leave=True)
    loss_record=[]
    model.train()
    for data in tqdm(train_pbar):
        egohist,egofut=data
        egohist,egofut=egohist.to(device),egofut.to(device)
        pred=model(egohist,egofut)
        
        loss=RMSE(pred,egofut)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        step+=1
        loss_record.append(loss.detach().item())

        train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})
    mean_train_loss=sum(loss_record)/len(loss_record)    
    writer.add_scalar('Loss/train', mean_train_loss, step)

    model.eval()
    loss_record = []
    for data in tqdm(validloader):
        egohist,egofut=data
        egohist,egofut=egohist.to(device),egofut.to(device)        
        with torch.no_grad():
            pred=model(egohist,egofut)
            loss = RMSE(pred, egofut)

        loss_record.append(loss.item())
    mean_valid_loss = sum(loss_record)/len(loss_record)
    print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
    writer.add_scalar('Loss/valid', mean_valid_loss, step)

    if mean_valid_loss < best_loss:
        best_loss = mean_valid_loss
        # ----save model----
        file_time=time.strftime('%Y-%m-%d-%H-%M-%S')
        if not os.path.isdir('model_results'):
            os.mkdir('model_results')
        model_path='model_results/Seq2SeqLstm'+file_time+'.ckpt' 
        # model_path=os.path.join('model_results','cslstm_m',file_time,'.ckpt')#这样的结果是models/cslstm_m/2022-12-06-14-34-53/.ckpt
        torch.save(model.state_dict(), model_path)
        
        torch.save(model.state_dict(),args.save_path) # Save your best model
        print('Saving model with loss {:.3f}...'.format(best_loss))
        early_stop_count = 0
    else: 
        early_stop_count += 1

    if early_stop_count >= args.early_stop:
        print('\nModel is not improving, so we halt the training session.')
        return





