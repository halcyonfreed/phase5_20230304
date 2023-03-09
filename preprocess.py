'''
@author: halcyonfreed
@date: 20230308
@comment: 
        define my own dataset: 定义自己的dataset
        不考虑周围车辆nbrs，只考虑自车的历史轨迹！！
'''
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import scipy.io as scp #读mat文件

'''args={
    't_h':30, # length of track history
    't_f':50,# length of predicted trajectory
    # 'd_s':1,# down sampling rate of all sequences 采样的步长，原来是2,减少计算量，改成1?  https://blog.csdn.net/hxxjxw/article/details/106175155
    'enc_size':128,  # size of encoder LSTM
    'grid_size':(13,3) # size of social context grid 3根车道 前后6车，一共约13辆车的车长
}'''
class ngsimDataset(Dataset):
    def __init__(self,mat_file,args):
        # 见Readme.md——数据说明.md看datatype,shape,含义
        self.D=scp.loadmat(mat_file)['traj']
        self.T = scp.loadmat(mat_file)['tracks']
        self.t_h = args.t_h
        self.t_f = args.t_f
        # self.enc_size = args.encoder_size

    def __len__(self):
        return len(self.D)
    def __getitem__(self, idx): # idx 某辆车某一时刻
        # Get features of ego vehicles
        dsId= self.D[idx, 0].astype(int) #datasetId
        vehId = self.D[idx, 1].astype(int)
        t = self.D[idx, 2]
        egoRelativeTrajHist= self.getEgoHistory(vehId,t,dsId) 
        egoRelativeTrajFut = self.getEgoFuture(vehId,t,dsId)

        # self.D[idx,7]=1.0或者2.0, 默认是float所以要int()，np.eye默认是float所以astype(int)
        # 只看当前这一帧的意图 推测后面的啊！！！
        # egoLonEnc=np.eye(2)[int(self.D[idx,7])-1].astype(int) # normal [1,0] brake [0,1]
        # egoLatEnc=np.eye(3)[int(self.D[idx,6])-1].astype(int) # keep[1,0,0] left[0,1,0] right[0,0,1]
        # return egoRelativeTrajHist,egoRelativeTrajFut,egoLatEnc,egoLonEnc #[31,2] [50,2] [1,2]认为是2 [1,3]认为是3 维度不统一，没法用默认dataloader的collate_fn
        return egoRelativeTrajHist,egoRelativeTrajFut # [31,2] [50,2]

    def getEgoHistTraj(self,vehId,t,dsId):
        '''
        返回当前时刻车相对自车初始的位置Δx,Δy,Hist含当前时刻
        egoRelativeTrajHist [end-start,2] 一般是[31,2]
        '''
        egoTrajAll=self.T[dsId-1][vehId-1].transpose() # [变长,4]
        egoTrajNow=egoTrajAll[np.where(egoTrajAll[:,0]==t)][0,1:3] # 只取这辆车最开始时刻的x,y 作为当前时刻的位置
        start=np.maximum(0,np.argwhere(egoTrajAll[:,0]==t).item()-self.t_h)
        end=np.argwhere(egoTrajAll[:,0]==t).item()+1 #要加1 因为下面a[1:3]实际是不含3的，小心！！
        egoRelativeTrajHist=egoTrajAll[start:end,1:3]-egoTrajNow # 含当前时刻
        return egoRelativeTrajHist 

    def getEgoFutureTraj(self,vehId,t,dsId):
        '''
        返回当前时刻车相对自车初始的位置Δx,Δy
        egoRelativeTrajFut [end-start,2] 一般是[50,2]
        '''
        egoTrajAll=self.T[dsId-1][vehId-1].transpose() # [变长,4]
        egoTrajNow=egoTrajAll[np.where(egoTrajAll[:,0]==t)][0,1:3] # 只取这辆车最开始时刻的x,y 作为当前时刻的位置
        start=np.argwhere(egoTrajAll[:,0]==t).item()
        end=np.minimum(len(egoTrajAll),np.argwhere(egoTrajAll[:,0]==t).item()+self.t_f+1)#要加1，因为实际a[1:3]，取不到3
        egoRelativeTrajFut=egoTrajAll[start+1:end,1:3]-egoTrajNow #不含当前时刻
        return egoRelativeTrajFut

    def collate_fn(self,samples):
        '''
        samples已经是batch了，但是要把他变成要的real_batch
        自定义collate_fn 把数据打包成batch的方法, https://blog.csdn.net/qq_43391414/article/details/120462055
        通常我们并不需要个函数,因为pytorch内部有一个默认的。但是如果你的数据不规整使用默认的会报错。例如下面的数据。
        假设我们还是4个输入,但是维度不固定的。之前我们是每一个数据的维度都为3。   a=[[1,2],[3,4,5],[1],[3,4,9]]
        这里egoRelativeTrajHist,egoRelativeTrajFut,egoLatEnc,egoLonEnc #[31,2] [50,2] [1,2]认为是[2],[1,3]认为是[3] 维度不统一有2维也有1维的，没法用默认dataloader的collate_fn
        '''
        # 1 把这四个打包成batch，初始化batch的tensor维度：egoRelativeTraj_Hist,egoRelativeTraj_Fut,egoLatEnc,egoLonEnc #[31,2] [50,2] [2,2] [3,3] 
        maxEgoHistLength=self.t_h+1 # 31 含当前时刻
        egoRelativeTrajHist_batch=torch.zeros(maxEgoHistLength,len(samples),2) # [31,batch_size,2]
        egoRelativeTrajFut_batch=torch.zeros(self.t_f,len(samples),2) ## [50,batch_size,2]
        # egoLatEnc_batch = torch.zeros(len(samples),3) # [batch_size,3]
        # egoLonEnc_batch = torch.zeros(len(samples), 2) # [batch_size,2] 
        # output_batch = torch.zeros(self.t_f,len(samples),2) # [50,batch_size,2]

        # 2 把最后要的batch 从sample读进来，同时从ndarray格式转成tensor
        # for sampleId,(hist, fut, lat_enc, lon_enc) in enumerate(samples):            
        for sampleId,(hist, fut) in enumerate(samples):
            egoRelativeTrajHist_batch[0:len(hist),sampleId,0] = torch.from_numpy(hist[:, 0]) # 0存x
            egoRelativeTrajHist_batch[0:len(hist), sampleId, 1] = torch.from_numpy(hist[:, 1]) # 1存y
            
            egoRelativeTrajFut_batch[0:len(fut), sampleId, 0] = torch.from_numpy(fut[:, 0])
            egoRelativeTrajFut_batch[0:len(fut), sampleId, 1] = torch.from_numpy(fut[:, 1])

            # egoLatEnc_batch[sampleId,:] = torch.from_numpy(lat_enc) 
            # egoLonEnc_batch[sampleId, :] = torch.from_numpy(lon_enc)
            # output_batch[0:len(fut),sampleId,:] = 1 # 可改 默认ouput的x,y都设成1 为什么要output_batch
        # return egoRelativeTrajHist_batch,egoRelativeTrajFut_batch,egoLatEnc_batch,egoLonEnc_batch,output_batch #[31,batch_size,2]，[50,batch_size,2]，[batch_size,3],[batch_size,2],[50,batch_size,2]
        return egoRelativeTrajHist_batch,egoRelativeTrajFut_batch #[31,batch_size,2]，[50,batch_size,2]