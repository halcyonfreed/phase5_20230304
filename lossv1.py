'''
@author: halcyonfreed
@date: 20230309
@comment: 
        define loss RMSE 不考虑lateral/longitudinal maneuver
'''

import torch
import math

# 官方是mse loss
# mse,mae,huber loss: https://blog.csdn.net/weixin_43229348/article/details/119760831

def RMSE(pred,gt):
    # 均方根
    length=gt.shape[1] #就是一个batch的大小batchsize
    x_pred=pred[:,:,0]
    y_pred=pred[:,:,1]
    x_gt=gt[:,:,0]
    y_gt=gt[:,:,1]
    # 单位m ，ft转成m所以0.3048
    loss=torch.pow(x_gt-x_pred, 2) + torch.pow(y_gt-y_pred, 2)
    loss=0.3048*torch.pow((loss),0.5)
    loss=loss.div(length)
    return loss
