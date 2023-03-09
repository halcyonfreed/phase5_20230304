'''
@author: halcyonfreed
@date: 20230309
@comment: 
        define vlstm seq2seq model
        即经典的seq2seq框架之一，encoder-decoder，都是lstm
        不考虑周围车辆  即不考虑interaction/social都是各开各的
'''

import torch
import torch.nn as nn
import random

def outputActivation(x): #x是pred:[outlength,batch_size,5]
    #五个参数都是[outlength,batch_size,1]
    muX = x[:,:,0] 
    muY = x[:,:,1]
    sigX = x[:,:,2]
    sigY = x[:,:,3]
    rho = x[:,:,4]
    # 变换 ux,uy->e^(ux),e^(uy),σx,σy,ρ=tanh(ρ)相关系数
    sigX = torch.exp(sigX)  
    sigY = torch.exp(sigY)
    rho = torch.tanh(rho) #有点像二元正态分布
    out = torch.cat([muX, muY, sigX, sigY, rho],dim=2) #按dim=2 串起来,dim0和1的维度一样 
    return out #[outlength,batch_size,5]


'''
dataloader传给打包好的数据维度:
egoRelativeTrajHist_batch [batch_size,31,2]
egoRelativeTrajFut_batch [batch_size,50,2]
# egoLatEnc_batch [batch_size,3]
# egoLonEnc_batch [batch_size,2]
# output_batch [batch_size,50,2]
'''
class Seq2SeqLstm(nn.Module):
    # seq2seq的enc-dec 且无nbrs的grid social/interaction信息 vanilla-lstm
    def __init__(self,**args) :
        super(Seq2SeqLstm,self).__init__()
        # self.use_maneuvers=args.use_maneuvers #true就是multi-modal的多条输出，false就是single-modal一条输出
        self.teacher_forcing_rate=args.teacher_forcing_rate
        
        ## 定义每层网络自己本身的参数
        self.encoderhidden_size = args.encoder_size
        self.decoderhidden_size = args.decoder_size
        self.out_length = args.out_length # 50
        self.input_embedding_size = args.input_embedding_size
        # self.num_lat_classes = args['num_lat_classes']
        # self.num_lon_classes = args['num_lon_classes']

        ## 定义每层网络是什么
        # input embedding+encoder
        self.input_emb=nn.Linear(2,self.input_embedding_size) # 输入编码换维度为啥呢，可以不要吗？ 输入：egoRelativeTrajHist_batch [31,batch_size,2]
        self.enc_lstm = nn.LSTM(self.input_embedding_size,self.encoderhidden_size,1) # lstm第一个和input最后一个一样 [input_size=input_embedding_size,hidden_size=encoder_size,num_layers*directions=1*1] 
        #----decoder
        self.enc_ht_emb=nn.Linear(self.encoderhidden_size,self.decoderhidden_size)
        self.last_dec_ot_emb=nn.Linear(self.decoderhidden_size,self.encoderhidden_size)
        self.dec_ip_emb=nn.Linear(2,self.encoderhidden_size) # 所以2->encoderhidden_size
        self.dec_lstm=nn.LSTM(self.encoderhidden_size,self.decoderhidden_size,1) #lstm第一个和input最后一个一样
        # self.dec_lstm2=nn.LSTM(self.encoderhidden_size+5,self.decoderhidden_size,1)
        #----输出层
        self.op=nn.Linear(self.decoderhidden_size,5) #为什么是5,因为是二维高斯分布的5个参数啊！！
        # self.oplat=nn.Linear(self.encoderhidden_size,self.num_lat_classes)
        # self.oplon=nn.Linear(self.encoderhidden_size, self.num_lon_classes)        
        #----ctivations:（neurons）
        self.leaky_relu =nn.LeakyReLU(0.1) #negative_slope=0.1
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) # 在dim=1这一维上归一化，这一维sum=1,都是算出来的概率值

    def forward(self,egoHistTraj,egoFutTraj,lastDecOutput=None):
        ''' step1 encoder
        fc->relu->lstm取h_n->fc->leaky_relu 核心只有lstm取h_n其他都可以没有
        enc_input=egoHistTraj
        enc_hidden=encoder的h0初始值 自己给 不给也可以默认会给合适大小的0, 所以这里图方便就不给了!!
        dec_input=egoFutTraj teacher forcing要用
        dec_hidden=enc输出的ht
        '''
        egoHistTraj=egoHistTraj.transpose(0,1) # [31,batch_size,2]
        egoFutTraj=egoFutTraj(0,1) # [50,batch_size,2]
        x1=self.input_emb(egoHistTraj) # [31,batch_size,2]->[31,batch_size,input_embedding_size]
        x2=self.relu(x1)
        # lstm输入格式(input,h0,c0) 其中h0 c0默认给0
        # input：[seq_len=31,batch_size,input=input_embedding_size]； 
        # hn格式= [num_layers(=1) * num_directions(=1), batch_size, n_hidden]
        # 输出格式(output, (h_n, c_n)) h是hidden c是cell,只取encoder的最后一时刻最后一层的h_n作为decoder的输入！！！
        _,(egoHistTrajEnc,_)=self.enc_lstm(x2)  # [1,batch_size,encoderhidden_size]
        egoHistTrajEnc=self.leaky_relu(egoHistTrajEnc)
        
        ''' step2 decoder
        1. 取encoder输出的h_n  再lstm+teacher forcing 最基础版decoder的lstm input=FutTraj,h0=enc的ht
        !!!这里和原版不一样,加了teacher forcing机制
        原nn.LSTM(input=egoHistTrajEnc,h0=0,c0=0)就是错的,根本不是seq2seq的正确做法
        现在nn.LSTM(input=egoFutTraj,h0=egoHistTrajEnc,c0=0)
        其中input [seq_len,batchsize,input_size]=[out_length,batch_size,encoderhidden_size]
        output [seq_len,batchsize,num_directions*hidden_size]
        
        2. 随机使用teacherforcing 给decoder看正确答案纠正错误 和自回归把上一时刻decoder的output作为这一时刻的decoder input
        3. 加上lon lat maneuver 是最终版
        
        # use_teacher_forcing=True if random.random()<self.teacherforcingrate else False
        # if use_teacher_forcing:

        # else:
        
        # if self.use_maneuvers:
        #     # 加上maneuver的v2
        #     lat_pred=self.oplat(egoHistTrajEnc) # [batch_size,encoderhidden_size]->[batch_size,num_lat_classes]
        #     lat_pred=self.softmax(lat_pred) # 要吗 因为crossentropyloss自带softmax的？？？？
        #     lon_pred=self.oplon(egoHistTrajEnc) # [batch_size,encoderhidden_size]->[batch_size,num_lon_classes]
        #     lon_pred=self.softmax(lon_pred)
        #     # train flag删了，没必要
        #     # 根据每种意图,一共3*2六种意图组合，生成轨迹
        #     pred=[]
        #     for k in range(self.num_lon_classes):
        #         for l in range(self.num_lat_classes):
        #             lat_enc_tmp = torch.zeros_like(egoLatEnc)  #[batch_size,3]
        #             lon_enc_tmp = torch.zeros_like(egoLonEnc)  #[batch_size,2]
        #             lat_enc_tmp[:, l] = 1
        #             lon_enc_tmp[:, k] = 1
                    
        #             # 按dim=1连起来,dim=0的维度一样->[batch_size,encoderhidden_size+3+2]
        #             enc_tmp = torch.cat((egoHistTrajEnc, lat_enc_tmp, lon_enc_tmp), 1) 
        #             enc_tmp=enc_tmp.repeat(self.out_length,1,1) # [batch_size,encoderhidden_size+5]->[out_length*1,1*batch_size,1*(encoderhidden_size+5)]
        #             dec_input=torch.cat((egoFutTraj, egoLatEnc, egoLonEnc), 1) #[]
                    
        #             output_dec,(_,_)=self.dec_lstm2(egoFutTraj,enc_tmp) # [out_length,batch_size,encoderhidden_size]->[out_length,batchsize,decoderhidden_size]
        #             output_dec=output_dec.permute(1,0,2) #[out_length,batchsize,decoderhidden_size]->[batch_size,outlength,decoderhiddensize]
        #             pred=self.op(output_dec) #[batch_size,outlength,decoderhiddensize]->[batch_size,outlength,5]
        #             pred=pred.permute(1,0,2) # [batch_size,outlength,5]->[outlength,batch_size,5]
        #             pred=outputActivation(pred) # [outlength,batch_size,5]经过数学变换取指数/tanh 虽然不知道为什么要这么搞
        #             # return pred #[outlength,batch_size,5]


        #             pred.append(self.decoder(egoFutTraj,enc_tmp))
        #     # [6,batch_size,encoderhidden_size+3+2],[batch_size,num_lat_classes],[batch_size,num_lon_classes]
        #     return pred, lat_pred, lon_pred 
        # else:
        '''        
        # v1：只用了traj，没用lateral/longitudinal maneuver
        use_teacher_forcing=True if random.random()<self.teacher_forcing_rate else False
        if use_teacher_forcing: #decoder用output的真值
            # lstm (input,h0,c0)
            # input [seq_len,batch,input_size]=[50=out_length,batch_size,encoderhidden_size]
            # h0 [layers,batchsize,hidden_size] h0=egoHistTrajEnc: [1,batchsize,decoderhidden_size]

            egoFutTrajEnc=self.dec_ip_emb(egoFutTraj) # [50=out_length,batch_size,2]->[50,batch_size,encoderhidden_size]
            egoHistTrajEnc=self.enc_ht_emb(egoHistTrajEnc) # [1,batchsize,decoderhidden_size]
            output_dec,(_,_)=self.dec_lstm(egoFutTrajEnc,egoHistTrajEnc) # [out_length,batchsize,decoderhidden_size]
            dec_output=self.last_dec_ot_emb(output_dec) # [out_length,batchsize,decoderhidden_size]->[out_length,batchsize,encoderhidden_size]
            output_dec=output_dec.permute(1,0,2) # [out_length,batchsize,decoderhidden_size]->[batch_size,outlength,decoderhiddensize]
            
            pred=self.op(output_dec) #[batch_size,outlength,decoderhiddensize]->[batch_size,outlength,5]
            pred=pred.permute(1,0,2) # [batch_size,outlength,5]->[outlength,batch_size,5]
            pred=outputActivation(pred) # [outlength,batch_size,5]经过数学变换取指数/tanh 虽然不知道为什么要这么搞
            return pred,dec_output #[outlength,batch_size,5],[out_length,batchsize,encoderhidden_size]
        else:
            # 用自回归 上一epoch的变成这一epoch的输入
            last_dec_output=lastDecOutput # [out_length,batchsize,encoderhidden_size]
            egoHistTrajEnc=self.enc_ht_emb(egoHistTrajEnc) # [1,batchsize,decoderhidden_size]
            output_dec,(_,_)=self.dec_lstm(last_dec_output,egoHistTrajEnc) # [out_length,batchsize,decoderhidden_size]
            dec_output=self.last_dec_ot_emb(output_dec) # [out_length,batchsize,decoderhidden_size]->[out_length,batchsize,encoderhidden_size]
            pred=output_dec.permute(1,0,2) #[out_length,batchsize,decoderhidden_size]->[batch_size,outlength,decoderhiddensize]
            
            pred=self.op(output_dec) #[batch_size,outlength,decoderhiddensize]->[batch_size,outlength,5]
            pred=pred.permute(1,0,2) # [batch_size,outlength,5]->[outlength,batch_size,5]
            pred=outputActivation(pred) # [outlength,batch_size,5]经过数学变换取指数/tanh 虽然不知道为什么要这么搞
            return pred,dec_output #[outlength,batch_size,5],[out_length,batchsize,encoderhidden_size]

        


        
