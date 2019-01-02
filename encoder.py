# coding=utf-8

import torch
import torch.nn as nn


class EncoderRNN(nn.Module):
    def __init__(self,hidden_size,embedding,n_layers=1,dropout=0):
        super(EncoderRNN,self).__init__()
        self.n_layers=n_layers
        self.hidden_size=hidden_size
        self.embedding=embedding

        self.gru=nn.GRU(hidden_size,hidden_size,n_layers,dropout=(0 if n_layers==1 else dropout),bidirectional=True)

    def forward(self,input_seq,input_length,hidden=None):
        embedded=self.embedding(input_seq)
        packed=torch.nn.utils.rnn.pack_padded_sequence(embedded,input_length)
        outputs,hidden=self.gru(packed,hidden)
        outputs,_=torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # 双向RNN，把两段的向量加起来
        outputs=outputs[:,:,:self.hidden_size]+outputs[:,:,self.hidden_size:]
        return outputs,hidden

