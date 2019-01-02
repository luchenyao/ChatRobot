# coding=utf-8

import torch
import torch.nn.functional as F
import torch.nn as nn
from hyperparams import device,SOS_token


class Attn(torch.nn.Module):
    def __init__(self,method,hidden_size):
        super(Attn,self).__init__()
        self.method=method
        if self.method not in ['dot','general','concat']:
            raise ValueError(self.method,"is not an appropriate attention method.")
        self.hidden_size=hidden_size
        if self.method=='dot':
            self.attn=torch.nn.Linear(self.hidden_size,hidden_size)
        elif self.method=='general':
            self.attn=torch.nn.Linear(self.hidden_size*2,hidden_size)
        elif self.method=='concat':
            self.attn=torch.nn.Linear(self.hidden_size*2,hidden_size)
            self.v=torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self,hidden,encoder_output):
        return torch.sum(hidden*encoder_output,dim=2)

    def general_score(self,hidden,encoder_output):
        energy=self.attn(encoder_output)
        return torch.sum(hidden*energy,dim=2)

    def concat_score(self,hidden,encoder_output):
        energy=self.attn(torch.cat((hidden.expand(encoder_output.size(0),-1,-1),encoder_output),2)).tanh()
        return torch.sum(self.v*energy,dim=2)

    def forward(self,hidden,encoder_outputs):
        if self.method=='general':
            attn_energies=self.general_score(hidden,encoder_outputs)
        elif self.method=='concat':
            attn_energies=self.concat_score(hidden,encoder_outputs)
        elif self.method=='dot':
            attn_energies=self.dot_score(hidden,encoder_outputs)

        # 把矩阵进行转置，把max_length和batch_size两个维度转置
        attn_energies=attn_energies.t()
        return F.softmax(attn_energies,dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self,attn_model,embedding,hidden_size,output_size,n_layres=1,dropout=0.1):
        super(LuongAttnDecoderRNN,self).__init__()
        self.attn_model=attn_model
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layres
        self.dropout=dropout

        self.embedding=embedding
        self.embedding_dropout=nn.Dropout(dropout)
        self.gru=nn.GRU(hidden_size,hidden_size,n_layres,dropout=(0 if n_layres==1 else 0))
        self.concat=nn.Linear(hidden_size*2,hidden_size)
        self.out=nn.Linear(hidden_size,output_size)
        self.attn=Attn(attn_model,hidden_size)

    def forward(self,input_step,last_hidden,encoder_outputs):
        # 一次执行一个单词
        embedded=self.embedding(input_step)
        embedded=self.embedding_dropout(embedded)
        # 解码器中用单向gru
        rnn_output,hidden=self.gru(embedded,last_hidden)
        # 计算attention的权重
        attn_weights=self.attn(rnn_output,encoder_outputs)
        # bmm:batch matrices multiply
        context=attn_weights.bmm(encoder_outputs.transpose(0,1))
        rnn_output=rnn_output.squeeze(0)
        context=context.squeeze(1)
        concat_input=torch.cat((rnn_output,context),1)
        concat_output=torch.tanh(self.concat(concat_input))
        output=self.out(concat_output)
        output=F.softmax(output,dim=1)

        return output,hidden


class GreedySearchDecoder(nn.Module):
    def __init__(self,encoder,decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder=encoder
        self.decoder=decoder

    def forward(self,input_seq,input_length,max_length):
        # encoder模型的前向输入
        encoder_outputs,encoder_hidden=self.encoder(input_seq,input_length)
        # encoder中的输出作为decoder的输入
        decoder_hidden=encoder_hidden[:self.decoder.n_layers]
        # 用SOS token 初始化decoder
        decoder_input=torch.ones(1,1,device=device,dtype=torch.long)*SOS_token
        # 初始化要加入解码单词的张量
        all_tokens=torch.zeros([0],device=device,dtype=torch.long)
        all_scores=torch.zeros([0],device=device)

        # 迭代decoder输出单词
        for _ in range(max_length):
            decoder_output,decoder_hidden=self.decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_scores,decoder_input=torch.max(decoder_output,dim=1)
            all_tokens=torch.cat((all_tokens,decoder_input),dim=0)
            all_scores=torch.cat((all_scores,decoder_scores),dim=0)

            decoder_input=torch.unsqueeze(decoder_input,0)

        return all_tokens,all_scores





