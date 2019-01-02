# coding=utf-8

import torch
import itertools
import jieba
from hyperparams import *
from load_data import loadPrepareData


def indexesFromSentence(voc,sentences):
    indexes=[]
    num_turns=0
    for sentence in sentences.split('\t'):
        indexes.append([voc.word2index[word] for word in jieba.cut(sentence,cut_all=False)] + [EOS_token])
        num_turns+=1
    real_length=[(len(eachrow)-1) for eachrow in indexes]
    for _ in range(MAX_TURN_NUM-num_turns):
        real_length.append(0)
    if num_turns<MAX_TURN_NUM:
        for i in range(MAX_TURN_NUM-num_turns):
            indexes.append([])
    for j in range(len(indexes)):
        for k in range(len(indexes[j]),MAX_LENGTH):
            indexes[j].append(0)
    return indexes,num_turns,real_length


# voc,data_A,data_B=loadPrepareData(corpus=corpus,corpus_name=corpus_name,datafile_A=datafile_A,datafile_B=datafile_B,save_dir=save_dir)
#
# indexes,num_turns,real_length=indexesFromSentence(voc,data_A[0])
# print(data_A[0])
# print(indexes)
# print(num_turns)
# print(real_length)


def binaryMatix(l,value=PAD_token):
    m=[]
    for i,seq in enumerate(l):
        m.append([])
        for token in seq:
            if token==PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


def inputVar(voc,l):
    indexes_batch=[]
    num_turns_list=[]
    real_length_list=[]
    for sentences in l:
        indexes, num_turns, real_length=indexesFromSentence(voc,sentences)
        indexes_batch.append(indexes)
        num_turns_list.append(num_turns)
        real_length_list.append(real_length)

    indexes_batch_tensor=torch.LongTensor(indexes_batch)
    return indexes_batch_tensor,num_turns_list,real_length_list


def outputVar(l,voc):
    indexes_batch = []
    num_turns_list = []
    real_length_list = []
    for sentences in l:
        indexes, num_turns, real_length = indexesFromSentence(voc, sentences)
        indexes_batch.append(indexes)
        num_turns_list.append(num_turns)
        real_length_list.append(real_length)

    indexes_batch_tensor = torch.LongTensor(indexes_batch)
    return indexes_batch_tensor, num_turns_list, real_length_list


def batch2TrainData(voc,data_A,data_B):
    data_combine=[]
    if(len(data_A)==len(data_B)):
        for i in range(len(data_A)):
            data_combine.append([])
            data_combine[i]=data_A[i].strip()+'###'+data_B[i].strip()

    # data_combine.sort(key=lambda x:len(x[0].split('###')),reverse=True)
    # print(data_combine[:10])
    input_batch,output_batch=[],[]
    for dialogue in data_combine:
        sentences=dialogue.split('###')
        input_batch.append(sentences[0])
        output_batch.append(sentences[1])

    input_indexes_batch_tensor, input_num_turns_list, input_real_length_list=inputVar(voc,data_A)
    output_indexes_batch_tensor, output_num_turns_list, output_real_length_list = outputVar(voc, data_B)
    # print(input_batch[:10])
    # print(output_batch[:10])
    # inp,lengths=inputVar(input_batch,voc)
    # output,mask,max_target_len=outputVar(output_batch,voc)
    # return inp,lengths,output,mask,max_target_len


#voc,data_A,data_B=loadPrepareData(corpus=corpus,corpus_name=corpus_name,datafile_A=datafile_A,datafile_B=datafile_B,save_dir=save_dir)
# data_A=open(datafile_A,'r',encoding='utf-8').readlines()
# data_B=open(datafile_B,'r',encoding='utf-8').readlines()
#batch2TrainData(voc,data_A,data_B)


# def cal(data_A):
#     num=[]
#     for eachrow in data_A:
#         l=eachrow.split('\t')
#         for sentence in
#         length=len(l)
#         num.append(length)
#     num.sort(reverse=True)
#     print(num)
#
# cal(data_A)
