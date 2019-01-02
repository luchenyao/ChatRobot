# coding=utf-8

import csv
import random
import torch.nn as nn
from torch import optim
from data_process import loadLines,loadConversations,extractSentencePairs,printLines
from load_data import loadPrepareData,trimRareWords
from encoder import EncoderRNN
from word2vec import batch2TrainData
from attn_deocder import LuongAttnDecoderRNN,GreedySearchDecoder
from train_process import trainIters
from hyperparams import *
from evaluation import evaluateInput
#######################################################################################
# lines={}
# conversation=[]
#
# print("\nProcessing corpus...")
# lines=loadLines(os.path.join(corpus,"movie_lines.txt"),MOVIE_LINES_FIELDS)
# print("\nLoading conversations...")
# conversations=loadConversations(os.path.join(corpus,"movie_conversations.txt"),lines,MOVIE_CONVERSATION_FIELDS)

# print("\nWriting newly formatted file...")
# with open(datafile,'w',encoding='utf-8') as outputfile:
#     writer=csv.writer(outputfile,delimiter=delimiter,lineterminator='\n')
#     for pair in extractSentencePairs(conversations):
#         writer.writerow(pair)

print("\nSample lines from file:")
printLines(datafile)


# loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


#####################################################################################
'''load_data'''

# voc,pairs=loadPrepareData(corpus,corpus_name,datafile,save_dir)
#
# print("\npairs:")
# for pair in pairs[:10]:
#     print(pair)

pairs=trimRareWords(voc,pairs,MIN_COUNT)


#####################################################################################
'''word to vector'''

batchs=batch2TrainData(voc,[random.choice(pairs) for _ in range(small_batch_size)])
input_variable,lengths,target_variable,mask,max_target_len=batchs

print("input_variable:",input_variable)
print("lengths:",lengths)
print("target_variable:",target_variable)
print("mask:",mask)
print("max_target_len:",max_target_len)


#####################################################################################
# 从文件中读取模型参数
if loadFilename:
    checkpoint=torch.load(loadFilename)
    # 如果从GPU转换到CPU
    # checkpoint=torch.load(loadFilename,map_location=torch.device('cpu')
    encoder_sd=checkpoint['en']
    decoder_sd=checkpoint['de']
    encoder_optimizer_sd=checkpoint['en_opt']
    decoder_optimizer_sd=checkpoint['de_opt']
    embedding_sd=checkpoint['embedding']
    voc.__dict__=checkpoint['voc_dict']


print('Building encoder and decoder...')
# 初始化embedding
embedding=nn.Embedding(voc.num_words,hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# 初始化encoder和decoder参数
encoder=EncoderRNN(hidden_size,embedding,encoder_n_layers,dropout)
decoder=LuongAttnDecoderRNN(attn_model,embedding,hidden_size,voc.num_words,decoder_n_layers,dropout)

if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# 选择device
encoder=encoder.to(device)
decoder=decoder.to(device)

print("Models built and ready to go !")


encoder.train()
decoder.train()


# 初始化优化器
print('Building optimizers...')
encoder_optimizer=optim.Adam(encoder.parameters(),lr=learning_rate)
decoder_optimizer=optim.Adam(decoder.parameters(),lr=learning_rate)

if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

print("Starting Training!")

trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,save_every,clip,corpus_name,loadFilename)
searcher=GreedySearchDecoder(encoder,decoder)
evaluateInput(encoder=encoder,decoder=decoder,searcher=searcher)