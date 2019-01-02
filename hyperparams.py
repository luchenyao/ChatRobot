# coding=utf-8
import torch
import os
import codecs

#################################################################
# 选择是否使用GPU
USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")


# 文件路径
corpus_name='neural-knowledge-diffusion'
corpus=os.path.join("data",corpus_name)
save_dir=os.path.join("data","dialogue-save")


# 单词和词向量的参数
PAD_token=0
SOS_token=1
EOS_token=2

MAX_LENGTH=30
MIN_COUNT=1

MAX_TURN_NUM=10


# 清洗文件的参数
MOVIE_CONVERSATION_FIELDS=["movie_info","raw_sentence","celebrity","triple",'time']

datafile=os.path.join(corpus,"formatted_douban_data")
delimiter='\t'
delimiter=str(codecs.decode(delimiter,"unicode_escape"))

datafile_A="./data/neural-knowledge-diffusion/dialogue_turns_A"
datafile_B="./data/neural-knowledge-diffusion/dialogue_turns_B"
#################################################################
# 模型参数
clip=50.0
teacher_forcing_ratio=1.0
learning_rate=1e-4
decoder_learning_ratio=5.0
n_iteration=4000
print_every=1
save_every=500

model_name='cb_model'
attn_model='dot'
# attn_model='general'
# attn_model='concat'
hidden_size=512
encoder_n_layers=2
decoder_n_layers=2
dropout=0.1
batch_size=64
small_batch_size=5


checkpoint_iter=4000
loadFilename = os.path.join(save_dir, model_name, corpus_name,
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))
# loadFilename=None
##################################################################






