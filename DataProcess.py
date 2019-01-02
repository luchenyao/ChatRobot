# coding=utf-8

import torch
import json
from hyperparams import *
import csv
import numpy as np
import os
# import jieba
from TranE.tranE import TransE
from load_data import Voc


"""
构建词表
"""
def construct_voc(path,corpus_name):
    voc = Voc(corpus_name)
    with open(path,'r') as f:
        content=f.readlines()
        for eachline in content:
            dialogue=json.loads(eachline)
            for eachsentence in dialogue:
                voc.addSentence(eachsentence['raw_sentence'])
    print(voc.num_words)
    for i in range(voc.num_words):
        print(voc.index2word[i])
    return voc


"""
将文件中的对话对提取出来并输出成文件。
"""
def extract_dialog_turns(path):
    multi_turns=[]
    with open(path,'r') as f:
        content=f.readlines()
        for i in range(len(content)):
            multi_turns.append([])
            dialogue=json.loads(content[i])
            for j in range(len(dialogue)):
                multi_turns[i].append(dialogue[j]['raw_sentence'])
    print(multi_turns[:10])
    multi_turns_A=[]
    multi_turns_B=[]
    for i in range(len(multi_turns)):
        multi_turns_A.append([])
        multi_turns_B.append([])
        for j in range(len(multi_turns[i])):
            if j%2==0:
                multi_turns_A[i].append(multi_turns[i][j])
            else:
                multi_turns_B[i].append(multi_turns[i][j])
        if (len(multi_turns_A[i])-len(multi_turns_B[i]))==1:
            multi_turns_A[i].pop(len(multi_turns_A[i])-1)
        if (len(multi_turns_A[i])-len(multi_turns_B[i]))==-1:
            multi_turns_B[i].pop(len(multi_turns_B[i])-1)
    print(multi_turns_A[:10])
    print(multi_turns_B[:10])
    save_path_A = "./data/neural-knowledge-diffusion/dialogue_turns_A"
    save_path_B = "./data/neural-knowledge-diffusion/dialogue_turns_B"
    with open(save_path_A, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in multi_turns_A:
            writer.writerow(pair)
    with open(save_path_B, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in multi_turns_B:
            writer.writerow(pair)

# extract_dialog_turns(".\\data\\neural-knowledge-diffusion\\data.experiment")


"""
把data.experiment里的关键信息提取出来
"""
def extract_keywords(path):
    key_features=[]
    save_key=".\\data\\neural-knowledge-diffusion\\dialogue_features"
    with open(path,'r') as f:
        content=f.readlines()
        for i in range(len(content)):
            key_features.append([])
            dialogue=json.loads(content[i])
            for j in range(len(dialogue)):
                if str(dialogue[j]['movie'])!='[]':
                    key_features[i].append(dialogue[j]['movie'])
                if str(dialogue[j]['celebrity'])!='[]':
                    key_features[i].append(dialogue[j]['celebrity'])
                if str(dialogue[j]['triple'])!='[]':
                    key_features[i].append(dialogue[j]['triple'])
                if str(dialogue[j]['time'])!='[]':
                    key_features[i].append(dialogue[j]['time'])
    with open(save_key, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in key_features:
            writer.writerow(pair)
    print(key_features)

# extract_keywords(".\\data\\neural-knowledge-diffusion\\data.experiment")


def extract_KB(path):
    triple = []
    entity=[]
    relation=['direct_by','act_by','release_time_is']
    save_KB = "./data/neural-knowledge-diffusion/Knowledge_base"
    with open(path, 'r') as f:
        contents = f.readlines()
        for content in contents:
            dialogue = json.loads(content)

            for key,value in dialogue['celebrity'].items():
                # entity.append(key)
                if value['name']!='':
                    entity.append(value['name'])
                # triple.append([key, 'id_is', value['id']])
                # triple.append([key,'name_is',value['name']])

            for key,value in dialogue['movie'].items():
                # entity.append(key)
                if value['title'] != '':
                    entity.append(value['title'])
                if value['release_time'] != '':
                    entity.append(value['release_time'])
                for item_director in value['director']:
                    if dialogue['celebrity'][item_director]['name'] != '':
                        entity.append(dialogue['celebrity'][item_director]['name'])
                        triple.append([value['title'],'direct_by',dialogue['celebrity'][item_director]['name']])
                for item_actor in value['actor']:
                    if dialogue['celebrity'][item_actor]['name'] != '':
                        entity.append(dialogue['celebrity'][item_actor]['name'])
                        triple.append([value['title'], 'act_by', dialogue['celebrity'][item_actor]['name']])
                if value['release_time'] != '':
                    triple.append([value['title'], 'release_time_is', value['release_time']])

    return entity,relation,triple

    # 写入文件
    # with open(save_key, 'w', encoding='utf-8') as outputfile:
    #     writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    #     for pair in key_features:
    #         writer.writerow(pair)
    # print(key_features)


entity_list,relation_list,triple_list=extract_KB("./data/neural-knowledge-diffusion/kb.experiment")

transE = TransE(entity_list,relation_list,triple_list,learingRate=0.0001,margin=1, dim=512,L1=False)
print("TranE初始化")
transE.initialize()
transE.transE(30000)


























"""
把文件print到控制台
"""

# with open('C:\\Private\\Code\\NLP\\KB-chatbox\\data\\neural-knowledge-diffusion\\data.experiment') as f:
#     content=f.readlines()
#     for eachrow in content[:10]:
#         dialogue=json.loads(eachrow)
#         print(dialogue)


"""
将json文件转换成list格式并输出
"""
def loadfile(filename):
    data=[]
    with open(filename,'r') as f:
        file=f.readlines()
        for i in range(len(file)):
            data.append([])
            dialogue = json.loads(file[i])
            for j in range(len(dialogue)):
                data[i].append([dialogue[j]['movie'],dialogue[j]['raw_sentence'],
                               dialogue[j]['celebrity'], dialogue[j]['triple'], dialogue[j]['time']])
        print(data)

    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for row in data:
            writer.writerow(row)
# loadfile(os.path.join(corpus,"data.experiment"))


