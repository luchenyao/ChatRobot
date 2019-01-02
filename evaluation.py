# coding=utf-8

import torch
from hyperparams import device,MAX_LENGTH
from word2vec import indexesFromSentence
from load_data import normalizeStrings


def evaluate(encoder,decoder,searcher,voc,sentence,max_length=MAX_LENGTH):
    indexes_batch=[indexesFromSentence(voc,sentence)]
    lengths=torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch=torch.LongTensor(indexes_batch).transpose(0,1)
    input_batch=input_batch.to(device)

    tokens,scores=searcher(input_batch,lengths,max_length)
    decoded_words=[voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder,decoder,searcher,voc):
    input_sentence=''
    while(1):
        try:
            input_sentence=input('> ')
            if input_sentence=='q' or input_sentence=='quit':break
            input_sentence=normalizeStrings(input_sentence)
            output_words=evaluate(encoder=encoder,decoder=decoder,searcher=searcher,voc=voc,sentence=input_sentence)
            output_words[:]=[x for x in output_words if not (x=='EOS' or x=='PAD')]
            print('Bot:',' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")
