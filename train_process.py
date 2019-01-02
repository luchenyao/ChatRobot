# coding=utf-8
import torch
import torch.nn as nn
# from hyperparams import device,SOS_token,teacher_forcing_ratio,MAX_LENGTH,hidden_size
from hyperparams import *
import random
from word2vec import batch2TrainData
import os


def maskNLLLoss(inp,target,mask):
    nTotal=mask.sum()
    crossEntropy=-torch.log(torch.gather(inp,1,target.view(-1,1)).squeeze(1))
    loss=crossEntropy.masked_select(mask).mean()
    loss=loss.to(device)
    return loss,nTotal.item()


def train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip,max_length=MAX_LENGTH):
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_variable=input_variable.to(device)
    lengths=lengths.to(device)
    target_variable=target_variable.to(device)
    mask=mask.to(device)

    loss=0
    print_losses=[]
    n_totals=0

    encoder_outputs,encoder_hidden=encoder(input_variable,lengths)
    decoder_input=torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input=decoder_input.to(device)

    decoder_hidden=encoder_hidden[:decoder.n_layers]
    use_teacher_forcing=True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            decoder_input=target_variable[t].view(1,-1)
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal

    else:
        for t in range(max_target_len):
            decoder_output,decoder_hidden=decoder(decoder_input,decoder_hidden,encoder_outputs)
            _,topi=decoder_output.topk(1)
            decoder_input=torch.LongTensor([[topi[i][0]] for i in range(batch_size)])
            decoder_input=decoder_input.to(device)
            mask_loss,nTotal=maskNLLLoss(decoder_output,target_variable[t],mask[t])
            loss+=mask_loss
            print_losses.append(mask_loss.item()*nTotal)
            n_totals+=nTotal

    loss.backward()
    _=torch.nn.utils.clip_grad_norm_(encoder.parameters(),clip)
    _=torch.nn.utils.clip_grad_norm_(decoder.parameters(),clip)

    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses)/n_totals


def trainIters(model_name,voc,pairs,encoder,decoder,encoder_optimizer,decoder_optimizer,embedding,encoder_n_layers,decoder_n_layers,save_dir,n_iteration,batch_size,print_every,save_every,clip,corpus_name,loadFilename):
    checkpoint = torch.load(loadFilename)
    training_batches=[batch2TrainData(voc,[random.choice(pairs) for _ in range(batch_size)]) for _ in range(n_iteration)]
    print("Initializing...")
    start_iteration=1
    print_loss=0
    if loadFilename:
        start_iteration=checkpoint['iteration']+1

    print("Training...")
    for iteration in range(start_iteration,n_iteration+1):
        training_batch=training_batches[iteration-1]
        input_variable,lengths,target_variable,mask,max_target_len=training_batch
        loss=train(input_variable,lengths,target_variable,mask,max_target_len,encoder,decoder,embedding,encoder_optimizer,decoder_optimizer,batch_size,clip)
        print_loss+=loss

        if iteration % print_every==0:
            print_loss_avg=print_loss/print_every
            print("Iteration:{};Percent complete:{:.1f}%; Average loss:{:.4f}".format(iteration,iteration/n_iteration*100,print_loss_avg))
            print_loss=0

        if iteration % save_every==0:
            directory=os.path.join(save_dir,model_name,corpus_name,'{}-{}_{}'.format(encoder_n_layers,decoder_n_layers,hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration':iteration,
                'en':encoder.state_dict(),
                'de':decoder.state_dict(),
                'en_opt':encoder_optimizer.state_dict(),
                'de_opt':decoder_optimizer.state_dict(),
                'loss':loss,
                'voc_dict':voc.__dict__,
                'embedding':embedding.state_dict()
            },os.path.join(directory,'{}_{}.tar'.format(iteration,'checkpoint')))

























