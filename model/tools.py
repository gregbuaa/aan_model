import torch

import time
from torch.nn import functional as F


### show the number of parameters in model in the framework of pytorch
def count_parameters(model):
    # print(f'The model has {count_parameters(model):,} trainable parameters')
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


### compute binary accuracy under the framework of pytorch.
def binary_accuracy(preds, y):
    """
    Returns the number of labels agreement with the truth-grouth labels per batch, i.e. if you get 8 right, and the batch size is 10,  this returns 8 and 10
    """
    #round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    return correct.sum(), len(correct)


def categorical_accuracy(preds, y):
    """
    Returns the number of labels agreement with the truth-grouth labels per batch, i.e. if you get 8 right, and the batch size is 10,  this returns 8 and 10
    """
    max_preds = preds.argmax(dim = 1, keepdim = True) # get the index of the max probability
    correct = max_preds.squeeze(1).eq(y)
    return correct.sum(), y.shape[0]


## compute the time of a epoch.
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


## ANN model train.
def train_adverisal(model, source_iterator, target_iterator, optim_task,optim_kernel, criterion, mmd_loss, cmmd_loss):
    epoch_loss = 0
    

    model.train()
    for sbatch in source_iterator:
        for tbatch in target_iterator:
            # maximzing the loss.
            slabel = sbatch.label.unsqueeze(1)
           
            slatent, spred, smlatent, sclatent = model(sbatch.text)
            tlatent, tpred, tmlatent, tclatent =  model(tbatch.text)
            device = slatent.device
            OUTPUT_DIM = spred.size(1)
            one_hot_slabel = torch.zeros(sbatch.text.size(0),OUTPUT_DIM).to(device).scatter_(1, slabel.long(), 1)
            tpred = F.softmax(tpred,dim = 1)
            mloss =  mmd_loss(smlatent,tmlatent)
            closs = cmmd_loss(sclatent,tclatent,one_hot_slabel,tpred)
            task_loss = criterion(spred, sbatch.label.long()) 
            optim_task.zero_grad()
           
            loss = task_loss + closs + mloss
            loss.backward(retain_graph=True)
            optim_task.step()
              
            optim_kernel.zero_grad()
            loss2 =  - mloss-closs
            loss2.backward()
            optim_kernel.step()
            epoch_loss += loss.item()
            break

    return epoch_loss / (len(target_iterator))

def train_normal(model, source_iterator, target_iterator, optim_task, criterion, mmd_loss, cmmd_loss, mu):
    epoch_loss = 0

    model.train()
    for sbatch in source_iterator:
        for tbatch in target_iterator:
            # maximzing the loss.
            slabel = sbatch.label.unsqueeze(1)
           
            slatent, spred = model(sbatch.text)
            tlatent, tpred =  model(tbatch.text)
            device = slatent.device
            OUTPUT_DIM = spred.size(1)
            one_hot_slabel = torch.zeros(sbatch.text.size(0),OUTPUT_DIM).to(device).scatter_(1, slabel.long(), 1)
            tpred = F.softmax(tpred,dim = 1)
            mloss =  mmd_loss(slatent,tlatent)
            closs = cmmd_loss(slatent,tlatent,one_hot_slabel,tpred)
            task_loss = criterion(spred, sbatch.label.long()) 
  
            
            optim_task.zero_grad()
           
            loss = task_loss + closs +  mu * mloss
            loss.backward()
            optim_task.step()
              

            epoch_loss += loss.item()

            break


    return epoch_loss / (len(target_iterator))

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    sample_num = 0

    model.eval()
    with torch.no_grad():
        for batch in iterator:
            task_pred = model.predict(batch.text)
            task_pred = F.softmax(task_pred,dim=1)
            acc, num = categorical_accuracy(task_pred, batch.label.long())
            epoch_loss += criterion(task_pred,batch.label.long()) * num
            epoch_acc += acc.item()
            sample_num += num
    return epoch_acc / sample_num, epoch_loss / sample_num


