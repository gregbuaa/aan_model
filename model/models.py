import torch
import torch.nn as nn
from torch.autograd import Function 
from torch.nn import functional as F


### simple MLP model.
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.5):
        
        super().__init__()
                    
        self.fc=nn.Sequential(
            nn.Linear(input_dim, input_dim //2),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 2, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latents):
        latents = self.dropout(latents)
        #latent = [batch size, latent len]        
        return self.fc(latents)

### TextCNN model.
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, embedding_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
                
        #text = [batch size, sent len]
        
        embedded = self.embedding(text)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


### MLP-based AAN model for Amazon-Review Dataset.
class AANMLP(nn.Module):
    def __init__(self,input_dim, latent_dim, output_dim, dropout = 0.5, version="AAN"):
        super().__init__()
        
        self.extractor = MLP(input_dim, latent_dim,dropout)
        self.predictor = MLP(latent_dim, output_dim,dropout=0.0)
        self.version = version
        if self.version == 'AAN-A':
            self.mmd_linear = nn.Linear(latent_dim,latent_dim)
            self.cmmd_linear = nn.Linear(latent_dim,latent_dim)


    def predict(self,target):
        return self.predictor(self.extractor(target))

    def forward(self,text):
        latent = self.extractor(text) 
        task_prediction = self.predictor(latent)
        if self.version == 'AAN':
            return latent, task_prediction
        ### version == 'AAN-A'
        else:
            mlatent = self.mmd_linear(latent)
            clatent = self.cmmd_linear(latent)
            return  latent, task_prediction, mlatent, clatent


### TextCNN-based AAN for Amazon-Text.
class AANTextCNN(nn.Module):
    def __init__(self,input_dim, embedding_dim, n_filters,filter_size,latent_dim, output_dim, pad_idx, dropout = 0.5,version='AAN'):
        super().__init__()
        self.version = version
        self.extractor = TextCNN(input_dim, embedding_dim, n_filters,filter_size,latent_dim,dropout,pad_idx)
        self.predictor = MLP(latent_dim,output_dim,dropout=0.0)
        if self.version == 'AAN-A':
            self.mmd_linear = nn.Linear(latent_dim,latent_dim)
            self.cmmd_linear = nn.Linear(latent_dim,latent_dim)

    def predict(self,target):
        return self.predictor(self.extractor(target))

    def forward(self,text):
        latent = self.extractor(text) 
        task_prediction = self.predictor(latent)
        if self.version == 'AAN':
            return latent, task_prediction
        ### version == 'AAN-A'
        else:
            mlatent = self.mmd_linear(latent)
            clatent = self.cmmd_linear(latent)
            return  latent, task_prediction, mlatent, clatent

### BertGRU-based AAN for Amazon-Text.
class AANBertGRU(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, n_layers, bidirectional, dropout,version='AAN'):
        
        super().__init__()
        self.version = version
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
        self.extractor = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, hidden_dim)
        self.predictor = MLP(hidden_dim, output_dim,dropout=0.0)
        if self.version == 'AAN-A':
            self.mmd_linear = nn.Linear(hidden_dim,hidden_dim)
            self.cmmd_linear = nn.Linear(hidden_dim,hidden_dim)


        self.dropout = nn.Dropout(dropout)

    
    def predict(self,target):
        # return self.predictor(self.extractor(target))
        with torch.no_grad():
            embedded, pool_embedded = self.bert(target)
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2,:,:].clone(), hidden[-1,:,:].clone()), dim = 1)
        else:
            hidden = hidden[-1,:,:].clone()
                
        #hidden = [batch size, hid dim]
        
        return self.predictor(self.extractor(hidden))

        
    def forward(self, text):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded, _ = self.bert(text)
                
        # embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:].clone(), hidden[-1,:,:].clone()), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:].clone())
                
        #hidden = [batch size, hid dim]
        
        latent = self.extractor(hidden)
        
        task_prediction = self.predictor(latent)


        if self.version == 'AAN':
            return latent, task_prediction
        ### version == 'AAN-A'
        else:
            mlatent = self.mmd_linear(latent)
            clatent = self.cmmd_linear(latent)
            return  latent, task_prediction, mlatent, clatent

