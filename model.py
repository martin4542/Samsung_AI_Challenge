import torch
import torch.nn as nn
from torch.nn.modules import dropout
from torchvision import models
from torch.utils.data import Dataset, DataLoader

class CNN_Encoder(nn.Module):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        model = models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.feature_extract_model = nn.Sequential(*modules)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, embedding_dim),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.feature_extract_model(x)
        x = x.permute(0,2,3,1)
        x = x.view(x.size(0), -1, x.size(3))
        x = self.classifier(x)
        return x

class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_layers):
        super(RNN_Decoder, self).__init__()
        self.embedding = nn.Embedding(max_len, embedding_dim)
        self.dropout = nn.Dropout(0.4)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim, num_layers)
        self.final_layer = nn.Linear((max_len+100) * embedding_dim, 2)
    
    def forward(self, enc_out, dec_inp):
        embedded = self.embedding(dec_inp)
        embedded = self.dropout(embedded)
        embedded = torch.cat([enc_out, embedded], dim=1)
        hidden, _ = self.lstm(embedded)
        hidden = hidden.view(hidden.size(0), -1)
        output = nn.ReLU()(self.final_layer(hidden))
        return output

class CNN2RNN(nn.Module):
    def __init__(self, embedding_dim, max_len, num_layers):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_layers)
    
    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        return output