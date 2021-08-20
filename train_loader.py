import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.utils import shuffle

class SMILES_Tokenizer():
    def __init__(self, max_length):
        self.txt2idx = {}
        self.idx2txt = {}
        self.max_length = max_length
    
    def fit(self, SMILES_list):
        unique_char = set()
        for smiles in SMILES_list:
            for char in smiles:
                unique_char.add(char)
        
        for i, char in enumerate(unique_char):
            self.txt2idx[char] = i + 2
            self.idx2txt[i+2] = char
    
    def txt2seq(self, texts):
        seqs = []
        for text in texts:
            seq = [0] * self.max_length
            for i, t in enumerate(text):
                if i == self.max_length:
                    break
                try:
                    seq[i] = self.txt2idx[t]
                except:
                    seq[i] = 1
            seqs.append(seq)
        return np.array(seqs)

class CustomDataset(Dataset):
    def __init__(self, imgs, seqs, labels=None, mode='train'):
        self.mode = mode
        self.imgs = imgs
        self.seqs = seqs
        if self.mode == 'train':
            self.labels = labels
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs[index]).astpye(np.float32)/255
        img = np.transpose(img, (2,0,1))
        if self.mode == 'train':
            return {
                'img': torch.tensor(img, dtpye=torch.float32),
                'seq': torch.tensor(self.seqs[index], dtype=torch.long),
                'label': torch.tensor(self.labels[index], dtpye=torch.float32)
            }
        else:
            return{
                'img': torch.tensor(img, dtype=torch.float32),
                'seq': torch.tensor(self.seqs[index], dtype=torch.long)
            }

def Get_DataLoader(args, type='Train'):
    if type == 'Train':
        data = pd.read_csv(args.train_path)
        sub_data = pd.read_csv(args.dev_path)
        train = pd.concat([data, sub_data])
    elif type == 'Test':
        data = pd.read_csv(args.test_path)
        return False
    
    max_len = train.SMILES.str.len().max()
    tokenizer = SMILES_Tokenizer(max_len)
    tokenizer.fit(train.SMILES)

    seqs = tokenizer.txt2seq(train.SMILES)
    imgs = ('./data/train_imgs/'+data.uid+'.png').to_numpy()
    labels = train[['S1_energy(eV)', 'T1_energy(eV)']].to_numpy()

    data_len = len(imgs)
    cut_off = int(data_len * 0.8)

    if type == 'Train':
        imgs, seqs, labels = shuffle(imgs, seqs, labels, random_state=2021)
        train_imgs, train_seqs, train_labels = imgs[:cut_off], seqs[:cut_off], labels[:cut_off]
        val_imgs, val_seqs, val_labels = imgs[cut_off:], seqs[cut_off:], labels[cut_off:]
        train_dataset, val_dataset = CustomDataset(train_imgs, train_seqs, train_labels), CustomDataset(val_imgs, val_seqs, val_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
        return train_dataloader, val_dataloader

    else:
        dset = CustomDataset(imgs, seqs, labels)
        dataloader = DataLoader(dset, batch_size=args.batch_size, shuffle=False)
        return dataloader