import torch
import argparse
from torch import optim
import torch.nn as nn
from tqdm import tqdm
from model import CNN2RNN
from train_loader import Get_DataLoader

parser = argparse.ArgumentParser(description='Training model')
parser.add_argument('--train_path', default='./data/train.csv', type=str, help='directory of training data')
parser.add_argument('--dev_path', default='./data/dev.csv', type=str, help='directory of dev data')
parser.add_argument('--epoch', default=50, type=int, help='epochs')
parser.add_argument('--embedding', default=128, type=int, help='embedding dimension')
parser.add_argument('--max_len', default=265, type=int, help='max length of smiles')
parser.add_argument('--num_layers', default=1, type=int, help='number of layers')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
args = parser.parse_args()

device = torch.device('cuda:0')

model = CNN2RNN(embedding_dim=args.embedding, max_len=args.max_len, num_layers=args.num_layers)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = nn.L1Loss()
#loss_fn.to(device)

train_loader, val_loader = Get_DataLoader(args)

best_loss = 100.

for epoch in tqdm(range(args.epoch)):
    avg_loss = 0
    val_loss = 0

    for idx, batch in enumerate(train_loader):
        img = batch['img'].to(device)
        seq = batch['seq'].to(device)
        label = batch['label'].to(device)

        model.train()
        optimizer.zero_grad()
        #with torch.cuda.amp.autocast():
        predict = model(img, seq)
        loss = loss_fn(predict, label)
        loss.backward()
        optimizer.step()
        
        avg_loss += loss.item()
    avg_loss /= (idx + 1)
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            img = batch['img'].to(device)
            seq = batch['seq'].to(device)
            label = batch['label'].to(device)

            predict = model(img, seq)
            loss = loss_fn(label, predict)

            val_loss += loss.item()
    val_loss /= (idx + 1)

    if best_loss > val_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'CNN2RNN')
    
    print(f'epoch: {epoch}, trian loss: {avg_loss}, val loss: {val_loss}')

print("Train Finished")
print(f'Best loss: {best_loss}')