import torch
import argparse
import pandas as pd
from tqdm import tqdm
from model import CNN2RNN
from train_loader import Get_DataLoader

parser = argparse.ArgumentParser(description='Test model')
parser.add_argument('--model', default='./CNN2RNN', type=str, help='pretrained model path')
parser.add_argument('--train_path', default='./data/train.csv', help='train csv path')
parser.add_argument('--test_path', default='./data/test.csv', help='test csv path')
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--embedding', default=128, type=int, help='embedding dimension')
parser.add_argument('--max_len', default=265, type=int, help='max length of smiles')
parser.add_argument('--num_layers', default=1, type=int, help='number of layers')
args = parser.parse_args()

device = torch.device("cuda:0")
model = CNN2RNN(embedding_dim=args.embedding, max_len=args.max_len, num_layers=args.num_layers)
model.load_state_dict(torch.load(args.model))
model.to(device)
model.eval()

train_loader = Get_DataLoader(args, type="Test")

submission = pd.read_csv('./data/sample_submission.csv')

predict = []
with torch.no_grad():
    for batch in tqdm(train_loader):
        imgs = batch['img'].to(device)
        seqs = batch['seq'].to(device)

        pred = model.forward(imgs, seqs)
        pred = pred.cpu().numpy()
        gap = pred[:,0] - pred[:,1]
        predict.extend(list(gap))

submission['ST1_GAP(eV)'] = predict
submission.to_csv('./data/cnn2rnn_baseline.csv', index=False)