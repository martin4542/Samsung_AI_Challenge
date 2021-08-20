import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

parser = argparse.ArgumentParser(description='prepare data for training')
parser.add_argument('--train_path', default='./data/train.csv', type=str, help='directory of training data')
parser.add_argument('--dev_path', default='./data/dev.csv', type=str, help='directory of dev data')
parser.add_argument('--test_path', default='./data/test.csv', type=str, help='directory of test data')
args = parser.parse_args()


train = pd.read_csv(args.train_path)
dev = pd.read_csv(args.dev_path)


for idx, row in tqdm(train.iterrows()):
    file = row['uid']
    smiles = row['SMILES']
    m = Chem.MolFromSmiles(smiles)
    if m != None:
        img = Draw.MolToImage(m, size=(300,300))
        img.save(f'data/train_img/{file}.png')