import json
import pickle
import os
import pandas as pd
import numpy as np
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import trange, tqdm
from utils import Vocab
from intentDataset import intentDataset
from lstm import LSTM

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def get_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct, total = 0, 0
        for batch, labels, seq_len in data_loader:
            batch, labels = batch.to(device), labels.to(device)        
            output = model(batch, seq_len, args.device)              
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.shape[0]
        
    return correct / total        


def main(args):
    
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
        
    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    
    datasets: Dict[str, intentDataset] = {
        split: intentDataset(split_data, vocab, intent2idx, args.max_len, train=1)
        for split, split_data in data.items()
    }
    
    # create DataLoader for train / dev datasets
    train_generator = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn)
    dev_generator = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn)
        
    # init model and move model to target device(cpu / gpu)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")    
    input_size = int(embeddings.shape[1])
    output_size = len(intent2idx)        
    model = LSTM(input_size, output_size, embeddings, args, 'intent')
    model.to(args.device)

    # init optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")

    for epoch in epoch_pbar:
        # Training loop - iterate over train dataloader and update model weights

        train_loss, dev_loss = [], []
        
        for batch, label, seq_len in train_generator:                  
            optimizer.zero_grad()                        
            batch, label = batch.to(args.device), label.to(args.device)
            output = model(batch, seq_len, args.device)
            loss = criterion(output, label)
            train_loss.append(loss.item())        
            loss.backward()
            optimizer.step()            

        for batch, label, seq_len in dev_generator:
            batch, label = batch.to(args.device), label.to(args.device)
            output = model(batch, seq_len, args.device)
            loss = criterion(output, label)
            dev_loss.append(loss.item())        
        
        # Evaluation loop - calculate accuracy and save model weights
        train_loss = np.mean(train_loss)
        dev_loss = np.mean(dev_loss)
        train_acc = get_accuracy(model, train_generator, args.device)
        dev_acc = get_accuracy(model, dev_generator, args.device)
        print(f'train loss: {train_loss}, dev loss: {dev_loss}')
        print(f'train acc: {train_acc}, dev acc: {dev_acc}')
             
    torch.save(model.state_dict(), args.ckpt_dir / 'intent.pt')

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
       "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()    
    return args


if __name__ == "__main__":
    args = parse_args()    
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
