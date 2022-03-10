import json
import pickle
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
from tqdm import trange, tqdm
from utils import Vocab
from tagDataset import tagDataset
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
        for batch, labels, seq_lens in data_loader:
            batch, labels = batch.to(device), labels.to(device)        
            output = model(batch, seq_lens, args.device)

            for out, label, seq_len in zip(output, labels, seq_lens):
                out, label = out[:seq_len], label[:seq_len]
                pred = out.max(1)[1]   
                correct += (torch.all(pred.eq(label))).int()
                total += 1
        
        return correct / total


def main(args):
    
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)
        
    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text())
            for split, path in data_paths.items()}
    
    datasets: Dict[str, tagDataset] = {
        split: tagDataset(split_data, vocab, tag2idx, args.max_len, train=1)
        for split, split_data in data.items()
    }
    
    # create DataLoader for train / dev datasets
    train_generator = DataLoader(datasets[TRAIN], batch_size=args.batch_size, collate_fn=datasets[TRAIN].collate_fn)
    dev_generator = DataLoader(datasets[DEV], batch_size=args.batch_size, collate_fn=datasets[DEV].collate_fn)
        
    # init model and move model to target device(cpu / gpu)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")    
    input_size = int(embeddings.shape[1])
    output_size = len(tag2idx)
    model = LSTM(input_size, output_size, embeddings, args, 'slot')
    model.to(device)

    # init optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)    
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # Training loop - iterate over train dataloader
        # and update model weights

        train_loss, dev_loss = [], []
        
        for batch, label, seq_lens in train_generator:
            optimizer.zero_grad()                        
            batch, label = batch.to(device), label.to(device)
            output = model(batch, seq_lens, args.device)

            loss = 0
            for pred, target, seq_len in zip(output, label, seq_lens):
                loss += criterion(pred[:seq_len], target[:seq_len])
            loss /= batch.shape[0]            
            train_loss.append(loss.item())
            loss.backward()  
            optimizer.step()

        for batch, label, seq_lens in dev_generator:
            batch, label = batch.to(device), label.to(device)
            output = model(batch, seq_lens, args.device)

            loss = 0
            for pred, target, seq_len in zip(output, label, seq_lens):
                loss += criterion(pred[:seq_len], target[:seq_len])
            loss /= batch.shape[0]
            
            dev_loss.append(loss.item())
            
        # Evaluation loop - calculate accuracy and save model weights
        train_acc = get_accuracy(model, train_generator, args.device)
        dev_acc = get_accuracy(model, dev_generator, args.device)
        train_loss = np.mean(train_loss)
        dev_loss = np.mean(dev_loss)                

        print(f'train loss: {train_loss}, dev loss: {dev_loss}')
        print(f'train acc: {train_acc}, dev acc: {dev_acc}')


    torch.save(model.state_dict(), args.ckpt_dir / 'slot.pt')

    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
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
       "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cuda:1"
    )
    parser.add_argument("--num_epoch", type=int, default=50)

    args = parser.parse_args()    
    return args


if __name__ == "__main__":
    args = parse_args()    
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
