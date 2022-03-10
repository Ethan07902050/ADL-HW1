import json
import pickle
import pandas as pd
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from intentDataset import intentDataset
from lstm import LSTM
from utils import Vocab

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = intentDataset(data, vocab, intent2idx, args.max_len, train=0)

    # create dataLoader for test dataset
    test_generator = DataLoader(dataset, batch_size=args.batch_size, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")    
    input_size = 300
    output_size = len(intent2idx)
    model = LSTM(input_size, output_size, embeddings, args, 'intent')
    
    # load weights into model
    ckpt = torch.load(args.ckpt_path / 'intent.pt')
    model.load_state_dict(ckpt, strict=False)
    model.to(args.device)
    model.eval()

    # predict dataset
    intents = []
    ids = []
    for batch, id, seq_len in test_generator:
        batch = batch.to(args.device)
        output = model(batch, seq_len, args.device)
        pred = output.max(1, keepdim=True)[1]
        pred = pred.squeeze(1)
        intents += pred
        ids += id
    
    # write prediction to file (args.pred_file)
    intents = [dataset.idx2label(idx.item()) for idx in intents]
    df = pd.DataFrame({'id': ids, 'intent': intents})
    df.to_csv(args.pred_file, index=False)

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./model",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
