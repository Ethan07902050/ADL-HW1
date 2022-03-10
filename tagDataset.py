from typing import List, Dict
from utils import Vocab
import numpy as np

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch

class tagDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
        train: int
    ):        
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: tag for tag, idx in self.label_mapping.items()}
        self.max_len = max_len
        self.train = train
        
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        num_classes = len(self.label_mapping)
        
        tokens = instance['tokens']
        words = [self.vocab.token2idx[word]
                 if word in self.vocab.token2idx else 1
                 for word in tokens]
        words = torch.tensor(words)
        id = instance['id']

        if self.train:
            tags = instance['tags']
            tags = torch.tensor([self.label_mapping[tag] for tag in tags])
            return (words, tags)

        return words, id
    
    def collate_fn(self, batch):

        sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)     
        seq = [x[0] for x in sorted_batch]
        padded_seq = pad_sequence(seq, batch_first=True)
        seq_len = torch.tensor([x.shape[0] for x in seq])
        
        if self.train:
            tags = [x[1] for x in sorted_batch]
            padded_tag = pad_sequence(tags, batch_first=True)
            return padded_seq, padded_tag, seq_len
        else:
            id = [x[1] for x in sorted_batch]            
            return padded_seq, id, seq_len

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
