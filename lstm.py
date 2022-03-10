from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class LSTM(nn.Module):
    def __init__(self, input_size, num_classes, embedding, args, task):
        super(LSTM, self).__init__()
        self.task = task
        self.emb = nn.Embedding.from_pretrained(embedding)
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers
        self.bidirectional = args.bidirectional        
        
        self.lstm = nn.LSTM(input_size, self.hidden_size,
                            num_layers=args.num_layers,
                            bidirectional=args.bidirectional,
                            dropout=args.dropout,
                            batch_first=True)
            
        if task == 'slot' and self.bidirectional:
            self.fc1 = nn.Linear(self.hidden_size*2, 64)
        else:
            self.fc1 = nn.Linear(self.hidden_size, 64)
            
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x, seq_lens, device):
        x = self.emb(x)

        if self.bidirectional:
            h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        else:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
            
        h0, c0 = h0.to(device), c0.to(device)
        
        x = pack_padded_sequence(x, seq_lens, batch_first=True)
        output, (hn, cn) = self.lstm(x, (h0, c0))

        if self.task == 'intent':
            out = hn[-1]
        elif self.task == 'slot':                    
            out, lens = pad_packed_sequence(output, batch_first=True)
        
        out = self.relu(out)        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
