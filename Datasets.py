import numpy as np
import torch
import random
from torch.utils.data import Dataset
import pandas as pd

PADConstant = 0

class AdaFilterDataset(Dataset):
    def __init__(self, args, event_seqs, time_seqs, max_len = 0, data_type = 'Train', separate_bins = None):
        self.args = args
        self.event_seqs, self.time_seqs, self.time_seq_codes = [], [], []
        self.separate_barrel_used_time_seqs = time_seqs
        self.separate_bins = separate_bins
        self.separate_barrel_size = self.args.barrel_size
        self.data_type = data_type
        if max_len != 0:
            self.max_len = max_len
        else:
            self.max_len = max([len(event_seq) for event_seq in event_seqs])
        args.max_seq_length = self.max_len
        if self.args.data_type == "session_based":
            for i in range(len(event_seqs)):
                assert len(event_seqs[i]) == len(time_seqs[i])
                input_seq = event_seqs[i][-self.max_len:]
                time_seq = time_seqs[i][-self.max_len:]
                time_seq_codes = self.separate_barrel(time_seq).tolist()
                for j in range(len(event_seqs[i])):
                    self.event_seqs.append(input_seq[:j+1])
                    self.time_seqs.append(time_seq[:j+1])
                    self.time_seq_codes.append(time_seq_codes[:j+1])

        elif self.args.data_type == "sequential_based":
            if data_type=='train':
                for i in range(len(event_seqs)):
                    input_seq = event_seqs[i][-(self.max_len + 2):-2] 
                    time_seq = time_seqs[i][-(self.max_len + 2):-2]
                    time_seq_codes = self.separate_barrel(time_seq).tolist()
                    if len(event_seqs[i]) < 3: #filtering out the seqeuences that are too short
                       continue
                    
                    #for j in range(len(event_seqs[i])):
                    #    self.event_seqs.append(input_seq[:j+1])
                    #    self.time_seqs.append(time_seq[:j+1])
                    #    self.time_seq_codes.append(time_seq_codes[:j+1])
                    
                    self.event_seqs.append(input_seq)
                    self.time_seqs.append(time_seq)
                    self.time_seq_codes.append(time_seq_codes)
            elif data_type=='valid':
                for i in range(len(event_seqs)):
                    input_seq = event_seqs[i][-(self.max_len + 1):-1] 
                    time_seq = time_seqs[i][-(self.max_len + 1):-1]
                    if len(event_seqs[i]) < 3:
                       continue
                    time_seq_codes = self.separate_barrel(time_seq).tolist()
                    self.event_seqs.append(input_seq)
                    self.time_seqs.append(time_seq)
                    self.time_seq_codes.append(time_seq_codes)
            else:
                for i in range(len(event_seqs)):
                    input_seq = event_seqs[i][-(self.max_len):] 
                    time_seq = time_seqs[i][-(self.max_len):] 
                    if len(event_seqs[i]) < 3:
                       continue
                    time_seq_codes = self.separate_barrel(time_seq).tolist()
                    self.event_seqs.append(input_seq)
                    self.time_seqs.append(time_seq)
                    self.time_seq_codes.append(time_seq_codes)
    def __len__(self):
        return len(self.event_seqs)
    
    def get_type_size(self):
        type_set = set()
        for event_seq in self.event_seqs:
            types = [int(item) for item in event_seq]
            type_set = type_set | set(types)
        max_type = max(type_set)
        return max_type + 1
    
    def __getitem__(self, index):
        events, times, time_codes = self.event_seqs[index], self.time_seqs[index], self.time_seq_codes[index]
        input_events, input_times, input_time_codes = events[:-1], times[:-1], time_codes[:-1]
        next_event, next_time, next_time_code = events[-1], times[-1], time_codes[-1]
        if self.data_type == 'train':
            neg_event_sample, neg_time_sample = self.neg_sample(next_event, next_time_code, sample_num = 1)
        else:
            neg_event_sample, neg_time_sample = self.neg_sample(next_event, next_time_code, sample_num = 199)
        if len(input_events) < self.max_len:
            pad_len = self.max_len - len(input_events)
            input_events = [PADConstant] * pad_len + input_events
            input_times = [PADConstant] * pad_len + input_times
            input_time_codes = [PADConstant] * pad_len + input_time_codes
        else:
            input_events, input_times, input_time_codes = input_events[-self.max_len:], input_times[-self.max_len:], input_time_codes[-self.max_len:]
            
        assert len(input_events) == self.max_len

        cur_tensors = (
            torch.tensor(input_events, dtype=torch.long),
            torch.tensor(input_times, dtype=torch.float),
            torch.tensor(input_time_codes, dtype=torch.long),
            torch.tensor(next_event, dtype=torch.long),
            torch.tensor(next_time, dtype=torch.float),
            torch.tensor(next_time_code, dtype=torch.long),
            torch.tensor(neg_event_sample, dtype=torch.long),
            torch.tensor(neg_time_sample, dtype=torch.long)
            )
        return cur_tensors
    
    def separate_barrel(self, time_seqs = None, barrel_num = 100):
        time_data = []
        if self.separate_bins is None:
            for seq in self.separate_barrel_used_time_seqs:
                time_data += seq
            time_data = np.array(time_data)
            _, self.separate_bins = pd.qcut(time_data, q=self.separate_barrel_size, retbins=True, duplicates='drop') 
            self.separate_bins[0] = self.separate_bins[0]  - 0.1                                                                                  
        cut = pd.cut(time_seqs, self.separate_bins)
        return cut.codes
    
    def neg_sample(self, event_seqs, time_seqs, sample_num = 1):
        neg_event = []
        neg_time = []
        if sample_num == 1:
            for i in range(sample_num):
                event_sample = random.randint(1, self.args.type_size)
                time_sample = random.randint(0, self.args.barrel_size-1)
                #type_seqs = event_seqs * self.args.barrel_size + time_seqs
                while event_sample == event_seqs and time_seqs == time_sample:
                    event_sample = random.randint(1, self.args.type_size)
                    time_sample = random.randint(0, self.args.barrel_size-1)
                neg_event.append(event_sample)
                neg_time.append(time_sample)
        else:
            total_sample = [i + self.args.barrel_size for i in range(self.args.barrel_size * self.args.type_size) if i + self.args.barrel_size != event_seqs * self.args.barrel_size + time_seqs]
            neg_event.append([i // self.args.barrel_size for i in total_sample])
            neg_time.append([i % self.args.barrel_size for i in total_sample])
        return neg_event, neg_time
    