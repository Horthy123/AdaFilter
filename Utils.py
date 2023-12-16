import os
import random
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from Datasets import AdaFilterDataset


PADConstant = -1
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def get_local_time():
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur

class EarlyStopping:
    def __init__(self, checkpoint_path, patience=10, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def get_event_seqs_and_max_item(data_file):
    lines = open(data_file).readlines()
    event_seq = []
    item_set = set()
    for line in lines:
        event, items = line.strip().split(' ', 1)
        items = items.split()
        items = [int(item) + 1 for item in items]
        event_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    return event_seq, max_item

def get_time_seqs(time_file):
    lines = open(time_file).readlines()
    time_seqs = []
    for line in lines:
        event, items = line.strip().split(' ', 1)
        items = items.split()
        items = [float(item) for item in items]
        time_seqs.append(items)
    return time_seqs

def get_event_seqs(data_file):
    lines = open(data_file).readlines()
    event_seq = []
    item_set = set()
    for line in lines:
        event, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) + 1 for item in items]
        event_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)
    num_events = len(lines)
    return event_seq, num_events

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def get_seq_dic(args):


    args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_times_file = args.data_dir + args.data_name + '_times.txt'

    if args.data_type == "session_based":
        event_seq, max_item = get_event_seqs_and_max_item(args.data_file)
        args.type_size = max_item + 1
        event_time_seq = get_time_seqs(args.data_times_file)
        event_seq_eval, num_events_eval = get_event_seqs(args.data_file_eval)
        event_time_seq_eval = get_time_seqs(args.data_times_file_eval)
        event_seq_test, num_events_test = get_event_seqs(args.data_file_test)
        event_time_seq_test = get_time_seqs(args.data_times_file_test)
        seq_dic = {'event_seq':event_seq, 
                   'event_seq_eval':event_seq_eval, 'num_events_eval':num_events_eval,
                   'event_seq_test':event_seq_test, 'num_events_test':num_events_test,
                   'event_time_seq':event_time_seq, 'event_time_seq_eval':event_time_seq_eval, 'event_time_seq_test':event_time_seq_test}
    elif args.data_type == "sequential_based":
        event_seq, max_item = get_event_seqs_and_max_item(args.data_file)
        event_seq = event_seq
        max_item = max([max_item])
        args.type_size = max_item + 1
        event_time_seq = get_time_seqs(args.data_times_file)
        seq_dic = {'event_seq':event_seq, 'event_time_seq':event_time_seq}

    max_item = (args.type_size)  * args.barrel_size - 1
    return seq_dic, max_item

def get_dataloder(args, seq_dic):
    if args.data_type == "session_based":
        
        train_dataset = AdaFilterDataset(args, seq_dic['event_seq'], seq_dic['event_time_seq'], data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        args.separate_bins = train_dataset.separate_bins
        eval_dataset = AdaFilterDataset(args, seq_dic['event_seq_eval'], seq_dic['event_time_seq_eval'], data_type='valid', separate_bins=train_dataset.separate_bins)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = AdaFilterDataset(args, seq_dic['event_seq_test'], seq_dic['event_time_seq_test'], data_type='test', separate_bins=train_dataset.separate_bins)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    
    elif args.data_type == "sequential_based":
        train_dataset = AdaFilterDataset(args, seq_dic['event_seq'], seq_dic['event_time_seq'], data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)
        
        args.separate_bins = train_dataset.separate_bins
        eval_dataset = AdaFilterDataset(args, seq_dic['event_seq'], seq_dic['event_time_seq'], data_type='valid', separate_bins=train_dataset.separate_bins)
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

        test_dataset = AdaFilterDataset(args, seq_dic['event_seq'], seq_dic['event_time_seq'], data_type='test', separate_bins=train_dataset.separate_bins)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
    return train_dataloader, eval_dataloader, test_dataloader
