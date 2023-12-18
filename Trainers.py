import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import random
import pandas as pd


from torch.optim import Adam
from Utils import get_metric

class Trainer:
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
        self.model = model
        if self.cuda_condition:
            self.model.cuda()
        
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        beta = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr = self.args.lr, betas=beta, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)
    
    def valid(self, epoch):
        return self.iteration(epoch, self.eval_dataloader, train=False)

    def test(self, epoch):
        return self.iteration(epoch, self.test_dataloader, train=False)
    
    def iteration(self, epoch, dataloader, train = True):
        raise NotImplementedError
    
    def get_scores(self, epoch, pred_list):
        if self.args.benchmark_type == "hybrid_retrieval":
            pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        print(original_state_dict.keys())
        new_dict = torch.load(file_name)
        print(new_dict.keys())
        for key in new_dict:
            original_state_dict[key] = new_dict[key]
        self.model.load_state_dict(original_state_dict)

    def cross_entropy(self, seq_emb, pos_event_ids, pos_time_ids, neg_event_ids, neg_time_ids):
        # [batch seq_len hidden_size]
        pos_event_emb = self.model.event_embeddings(pos_event_ids)
        pos_time_emb = self.model.time_embeddings(pos_time_ids)

        neg_event_emb = self.model.event_embeddings(neg_event_ids)
        neg_time_emb = self.model.time_embeddings(neg_time_ids)

        pos_emb = pos_event_emb + pos_time_emb
        neg_emb = neg_event_emb + neg_time_emb
        

        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch*seq_len]
        neg_logits = torch.sum(neg_emb * seq_emb, -1)

        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 -  torch.sigmoid(neg_logits) + 1e-24)
        )

        return loss

    def predict(self, seq_out, neg_event_sample, neg_time_sample):
        test_event_emb = self.model.event_embeddings(neg_event_sample)
        test_time_emb = self.model.time_embeddings(neg_time_sample)
        test_logits = torch.bmm((test_event_emb + test_time_emb), seq_out.unsqueeze(-1)).squeeze(-1)
        return test_logits

class AdaFilterTrainer(Trainer):
    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args):
        super().__init__(model, train_dataloader, eval_dataloader, test_dataloader, args)
    

    def iteration(self, epoch, dataloader, train=True):
        str_code = "train" if train else "test"
        data_iter = tqdm.tqdm(enumerate(dataloader), 
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(dataloader),
                              bar_format="{l_bar}{r_bar}") 
        if train:
            self.model.train()
            total_loss = 0.0
            for i, batch in data_iter:
                self.optim.zero_grad()
                batch = tuple(t.to(self.device) for t in batch)
                input_events, input_times, input_time_codes, next_event, next_time, next_time_codes, neg_event_sample, neg_time_sample = batch
                if self.args.benchmark_type == "hybrid_retrieval":
                    seq_emb = self.model(input_events, input_times, input_time_codes)
                    loss = self.cross_entropy(seq_emb, next_event, next_time_codes, neg_event_sample, neg_time_sample)
                elif self.args.benchmark_type == "linear_prediction":
                    seqs_length = (input_times.new_ones(input_times.shape[0]) * len(input_times[0])+1).long()
                    seq_emb = self.model(input_events, input_times)
                    time_pred = self.model.time_predictor(seq_emb)
                    type_pred = self.model.type_predictor(seq_emb)
                    type_pred = torch.cat([type_pred.new_zeros(type_pred.shape[0], 1),type_pred], dim=-1)
                    type_loss = self.model.type_loss(type_pred, next_event)
                    time_loss = self.model.time_loss(time_pred, next_time) / 20000 # scale time loss
                    #log_likelihood = self.model.log_likelihood(input_times, next_event, next_time) #optional

                    loss = torch.mean(type_loss + time_loss)# - log_likelihood)
                else:
                    raise NotImplementedError

                loss.backward()
                self.optim.step()

                total_loss += loss.item()
            post_fix = {
                "epoch": epoch,
                "total_loss": '{:.4f}'.format(total_loss / len(data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')
        else:
            self.model.eval()
            pred_list = None
            for i, batch in data_iter:
                batch = tuple(t.to(self.device) for t in batch)
                input_events, input_times, input_time_codes, next_event, next_time, next_time_codes, neg_event_sample, neg_time_sample = batch
                if self.args.benchmark_type == "hybrid_retrieval":
                    seq_emb = self.model(input_events, input_times, input_time_codes)
                    test_neg_events = torch.cat((next_event.unsqueeze(-1), neg_event_sample.squeeze(1)), -1)
                    test_neg_time = torch.cat((next_time_codes.unsqueeze(-1), neg_time_sample.squeeze(1)), -1)
                       
                    test_logits = self.predict(seq_emb, test_neg_events, test_neg_time)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                elif self.args.benchmark_type == "linear_prediction":
                    seq_emb = self.model(input_events, input_times)
                    time_pred = self.model.time_predictor(seq_emb)
                    type_pred = self.model.type_predictor(seq_emb)
                    type_pred = torch.argmax(type_pred, dim=-1)
                    time_pred = time_pred.cpu().detach().numpy().copy()
                    type_pred = type_pred.cpu().detach().numpy().copy()
                    time_pred = time_pred.reshape(-1)
                    time_pred = pd.cut(time_pred, self.args.separate_bins).codes
                    type_pred = type_pred.reshape(-1)
                    test_logits = (type_pred + 1) * self.args.barrel_size +  time_pred
                    next_code = (next_event* self.args.barrel_size + next_time_codes).cpu().detach().numpy().copy()
                    rank = abs(next_code - test_logits) 
                    test_logits = rank
        
                if i == 0:
                    pred_list = test_logits
                else:
                    pred_list = np.append(pred_list, test_logits, axis=0)

            return self.get_scores(epoch, pred_list)