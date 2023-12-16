import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import grad
from Modules import LayerNorm

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(0).type(torch.float)

class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        return out

class AdaFilterModel(nn.Module):
    def __init__(self, args):
        super(AdaFilterModel, self).__init__()
        self.args = args
        self.event_embeddings = nn.Embedding((args.type_size), args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.time_embeddings = nn.Embedding(args.barrel_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.omega = nn.Parameter(torch.randn(1, 1, args.hidden_size, 2) * np.pi / 2)
        if args.use_baseline is False:
            from Modules import Encoder
            self.event_encoder = Encoder(args)
        else:
            from Baselines import Encoder
            if args.model_name == "THP":
                self.linear = nn.Linear(args.hidden_size, args.type_size)
                self.alpha = nn.Parameter(torch.tensor(-0.1))
                self.beta = nn.Parameter(torch.tensor(1.0))
                self.time_predictor = Predictor(args.hidden_size, 1)
                self.type_predictor = Predictor(args.hidden_size, args.type_size)
            self.event_encoder = Encoder(args)
        #self.beta = nn.Parameter(torch.tensor(1.0))
        self.softmax = nn.Softmax(dim=-1)
        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def temporal_enc(self, event_seq, time, time_code):
        if self.args.temporal_encoding_method == "LTE":
            non_pad_mask = (event_seq != 0)
            event_embeddings = self.event_embeddings(event_seq)
            seq_length = event_seq.size(1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=event_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(event_seq)
            position_embedding = self.position_embeddings(position_ids)
            time_embedding = self.time_embeddings(time_code)    
            #time_emb = self.log_time_encode
            tem_enc = (position_embedding + event_embeddings + time_embedding) * non_pad_mask.int().unsqueeze(-1)
            tem_enc = self.LayerNorm(tem_enc)
            tem_enc = self.dropout(tem_enc)
        elif self.args.temporal_encoding_method == "TE-T":
            non_pad_mask = (event_seq != 0)
            self.position_vec  = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / self.args.hidden_size) for i in range(self.args.hidden_size)]).to(time.device)
            time = torch.cumsum(time, dim=1)
            result = time.unsqueeze(-1) / self.position_vec 
            result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
            result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
            event_embeddings = self.event_embeddings(event_seq)
            tem_enc = (result + event_embeddings) * non_pad_mask.int().unsqueeze(-1)            
            tem_enc = self.LayerNorm(tem_enc)
            tem_enc = self.dropout(tem_enc)            
        elif self.args.temporal_encoding_method == "TE-S":
            seq_length = event_seq.size(1)
            time = torch.cumsum(time, dim=1)
            position_ids = torch.arange(seq_length, dtype=torch.long, device=event_seq.device)
            position_ids = position_ids.unsqueeze(0).expand_as(event_seq).flip(1).unsqueeze(-1)
            self.omega = self.omega.to(time.device)
            result = position_ids * self.omega[:,:,:,0] + time.unsqueeze(-1) * self.omega[:,:,:,1]
            result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
            result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
            event_embeddings = self.event_embeddings(event_seq)
            tem_enc = (result + event_embeddings) * non_pad_mask.int().unsqueeze(-1)
            tem_enc = self.LayerNorm(tem_enc)
            tem_enc = self.dropout(tem_enc)  
        return tem_enc
    
    def forward(self, event_seq, time_seq, time_code_seq = None):
        sequence_emb = self.temporal_enc(event_seq, time_seq, time_code_seq)
        if self.args.use_baseline is True:
            attention_mask = (event_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            max_len = attention_mask.size(-1)
            attn_shape = (1, max_len, max_len)
            subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)
            subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
            subsequent_mask = subsequent_mask.long()

            if self.args.cuda_condition:
                subsequent_mask = subsequent_mask.cuda()
            extended_attention_mask = extended_attention_mask * subsequent_mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
            extended_attention_mask = (1 - extended_attention_mask) * -10000.0
            
            event_encoded_layers = self.event_encoder(sequence_emb, time_seq, extended_attention_mask, output_all_encoded_layers = True)
        else:
            event_encoded_layers = self.event_encoder(sequence_emb, time_seq, output_all_encoded_layers = True)
        
        seq_out = event_encoded_layers[-1]
        seq_out = seq_out[:, -1, :]
        self.h = seq_out
        return seq_out
    
    "only for THP, referred to https://github.com/SimiaoZuo/Transformer-Hawkes-Process.git"
    def log_likelihood(self, time_seq, next_event, next_event_time):
        all_hid = self.linear(self.h)
        all_lambda = softplus(all_hid, self.beta)
        type_lambda = torch.diag(torch.index_select(all_lambda, dim=1, index=next_event))
 
        event_ll = torch.log(type_lambda)
        
        num_samples = 100
        sample_time = next_event_time.unsqueeze(-1) * torch.rand([*next_event_time.size(), num_samples], device=next_event.device)
        time = torch.cumsum(time_seq, dim=1) 
        sample_time = sample_time / (time[:, -1].unsqueeze(-1) + 1)
        temp_hid = self.linear(self.h)
        sample_all_lambda = softplus(temp_hid.unsqueeze(-1) + self.alpha * sample_time.unsqueeze(1), self.beta)
        sample_all_lambda = torch.sum(sample_all_lambda, dim=1)
        sample_all_lambda = torch.sum(sample_all_lambda, dim=-1) / num_samples
        non_event_ll = next_event_time * sample_all_lambda
        
        loglikelihood = event_ll - non_event_ll
        return loglikelihood
    
    def type_loss(self, pred, event_types):
        assert pred.shape[0] == event_types.shape[0]
        targets = event_types
        type_loss = F.cross_entropy(torch.sigmoid(pred), targets, reduction='mean')
        pred_type = torch.max(pred, dim=-1)[1]
        correct_num = (pred_type == targets).sum()
        #print(correct_num)
        return type_loss

    def time_loss(self, pred, event_times):
        pred = pred.squeeze(dim=-1)
        diff = pred - event_times
        return torch.sqrt(torch.mean(diff * diff))


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss
  
def softplus(x, beta):
    temp = beta * x
    temp[temp > 20] = 20
    return 1.0 / beta * torch.log(1 + torch.exp(temp))

class CTLSTM(nn.Module):
    "code for NHP, referred to https://github.com/xiao03/nh.git"
    def __init__(self, args):
        super(CTLSTM, self).__init__()
        
        self.args = args
        self.hidden_size = args.hidden_size
        self.type_size = args.type_size
        self.batch_first = True
        self.num_layers = args.num_hidden_layers

        self.rec = nn.Linear(2*self.hidden_size, 7*self.hidden_size)
        self.w = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.args.hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.wa = nn.Linear(self.hidden_size, self.type_size)
        self.time_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.args.hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )
        self.type_predictor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Dropout(self.args.hidden_dropout_prob),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.type_size)
        )
        self.event_embeddings = nn.Embedding(self.type_size+1, self.hidden_size)
        self.time_embeddings = nn.Embedding(args.barrel_size, args.hidden_size)
        
    def init_states(self, batch_size, device):
        self.h_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=device)
        self.c_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=device)
        self.c_bar = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=device)
        self.c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float, device=device)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        (gate_i,
        gate_f,
        gate_z,
        gate_o,
        gate_i_bar,
        gate_f_bar,
        gate_delta) = torch.chunk(self.rec(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
            torch.exp(-delta_t * duration_t.view(-1,1))

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t
    
    def forward(self, event_seqs, time_seqs, code_seqs = None, batch_first = True):
        if batch_first:
            event_seqs = event_seqs.transpose(0,1)
            duration_seqs = time_seqs.transpose(0,1)
        
        batch_size = event_seqs.size()[1]
        batch_length = event_seqs.size()[0]

        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []
        self.init_states(batch_size, event_seqs.device)
        c, self.c_bar, o_t, delta_t = self.recurrence(self.event_embeddings(event_seqs.new_zeros(batch_size)), self.h_d, self.c_d, self.c_bar)
        for t in range(batch_length):
            self.c_d, self.h_d = self.decay(c, self.c_bar, o_t, delta_t, duration_seqs[t])
            c, self.c_bar, o_t, delta_t = self.recurrence(self.event_embeddings(event_seqs[t]), self.h_d, self.c_d, self.c_bar)
            h_list.append(self.h_d)
            c_list.append(c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)
        
        self.output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))

        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        
        seq_out = self.w(h.transpose(0,1)[:, -1, :])
        
        return seq_out

    def log_likelihood(self, event_seqs, time_seqs, next_time, seqs_length, batch_first=True):
        "Calculate log likelihood per sequence."
        batch_size, batch_length = event_seqs.shape
        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)
        diff_times = torch.cat([time_seqs, next_time.unsqueeze(-1)], -1)[:, 1:]
        hd_list = []
        for idx, sim_duration in enumerate(diff_times):
            _, h_d_idx = self.decay(c[:, idx], c_bar[:, idx], o[:, idx], delta[:, idx], sim_duration)
            hd_list.append(h_d_idx)
        h = torch.stack(hd_list, dim=0)
        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size).to(event_seqs.device)
        lambda_k = F.softplus(self.wa(h))
        lambda_k = torch.cat([lambda_k.new_ones(lambda_k.shape[0], lambda_k.shape[1], 1), lambda_k], -1)
        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            origin_ll = torch.log(lambda_k[idx, torch.arange(seq_len - 1).long(), event_seq[1:int(seq_len)].long()])
            non_pad_mask = (origin_ll == 0).sum() + 1
            origin_ll = origin_ll[non_pad_mask.long():]
            original_loglikelihood[idx] = torch.mean(origin_ll)

        # Calculate simulated loss from MCMC method
        h_d_list = []
        n_samples = 10
        diff_times = torch.cat([time_seqs, next_time.unsqueeze(-1)], -1)[:, 1:]
        sim_time = diff_times.unsqueeze(-1) * torch.rand([diff_times.shape[0], diff_times.shape[1], n_samples], device=diff_times.device)
        for idx, sim_duration in enumerate(sim_time):
            hd_list = []
            for i in range(n_samples):
                _, h_d_idx = self.decay(c[:, idx], c_bar[:, idx], o[:, idx], delta[:, idx], sim_duration[:, i])
                hd_list.append(h_d_idx)
            h_d_idx = torch.stack(hd_list, dim=0)
            h_d_list.append(h_d_idx)            
        h_d = torch.stack(h_d_list)
        
        non_pad_mask = (event_seqs != 0)[:, :-1] 
        sim_lambda_k = F.softplus(self.wa(h_d)).permute(0,2,1,3)
        simulated_likelihood = torch.zeros(batch_size).to(event_seqs.device)
        simulated_likelihood = (sim_lambda_k.sum(dim = -1).sum(dim = -1) * diff_times / n_samples).mean(dim = -1)

        loglikelihood = torch.mean(original_loglikelihood - simulated_likelihood)
        return loglikelihood
    
    def get_lambda(self, duration, c, c_bar, o, delta):
         _, h = self.decay(c.unsqueeze(0), c_bar.unsqueeze(0), o.unsqueeze(0), delta.unsqueeze(0), duration)
         lambda_k = F.softplus(self.wa(h))
         return lambda_k

    def get_time_and_type_prediction(self):
        """"by thinning algorithm, not used"""
        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        h = torch.squeeze(h, 0)[-1]
        c = torch.squeeze(c, 0)[-1]
        o = torch.squeeze(o, 0)[-1]
        c_bar = torch.squeeze(c_bar, 0)[-1]
        delta = torch.squeeze(c_bar, 0)[-1]
        c_stack = torch.stack([c, c_bar], dim=0)
        c_max = torch.max(c_stack, dim=0)[0]
        c_min = torch.min(c_stack, dim=0)[0]
        h_d = o * torch.tanh(c_max)
        h_t = o * torch.tanh(c_min)
        h_stack = torch.stack([h_d, h_t], dim=0)
        h_max = torch.max(h_stack, dim=0)[0]
        h_min = torch.min(h_stack, dim=0)[0]
        wa_weight = self.wa.weight
        pos_mask = (wa_weight > 0)
        neg_mask = torch.logical_not(pos_mask)
        lambda_max = torch.matmul(h_max, (wa_weight * pos_mask.int()).transpose(0,1)) + torch.matmul(h_min, (wa_weight * neg_mask.int()).transpose(0,1))
        lambda_max = F.softplus(lambda_max)
        
        t_pred_list = []
        type_pred_list = []
        for _ in range(99):
            t_k_list = []
            max_iter_num = 10
            for i in range(self.args.type_size):
                j = 0
                single_lambda_max = lambda_max[:, i]
                bs_mask = single_lambda_max.new_ones(single_lambda_max.shape[0]).bool()
                delta_t = 0
                while j < max_iter_num:
                    delta_t += exponential_rand(single_lambda_max) * (bs_mask.int())
                    u = torch.rand(delta_t.shape[0]).to(single_lambda_max.device)
                    lambda_k = (self.get_lambda(delta_t, c, c_bar, o, delta).squeeze(0))[:, i]
                    bs_mask = torch.logical_and(bs_mask, (u * single_lambda_max > lambda_k))
                    if (bs_mask.int().sum()) == 0:
                        break
                t_k_list.append(delta_t)
            t_k = torch.stack(t_k_list, dim=0)
            t_pred, type_pred = torch.min(t_k, dim=0)
            t_pred_list.append(t_pred)
            type_pred_list.append(type_pred)
        
        t_pred = torch.stack(t_pred_list, dim=-1)
        type_pred = torch.stack(type_pred_list, dim=-1)
        t_pred, indices = torch.sort(t_pred, dim=-1)
        type_pred = torch.stack([type_pred[idx, indices[idx]] for idx in range(type_pred.shape[0])], dim=0)
        return t_pred, type_pred
    
def exponential_rand(lam):
    neg_mask = (lam < 0)
    u = torch.rand(lam.shape[0]).to(lam.device)
    re = (-1.0 / lam) * torch.log(u)
    re[neg_mask] = 0
    return re



class IntensityNet(nn.Module):
    def __init__(self, args):
        super(IntensityNet, self).__init__()

        self.linear1 = nn.Linear(in_features=1, out_features=1)
        self.linear2 = nn.Linear(in_features=args.hidden_size+1, out_features=args.hidden_size)
        self.module_list = nn.ModuleList([nn.Linear(in_features=args.hidden_size, out_features=args.hidden_size) for _ in range(args.num_hidden_layers-1)])
        self.linear3 =  nn.Sequential(nn.Linear(in_features=args.hidden_size, out_features=1), nn.Softplus())

    def forward(self, hidden_state, target_time):

        for p in self.parameters():
            p.data *= (p.data>=0)

        t = self.linear1(target_time.unsqueeze(dim=-1))

        out = F.tanh(self.linear2(torch.cat([hidden_state[:,-1,:], t], dim=-1)))
        for layer in self.module_list:
            out = F.tanh(layer(out))
        int_lmbda = F.softplus(self.linear3(out))
        int_lmbda = torch.mean(int_lmbda)

        lmbda = grad(int_lmbda, target_time, create_graph=True, retain_graph=True)[0]
        nll = torch.add(int_lmbda, -torch.mean(torch.log((lmbda+1e-10))))

        return [nll, torch.mean(torch.log((lmbda+1e-10))), int_lmbda, lmbda]

class GTPP(nn.Module):
    "Code for FullyNN, referred to https://github.com/KanghoonYoon/torch-neuralpointprocess.git"
    def __init__(self, args):
        super(GTPP, self).__init__()
        self.batch_size = args.batch_size
        self.args = args
        self.event_embeddings = nn.Embedding(args.type_size+1, args.hidden_size)
        self.time_embeddings = nn.Embedding(args.barrel_size, args.hidden_size)
        self.emb_drop = nn.Dropout(p=self.args.hidden_dropout_prob)
        self.lstm = nn.LSTM(input_size=1+args.hidden_size,
                            hidden_size=args.hidden_size,
                            batch_first=True,
                            bidirectional=False)
        self.intensity_net = IntensityNet(args)

    def forward(self, event_seqs, time_seqs):
        event_seqs = event_seqs.long()
        emb = self.event_embeddings(event_seqs)
        emb = self.emb_drop(emb)
        lstm_input = torch.cat([emb, time_seqs.unsqueeze(-1)], dim=-1)
        hidden_state, _ = self.lstm(lstm_input)

        nll, log_lmbda, int_lmbda, lmbda = self.intensity_net(hidden_state, time_seqs[:, -1])
        return hidden_state[:, -1, :], nll
