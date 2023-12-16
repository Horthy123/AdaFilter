import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import cont2discrete
import numpy as np
import ptwt, pywt

ACT2FN = {
        "gelu": lambda x: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0))), 
        "relu": F.relu, 
        "swish": lambda x: x * torch.sigmoid(x)}

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #attention_probs = attention_probs.masked_fill(~attention_mask, torch.tensor(0.0))

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states


class FastSelfAttention(SelfAttention):
    def __init__(self, args):
        super().__init__(args)

        self.query_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.key_att = nn.Linear(self.all_head_size, self.num_attention_heads)
        self.transform = nn.Linear(self.all_head_size, self.all_head_size)

    def forward(self, input_tensor, attention_mask):
        batch, seq_len, _ = input_tensor.shape
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        query_for_score = self.query_att(mixed_query_layer).transpose(1, 2) / (self.attention_head_size ** 0.5) # batch, num_head, seq_len
        query_for_score += attention_mask[:, :, -1]
        query_weight = nn.Softmax(dim=-1)(query_for_score).unsqueeze(2)
        query_layer = self.transpose_for_scores(mixed_key_layer)
        pooled_query = torch.matmul(query_weight, query_layer).transpose(1, 2).view(-1,1,self.num_attention_heads*self.attention_head_size)
        pooled_query_repeat= pooled_query.repeat(1, seq_len,1) # batch_size, num_head, seq_len, head_dim

        mixed_query_key_layer=mixed_key_layer * pooled_query_repeat
        query_key_score=(self.key_att(mixed_query_key_layer)/ self.attention_head_size**0.5).transpose(1, 2)
        query_key_score += attention_mask[:, :, -1]
        query_key_weight = nn.Softmax(dim=-1)(query_key_score).unsqueeze(2)
        key_layer = self.transpose_for_scores(mixed_query_key_layer)
        pooled_key = torch.matmul(query_key_weight, key_layer)

        #query = value
        weighted_value =(pooled_key * query_layer).transpose(1, 2)
        weighted_value = weighted_value.reshape(
            weighted_value.size()[:-2] + (self.num_attention_heads * self.attention_head_size,))
        weighted_value = self.out_dropout(weighted_value)
        hidden_states = self.transform(weighted_value) + mixed_key_layer

        return hidden_states
    
def get_frequency_modes(seq_len, modes=4, mode_select_method='random'):
    """
    get modes on frequency domain:
    'random' means sampling randomly;
    'else' means sampling the lowest modes;
    """
    modes = min(modes, seq_len//2)
    if mode_select_method == 'random':
        index = list(range(0, seq_len // 2))
        np.random.shuffle(index)
        index = index[:modes]
    else:
        index = list(range(0, modes))
    index.sort()
    return index
    

class FedformerLayer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.n_head = args.num_attention_heads
        self.d_k = args.hidden_size // self.n_head
        seq_len = args.max_seq_length
        self.w_qs = nn.Linear(args.hidden_size, args.hidden_size)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (args.hidden_size + self.d_k)))

        self.index = get_frequency_modes(seq_len, modes=4, mode_select_method='random')
        self.scale = (1 / (args.hidden_size * args.hidden_size))
        self.weights1 = nn.Parameter(self.scale * torch.rand(self.n_head, self.d_k, self.d_k, len(self.index), dtype=torch.cfloat))

        self.layer_norm = nn.LayerNorm(args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bhi,hio->bho", input, weights)
    

    def forward(self, x, mask=None):

        d_k, n_head = self.d_k, self.n_head

        bs, seq_len, _ = x.size()
        # residual = self.res(q)

        q = self.w_qs(x).view(bs, seq_len, n_head, d_k)
        q = q.permute(0, 2, 3, 1) # bs, n, dk, seq
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(q, dim=-1)
        # Perform Fourier neural operations
        out_ft = torch.zeros(bs, n_head, d_k, seq_len // 2 + 1, device=x.device, dtype=torch.cfloat)
        for wi, i in enumerate(self.index):
            out_ft[:, :, :, wi] = self.compl_mul1d(x_ft[:, :, :, i], self.weights1[:, :, :, wi])
        # Return to time domain
        x_if = torch.fft.irfft(out_ft, n=q.shape[-1]) # bs, h, e, s
        output = x_if.permute(0, 3, 2, 1).contiguous().view(bs, seq_len, -1) # b x lq x (n*dv)
        output = self.dropout(output)
        output = self.layer_norm(output + x)

        return output


class GRULayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        if args.use_bidirect == 1:
            self.gru = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size//2, batch_first=True, bidirectional=True, num_layers=args.num_rnn_layers)
        else:
            self.gru = nn.GRU(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=False, num_layers=args.num_rnn_layers)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.use_layernorm = args.use_layernorm

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x, _ = self.gru(input_tensor)
        hidden_states = self.out_dropout(x) + input_tensor
        if self.use_layernorm:
            hidden_states = self.layernorm(hidden_states)

        return hidden_states
    
class LSTMLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        if args.use_bidirect == 1:
            self.lstm = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size//2, batch_first=True, bidirectional=True, num_layers=args.num_rnn_layers)
        else:
            self.lstm = nn.LSTM(input_size=args.hidden_size, hidden_size=args.hidden_size, batch_first=True, bidirectional=False, num_layers=args.num_rnn_layers)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)
        self.use_layernorm = args.use_layernorm

    def forward(self, input_tensor):
        batch, seq_len, hidden = input_tensor.shape
        x, _ = self.lstm(input_tensor)
        hidden_states = self.out_dropout(x) + input_tensor
        if self.use_layernorm:
            hidden_states = self.layernorm(hidden_states)

        return hidden_states

class Intermediate(nn.Module):
    """
    The efficacy of FFN remains to be explored.
    """
    def __init__(self, args):
        super().__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * args.ffn_multiplier)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.ffn_multiplier * args.hidden_size, args.hidden_size)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense_2(hidden_states) 
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_tensor)
        return hidden_states

class PoolingSelfAttention(SelfAttention):
    """
    Canonical PoolingFormer
    """
    def __init__(self, args):
        super().__init__(args)
        self.local_mask = nn.Parameter(
            (
                torch.triu(torch.ones(args.max_seq_length, args.max_seq_length), args.local) +
                torch.tril(torch.ones(args.max_seq_length, args.max_seq_length), -args.local)
            ) * -1e4, 
            requires_grad=False)
        self.query_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.key_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.value_2 = nn.Linear(args.hidden_size, self.all_head_size)
        self.pool = nn.MaxPool1d(kernel_size=args.pool_size, stride=args.pool_size)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        mask = self.local_mask
        attention_scores = attention_scores + mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        output_1 = context_layer.view(*new_context_layer_shape)

        mixed_query_layer = self.query_2(output_1)
        mixed_key_layer = self.pool(self.key_2(output_1).transpose(1, 2)).transpose(1, 2)
        mixed_value_layer = self.pool(self.value_2(output_1).transpose(1, 2)).transpose(1, 2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = context_layer
        hidden_states = self.layernorm(hidden_states + input_tensor)

        return hidden_states    


class MetaFormerBlock(nn.Module):
    """
    Construct a metaFormer layer with args.model_name
    """
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        self.intermediate = Intermediate(args)

        models = {
            'THP': SelfAttention,
            'FastSelfAttn': FastSelfAttention,
            'PoolingSelfAttn': PoolingSelfAttention,
            'GRU': GRULayer,
            'Fedformer': FedformerLayer
        }
        self.attention = models[args.model_name](args)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output

class CananicalBlock(nn.Module):
    """
    Construct a layer with args.model_name, i.e., without FFN.
    """
    def __init__(self, args):
        super().__init__()

        self.model_name = args.model_name
        models = {
            'THP': SelfAttention,
            'FastSelfAttn': FastSelfAttention,
            'PoolingSelfAttn': PoolingSelfAttention,
            'GRU': GRULayer,
            'Fedformer': FedformerLayer
        }
        self.attention = models[args.model_name](args)

    def forward(self, hidden_states, attention_mask):
        hidden_states = self.attention(hidden_states, attention_mask)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = MetaFormerBlock(args) if args.use_metaformer is True else CananicalBlock(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, times, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers