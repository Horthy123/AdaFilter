import copy
import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def nufftfreqs(M, df=1):
    return df * torch.arange(0, (M // 2) + 1)

def matmul_complex(t1,t2):

    return torch.view_as_complex(torch.stack((torch.bmm(t1.real, t2.real) - torch.bmm(t1.imag, t2.imag), torch.bmm(t1.real, t2.imag) + torch.bmm(t1.imag, t2.real)),dim=-1))

def aft(t, x, M):
    sign = -1
    df = 1.0
    x = torch.view_as_complex(torch.stack([x, x.new_zeros(x.shape)], dim=-1))
    c = torch.exp(sign * 2 * np.pi* 1j * nufftfreqs(M, df).unsqueeze(0).unsqueeze(-1).to(t.device) * t.unsqueeze(1))
    a = matmul_complex(c, x)
    return a / np.sqrt(a.shape[1])

def iaft(t, y, M):
    sign = 1
    df = 1.0
    c = torch.exp(sign  * 2 * np.pi* 1j * (nufftfreqs(M, df)).unsqueeze(0).unsqueeze(-1).to(t.device) * t.unsqueeze(1))
    c = c.transpose(1,2)
    a = matmul_complex(c, y)
    return a / np.sqrt(a.shape[1])

def get_moor(x):
    return x.real

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x): 
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, x):
        u = x.mean(-1, keepdim = True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class FilterLayer(nn.Module):
    def __init__(self, args):
        super(FilterLayer, self).__init__()
        self.complex_weight = nn.Parameter(torch.randn(1, args.max_seq_length//2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)

    def forward(self, input_tensor, time = None):
        batch, seq_len, hidden = input_tensor.shape
        if time is not None:
            time = torch.cumsum(time, dim=1) 
            time =  (1 - time / torch.clamp(torch.max(time, dim=-1)[0].unsqueeze(-1), min=1e-5)) * (seq_len - 1) / (seq_len) / 4
            x = aft(time, input_tensor, seq_len)
        else:
            x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        sequence_emb_fft = iaft(time, x, seq_len)
        sequence_emb_fft = get_moor(sequence_emb_fft)

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    
class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(4 * args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states
    
class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.filterlayer = FilterLayer(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, time = None):
        hidden_states = self.filterlayer(hidden_states, time)
        intermediate_output = self.intermediate(hidden_states)
        return intermediate_output
    
class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, time = None, output_all_encoded_layers=True):
        all_encoder_layers = []
        layer_num = 0
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, time)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            layer_num = layer_num + 1
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers