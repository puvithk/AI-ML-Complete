import torch
from torch import nn
import numpy as np

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self , d_model , s_len , num_head):
        super(MultiHeadAttentionLayer , self).__init__()
        assert d_model %  num_head == 0  # Check weather the d_model is divisible by number of head 
        self.dv = d_model //  num_head
        self.d_model = d_model
        self.s_len = s_len
        self.num_head = num_head
        self.wq  = nn.Linear(d_model , d_model)
        self.wk = nn.Linear(d_model , d_model)
        self.wv = nn.Linear(d_model , d_model)
        self.wo = nn.Linear(d_model , d_model)
        # Shape - wq , wk ,wv ,wo - (d_model * d_model)
    def resize(self , x:torch.Tensor ):
        batch_size  , seq_len ,  _  = x.size()
        return x.view((batch_size , seq_len , self.num_head , self.dv)).transpose(1,2)
    
    def combine(self , x:torch.Tensor):
        batch_size , _ , seq_len , _ =  x.size()
        return x.transpose(1,2).contiguous().view((batch_size , seq_len , self.d_model)) 
    
    
    def scalar_dot_product(self , q:torch.Tensor,k:torch.Tensor,v:torch.Tensor  , masked:torch.Tensor =  None):
        attention_score:torch.Tensor = torch.matmul(q , k.transpose(2,3)) / np.sqrt(self.dv)
        if masked is not None:
            attention_score = attention_score.masked_fill(mask=0 , value=-1e9)
        softmax = torch.softmax(attention_score , dim=-1)  
        return torch.matmul(softmax , v)

    def forward(self , q:torch.Tensor,k:torch.Tensor,v:torch.Tensor , masked : torch.Tensor = None):
        q = self.resize(self.wq(q))
        k =  self.resize(self.wk(k))
        v = self.resize(self.wv(v))
        attention = self.scalar_dot_product(q,k,v , masked)
     
        return self.wo(self.combine(attention))
        
class AddAndNormalize():
    def __init__(self, eps=1e-6):
        self.eps = eps

    def layer_norm(self, x):
        mean =torch.mean(x, dim=-1, keepdims=True)
        var = torch.var(x, dim=-1, keepdims=True)
        return (x - mean) / torch.sqrt(var + self.eps)

    def forward(self, x, sublayer_output):
        added = x + sublayer_output
        return self.layer_norm(added)

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model ):
        super(FeedForwardNetwork, self).__init__()
        self.f1 = nn.Linear( d_model , 4*d_model )
        self.f2 =  nn.Linear(4*d_model ,d_model  )
        self.relu = nn.ReLU()
    def forward(self , x):
        return self.f2(self.relu(self.f1(x)))
class EncoderBlock(nn.Module):
    def __init__(self, d_model , seq_len , num_head):
        super(EncoderBlock , self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_head = num_head
        self.multi_attention_layer = MultiHeadAttentionLayer(d_model , seq_len , num_head)
        self.add_and_normalize = AddAndNormalize()
        self.feed_forward_network = FeedForwardNetwork(d_model )
    def forward(self, x):
        attention_output = self.multi_attention_layer(x , x , x)
        add_normalize_output = self.add_and_normalize.forward(x ,  attention_output)
        feed_forward_output = self.feed_forward_network(add_normalize_output)
        add_normalize_output = self.add_and_normalize.forward(add_normalize_output  , feed_forward_output)
        return add_normalize_output


batch = 5 
seq_len = 64
d_model = 1080
num_head = 8



X  = torch.rand((batch , seq_len , d_model))



