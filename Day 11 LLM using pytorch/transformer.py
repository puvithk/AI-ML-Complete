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
            attention_score = attention_score.masked_fill(masked==0 , value=-1e9)
        softmax = torch.softmax(attention_score , dim=-1)  
        return torch.matmul(softmax , v)

    def forward(self , q:torch.Tensor,k:torch.Tensor,v:torch.Tensor , masked : torch.Tensor = None):
        q = self.resize(self.wq(q))
        k =  self.resize(self.wk(k))
        v = self.resize(self.wv(v))
        attention = self.scalar_dot_product(q,k,v , masked)
     
        return self.wo(self.combine(attention))
        
class AddAndNormalize(nn.Module):
    def __init__(self, eps=1e-6):
        super(AddAndNormalize, self).__init__()
        self.eps = eps

    def forward(self, x, sublayer_output):
        added = x + sublayer_output
        mean = torch.mean(added, dim=-1, keepdim=True)
        var = torch.var(added, dim=-1, keepdim=True)
        return (added - mean) / torch.sqrt(var + self.eps)
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
    


class DecoderBlock(nn.Module):
    def __init__(self, d_model , seq_len , num_head):
        super(DecoderBlock , self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_head = num_head
        self.masked_attention_layer = MultiHeadAttentionLayer(d_model , seq_len , num_head)
        self.add_and_normize_masked = AddAndNormalize()
        self.multi_attention_layer = MultiHeadAttentionLayer(d_model , seq_len , num_head)
        self.add_and_normize_attention = AddAndNormalize()
        self.feed_forward_network =  FeedForwardNetwork(d_model)
        self.add_and_normize_feed = AddAndNormalize()
    def forward(self , x , encoder_value , mask ):
        masked_attention = self.masked_attention_layer(x,x,x , masked=mask)
        add_and_normize_masked_output = self.add_and_normize_masked(x ,masked_attention )
        multi_attention = self.multi_attention_layer( add_and_normize_masked_output , encoder_value , encoder_value)
        add_and_normize_output = self.add_and_normize_attention(add_and_normize_masked_output ,multi_attention )
        feed_forward_output = self.feed_forward_network(add_and_normize_output)
        output = self.add_and_normize_feed(add_and_normize_output ,feed_forward_output )
        return output

class Mask(nn.Module):
    def __init__(self, seq_len):
        super(Mask, self).__init__()
        self.seq_len = seq_len

    def forward(self):
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,seq,seq)

class Transformer(nn.Module):
    def __init__(self , d_model , seq_len , num_head  , vocal_size):
        super(Transformer , self ).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_head = num_head
        self.encoder_layers = nn.ModuleList(
                [EncoderBlock(d_model, seq_len, num_head) for _ in range(4)]
            )

        self.decoder_layers = nn.ModuleList(
            [DecoderBlock(d_model, seq_len, num_head) for _ in range(4)]
        )
        self.linear =  nn.Linear(d_model , vocal_size)
        self.masked = Mask( seq_len)
    def forward(self, src, tgt):
        

        mask = self.masked.forward()
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)


        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output , mask)

        output = self.linear(dec_output)

        return output


batch = 5 
seq_len = 64
d_model = 1080
num_head = 8



X = torch.rand((batch, seq_len, d_model))
Y = torch.rand((batch, seq_len, d_model))

transform = Transformer(d_model, seq_len, num_head, 516)

output = transform(X, Y)

print(output.shape)