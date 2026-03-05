
import numpy as np
class Embeddings:
    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
    def token_embedings(self, text: str) -> np.array:
        return text.split()
    
class PositionalEmbedings:
    def __init__(self , d_model):
        self.d_model = d_model
    def apply_sisodial_embedings(self , token_array : np.ndarray)-> np.ndarray:
        B, S , D = token_array.shape
        for i in range(S):
            for j in range(D):
                if j %2 == 0 :
                    token_array[: , i ,  j ] += np.sin(i / (10000 ** (j / self.d_model)))
                else :
                    token_array[: , i ,  j ] += np.cos(i / (10000 ** (j / self.d_model)))
        return token_array
    
class MultiHeadSelfAttentionLayer:
    def __init__(self ,d_model , no_of_head = 16 ):
    
        self.d_model = d_model
        self.no_head = no_of_head
        
        self.dh = self.d_model // self.no_head    
        seed = 42
        self.rng = np.random.default_rng(seed)

        self.wq = self.rng.standard_normal(( self.no_head ,self.d_model, self.dh  ))
        self.wk = self.rng.standard_normal((self.no_head ,self.d_model, self.dh ))
        self.wv = self.rng.standard_normal((self.no_head ,self.d_model, self.dh))
        self.wo = self.rng.standard_normal((self.d_model, self.d_model ))
    def print_weights(self): 
        print("Weights Key")
        print(self.wk)
        print("Weights Query")
        print(self.wq)
        print("Weights Output")
        print(self.wo)
        print("Weights Values")
        print(self.wv)
    def softmax(self , x :np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    def forward(self , X ):
        final_output = []
        for i in range(self.no_head):
            curr_head = X 
            q = curr_head @  self.wq[i]   
            k = curr_head @ self.wk[i]
            v = curr_head @ self.wv[i]
        
            k_t = np.transpose(k , (0 , 2 , 1 ))
            qk = q @ k_t
            softmax_output = self.softmax(qk/np.sqrt(self.dh)) 
            head_output   = softmax_output @ v
            final_output.append(head_output)
        concated = np.concatenate(final_output , axis=2)
     
        output = concated @ self.wo
        return output
    def forward(self , xk ,xq , xv ):
        final_output = []
        for i in range(self.no_head):
            
            q = xq @  self.wq[i]   
            k = xk @ self.wk[i]
            v = xv @ self.wv[i]
        
            k_t = np.transpose(k , (0 , 2 , 1 ))
            qk = q @ k_t
            softmax_output = self.softmax(qk/np.sqrt(self.dh)) 
            head_output   = softmax_output @ v
            final_output.append(head_output)
        concated = np.concatenate(final_output , axis=2)

        output = concated @ self.wo
        return output
class AddAndNormalize:
    def __init__(self, eps=1e-6):
        self.eps = eps

    def layer_norm(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.eps)

    def forward(self, x, sublayer_output):
        added = x + sublayer_output
        return self.layer_norm(added)


class FeedForwardNetwork:
    def __init__(self  , d_model,n_token , learning_rate = 0.01):
        self.learing_rate = learning_rate
        self.d_model = d_model
        self.rng = np.random.default_rng(42)
        self.w2 = self.rng.standard_normal(( d_model * 4 , d_model))
        self.w1 = self.rng.standard_normal(( d_model , d_model * 4))
        self.b2 = np.zeros((d_model, ))
        self.b1 = np.zeros((d_model*4, ))
    def relu(self , x :np.ndarray) -> np.ndarray:
        x = np.maximum(x , 0)
        return x 
    def forward(self , x : np.ndarray) -> np.ndarray :
   
        x = x @ self.w1 + self.b1
        x = self.relu(x)
        x = x @ self.w2 + self.b2 
        return x 
    
class Encoder:
    def __init__(self  , d_model :int , no_head : int , no_token:int ):
        self.d_model = d_model
        self.no_head = no_head
        self.no_token = no_token
        if(d_model % no_head) != 0 :
           
            raise ValueError("no_head must be the factor of d_model")
        self.posistional_embedings  = PositionalEmbedings(d_model)
        self.multi_head_attention =  MultiHeadSelfAttentionLayer(self.d_model , self.no_head )
        self.feed_forward = FeedForwardNetwork(self.d_model ,  self.no_token )
        self.add_and_normalize = AddAndNormalize()
    def forward(self , input_embedings ):
        posistional_embeding = self.posistional_embedings.apply_sisodial_embedings(input_embedings)
        multi_attention1 = self.multi_head_attention.forward(posistional_embeding , posistional_embeding , posistional_embeding)
        add_normaize1 = self.add_and_normalize.forward(posistional_embeding ,multi_attention1)
        feed_forward1  = self.feed_forward.forward(add_normaize1)
        add_normaize2 = self.add_and_normalize.forward(add_normaize1 , feed_forward1)
        return add_normaize2

class MaskedMultiHeadSelfAttention:

    def __init__(self, d_model, no_of_head=16):

        self.d_model = d_model
        self.no_head = no_of_head
        self.dh = d_model // no_of_head

        self.rng = np.random.default_rng(42)

        self.wq = self.rng.standard_normal((no_of_head, d_model, self.dh))
        self.wk = self.rng.standard_normal((no_of_head, d_model, self.dh))
        self.wv = self.rng.standard_normal((no_of_head, d_model, self.dh))
        self.wo = self.rng.standard_normal((d_model, d_model))

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def create_mask(self, seq_len):

        mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        mask = mask * -1e9
        return mask

    def forward(self, X):

        B, S, D = X.shape

        mask = self.create_mask(S)

        final_output = []

        for i in range(self.no_head):

            q = X @ self.wq[i]
            k = X @ self.wk[i]
            v = X @ self.wv[i]

            k_t = np.transpose(k, (0, 2, 1))

            qk = q @ k_t
            qk = qk / np.sqrt(self.dh)

            #  MASK
            qk = qk + mask

            softmax_output = self.softmax(qk)

            head_output = softmax_output @ v

            final_output.append(head_output)

        concated = np.concatenate(final_output, axis=2)

        output = concated @ self.wo

        return output


class Decoder:
    def __init__(self  , d_model :int , no_head : int , no_token:int ):
        self.d_model = d_model
        self.no_head = no_head
        self.no_token = no_token
        if(d_model % no_head) != 0 :
            print("no_head must be the factor of d_model")
            raise ValueError("no_head must be the factor of d_model")
        self.posistional_embedings  = PositionalEmbedings(d_model)
        self.multi_head_attention =  MultiHeadSelfAttentionLayer(self.d_model , self.no_head )
        self.masked_multi_attention_layer = MaskedMultiHeadSelfAttention(self.d_model , self.no_head)
        self.feed_forward = FeedForwardNetwork(self.d_model ,  self.no_token )
        self.add_and_normalize = AddAndNormalize()
    def forward(self , output_embedings , encoder_output ):
        posistional_embedings  = self.posistional_embedings.apply_sisodial_embedings(output_embedings)

        masked_multi_attention_layer = self.masked_multi_attention_layer.forward(posistional_embedings )

        add_norm_1 = self.add_and_normalize.forward(posistional_embedings , masked_multi_attention_layer)

        multi_head_attention = self.multi_head_attention.forward(xk=encoder_output , xv=encoder_output , xq=add_norm_1)

        add_norm_2 = self.add_and_normalize.forward(add_norm_1 , multi_head_attention)

        feed_forward_1 = self.feed_forward.forward(add_norm_2)
        
        add_norm_3 = self.add_and_normalize.forward(add_norm_2 , feed_forward_1)
        return add_norm_3

# Dummy code given by GPT 
if __name__ == "__main__":

    # Model parameters
    d_model = 64
    no_head = 8
    seq_len = 10
    batch_size = 2
    vocab_size = 100

    # Create encoder and decoder
    encoder = Encoder(d_model, no_head, vocab_size)
    decoder = Decoder(d_model, no_head, vocab_size)

    # Dummy input embeddings (random)
    encoder_input = np.random.randn(batch_size, seq_len, d_model)

    # Pass through encoder
    encoder_output = encoder.forward(encoder_input)

    print("Encoder Output Shape:")
    print(encoder_output.shape)

    # Dummy decoder input
    decoder_input = np.random.randn(batch_size, seq_len, d_model)

    # Pass through decoder
    decoder_output = decoder.forward(decoder_input, encoder_output)

    print("Decoder Output Shape:")
    print(decoder_output.shape)