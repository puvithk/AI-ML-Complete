import numpy as np
# Step one  embeding 
# Step 2 Positioal embeding 
# Step 3 Multi attention layer 
# Step 4 Adding and normalizer (Residual layers)
# Step 5 Feed forward layers 
# Step 6 Adding and normalizer (Residual layers)
# Return value to give to the Decoders 

# This file talks about the encoder 
# class Encoder:
#     Wq =  np.ndarray
#     Wk = np.ndarray
#     Wv = np.ndarray
#     def __init__(self, d_model: int, no_head: int = 16):
#         self.d_model = d_model
#         self.no_head = no_head
        
#         self.dh = d_model // no_head    

#         self.Wq = np.random.randn(d_model, self.dh)
#         self.Wk = np.random.randn(d_model, self.dh)
#         self.Wv = np.random.randn(d_model, self.dh)
#         self.Wo = np.random.randn(d_model, d_model)

#     def print_weights(self):
#         print("Weights Key")
#         print(self.Wk)
#         print("Weights Query")
#         print(self.Wq)
#         print("Weights Output")
#         print(self.Wo)
#         print("Weights Values")
#         print(self.Wv)

#     def multi_Attention_layer(self ,X : np.ndarray) -> np.ndarray:
#         return np.array()

#     def softmax(self ,X : np.array) -> np.array :
#         return np.array()

#     def normalize(self ,X : np.array, input : np.array)-> np.array:
#         """
#         Function used to add the input and make layer normalization 
        
#         Keyword arguments:
#         Return: Normalized array with same dimension as the input 
#         """
        
#         return np.array

#     def feed_forward_layer(self ,X: np.array)-> np.array:
#         """
#         Add a simple relu function for feed forward layer 
        
    
#         Return: np.array oncee after applying relu 
#         """
        
#         return np.array

#     def encoder(self ,X :np.array ) -> np.array:
#         """
#         This method is used to encoder where we take the the input matrix and 
#         we perform task starting from 
#         Multihead attention -> Add & Normalize -> Feed Forward netwrok -> Add & Normalize -> Output or return 

#         Input :
#         X : np.array which is the tokens or vector or tensorof the token after adding the posistionl vector 
#         Output :
#         np.arry which is the vector after the process 

#         """    
#         skip_input = X
#         multihead = self.multi_Attention_layer(X)
#         add_norm_1 = self.normalize(multihead , skip_input)
#         feed_forward = self.feed_forward_layer(add_norm_1)
#         output  = self.adding_normalize(feed_forward , add_norm_1)
#         return output


def token_embedings( text : str)-> np.array:
    return np.array(np.array([])) 

def posistional_embeding(token_embedings : np.array) -> np.array:
    return np.array([])




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
        print(concated.shape)
        print(self.wo.shape)
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
        self.w2 = self.rng.standard_normal(( n_token, d_model))
        self.w1 = self.rng.standard_normal(( d_model ,n_token))
        self.b2 = np.zeros((d_model, ))
        self.b1 = np.zeros((n_token, ))
    def relu(self , x :np.ndarray) -> np.ndarray:
        x = np.maximum(x , 0)
        return x 
    def forward(self , x : np.ndarray) -> np.ndarray :
        print(x.shape)
        print(self.w1.shape)
        print(self.b1.shape)
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
            print("no_head must be the factor of d_model")
            raise ValueError("no_head must be the factor of d_model")
        self.posistional_embedings  = PositionalEmbedings(d_model)
        self.multi_head_attention =  MultiHeadSelfAttentionLayer(self.d_model , self.no_head )
        self.feed_forward = FeedForwardNetwork(self.d_model ,  self.no_token )
        self.add_and_normalize = AddAndNormalize()
    def forward(self , input_embedings ):
        posistional_embeding = self.posistional_embedings.apply_sisodial_embedings(input_embedings)
        multi_attention1 = self.multi_head_attention.forward(posistional_embeding)
        add_normaize1 = self.add_and_normalize.forward(posistional_embeding ,multi_attention1)
        feed_forward1  = self.feed_forward.forward(add_normaize1)
        add_normaize2 = self.add_and_normalize.forward(add_normaize1 , feed_forward1)
        return add_normaize2

# Testing Multihead attentions layers 

# d_model = 81 # This must be the multiple of no_head 
# no_head = 9
# no_token =  20
# Create object of mulitheadattention 
# multi_head_selfattention_layer = MultiHeadSelfAttentionLayer(d_model=d_model , no_of_head=no_head)

batch_size = 5
sequence_length = 20
d_model = 128

vector = np.random.rand(batch_size,  sequence_length, d_model)

encoder = Encoder(d_model=128, no_head=8, no_token=20)
print(vector.shape)
output = encoder.forward(vector)

print(output.shape)
print(output)


