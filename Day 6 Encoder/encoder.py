import numpy as np
# Step one  embeding 
# Step 2 Positioal embeding 
# Step 3 Multi attention layer 
# Step 4 Adding and normalizer (Residual layers)
# Step 5 Feed forward layers 
# Step 6 Adding and normalizer (Residual layers)
# Return value to give to the Decoders 

#This file talks about the encoder 
class Encoder:
    Wq =  np.ndarray
    Wk = np.ndarray
    Wv = np.ndarray
    def __init__(self, d_model: int, no_head: int = 16):
        self.d_model = d_model
        self.no_head = no_head
        
        self.dh = d_model // no_head    

        self.Wq = np.random.randn(d_model, self.dh)
        self.Wk = np.random.randn(d_model, self.dh)
        self.Wv = np.random.randn(d_model, self.dh)
        self.Wo = np.random.randn(d_model, d_model)

    def print_weights(self):
        print("Weights Key")
        print(self.Wk)
        print("Weights Query")
        print(self.Wq)
        print("Weights Output")
        print(self.Wo)
        print("Weights Values")
        print(self.Wv)

    def multi_Attention_layer(self ,X : np.ndarray) -> np.ndarray:
        return np.array()

    def softmax(self ,X : np.array) -> np.array :
        return np.array()

    def normalize(self ,X : np.array, input : np.array)-> np.array:
        """
        Function used to add the input and make layer normalization 
        
        Keyword arguments:
        Return: Normalized array with same dimension as the input 
        """
        
        return np.array

    def feed_forward_layer(self ,X: np.array)-> np.array:
        """
        Add a simple relu function for feed forward layer 
        
    
        Return: np.array oncee after applying relu 
        """
        
        return np.array

    def encoder(self ,X :np.array ) -> np.array:
        """
        This method is used to encoder where we take the the input matrix and 
        we perform task starting from 
        Multihead attention -> Add & Normalize -> Feed Forward netwrok -> Add & Normalize -> Output or return 

        Input :
        X : np.array which is the tokens or vector or tensorof the token after adding the posistionl vector 
        Output :
        np.arry which is the vector after the process 

        """    
        skip_input = X
        multihead = self.multi_Attention_layer(X)
        add_norm_1 = self.normalize(multihead , skip_input)
        feed_forward = self.feed_forward_layer(add_norm_1)
        output  = self.adding_normalize(feed_forward , add_norm_1)
        return output


def token_embedings( text : str)-> np.array:
    return np.array(np.array([])) 

def posistional_embeding(token_embedings : np.array) -> np.array:
    return np.array([])


encoder = Encoder(64)
encoder.print_weights()