import numpy as np 
# #Starting 
# #Created by : Puvith kumar 

def softmax(X):
    X = X - np.max(X, axis=1, keepdims=True)
    ex = np.exp(X)
    return ex / np.sum(ex, axis=1 , keepdims=True)
def self_attention(Q: np.array  , K: np.array , V: np.array , Dk : int = None):
    """
    Attention = softmax(Q K^T /√Dk)*V
    Q - Query 
    K - Key
    V - Value
    K^T - transpose of K  
    Dk - Dimention of the K 
    V - Value 
    """
    x_transpose = np.transpose(K)
    if Dk is None:
        Dk = K.shape[0]
    QK = np.dot(Q, x_transpose)/np.sqrt(Dk) 
    softmax_output =  softmax(QK)
    attention = np.dot(softmax_output , V)
    return attention 
# import numpy as np


# Q = np.array([
#     [1, 0, 1],
#     [0, 1, 1]
# ])

# K = np.array([
#     [1, 1, 0],
#     [0, 1, 1]
# ])


# V = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])

# self_attention_output = self_attention(Q, K, V)
# print("Self-Attention Output:")
# print(self_attention_output)



# #  Ending 
# def self_attention(Q: np.array  , K: np.array , V: np.array):
    
#     d_k = K.shape[1]

#     scores = np.matmul(Q, K.T)
#     print("QK^T:\n", scores)
#     scaled_scores = scores / np.sqrt(d_k)
#     print("Scaled Scores:\n", scaled_scores)

#     def softmax(x):
#         exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
#         return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

#     attention_weights = softmax(scaled_scores)
#     print("Attention Weights:\n", attention_weights)

#     output = np.matmul(attention_weights, V)
#     print("Final Output:\n", output)



    #===================================DAY 2====================================#

import numpy as np
num_head = 2

# Input (Embedding_dim=4, Tokens=3)
X = np.array([
    [1, 0, 1, 0],   # Token 1 embedding
    [0, 1, 1, 1],   # Token 2 embedding
    [1, 1, 0, 0]    # Token 3 embedding
])

# Head 1 Weights
Wk = [
    np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]),
    
    np.array([
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ])
]

Wq = [
    np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]),
    
    np.array([
        [1, 0, 0, 1],
        [0, 1, 1, 0]
    ])
]

Wv = [
    np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]),
    
    np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
]
W0 = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
])
def multi_head_attention( X : np.array , num_head : int):
    """
    Multi head attention 
    Create multipe Wk , Wq, Wv 
    
    """
    concate = []
    for i in range(num_head):
        Wk_head = Wk[i]
        Wq_head =  Wq[i]
        Wv_head = Wv[i]
        attention = self_attention(np.matmul( X , Wq_head.T ) , np.matmul( X ,Wk_head.T) , np.matmul( X , Wv_head.T ) , Dk = num_head)
        concate.append(attention)
    z =  np.concatenate(concate , axis=1)
    print(z.shape , "z shape")
    print(z)
    print(W0.shape , "W0 shape")
    return np.matmul( z,W0.T )

print(multi_head_attention(X , 2))