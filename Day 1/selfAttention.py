import numpy as np 
#Starting 
#Created by : Puvith kumar 

def softmax(X):
    ex = np.exp(X)
    return ex / np.sum(ex, axis=1 , keepdims=True)

def self_attention(Q: np.array  , K: np.array , V: np.array):
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
    Dk = K.shape[1]
    QK = np.dot(Q, x_transpose)/np.sqrt(Dk) 
    softmax_output =  softmax(QK)
    attention = np.dot(softmax_output , V)
    return attention 




Q = np.array([
    [1, 0, 1],
    [0, 1, 1]
])

K = np.array([
    [1, 1, 0],
    [0, 1, 1]
])


V = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

self_attention_output = self_attention(Q, K, V)
print("Self-Attention Output:")
print(self_attention_output)

#  Ending 

d_k = K.shape[1]

scores = np.matmul(Q, K.T)
print("QK^T:\n", scores)
scaled_scores = scores / np.sqrt(d_k)
print("Scaled Scores:\n", scaled_scores)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scaled_scores)
print("Attention Weights:\n", attention_weights)

output = np.matmul(attention_weights, V)
print("Final Output:\n", output)