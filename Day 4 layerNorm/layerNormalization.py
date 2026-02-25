import numpy as np 

def layer_normalization(x, epsilon=1e-5):
    mean = np.mean(x, axis=1, keepdims=True)
    variance = np.var(x, axis=1, keepdims=True)
    normalized_x = (x - mean) / np.sqrt(variance + epsilon)
    return normalized_x


layer_input = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
normalized_output = layer_normalization(layer_input)
print("Input:\n", layer_input)
print("Output : " , normalized_output)
print("Mean of output:", np.mean(normalized_output, axis=-1))
print("Variance of output:", np.var(normalized_output, axis=-1))