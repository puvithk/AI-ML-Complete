import numpy as np 
def sinusodal_encodings(Dimension, max_len=1000):
    for i in range(20):
        for j in range(Dimension):
            if j % 2 == 0:
                encoding = np.sin(i / (10000 ** (j / Dimension)))
            else:
                encoding = np.cos(i / (10000 ** ((j - 1) / Dimension)))
            print(f"Position: {i}, Dimension: {j}, Encoding: {encoding}")
sinusodal_encodings(4)