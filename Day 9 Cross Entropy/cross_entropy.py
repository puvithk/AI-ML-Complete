import numpy as np 
class CrossEntropy():
    def binary_cross_entropy_loss(self, y_pred :int, y_acctual :int) -> float:
        return (-y_acctual * np.log(y_pred) ) -((1-y_acctual )*(np.log(1-y_pred)))
        
    def binary_cross_entorpy_cost(self, y_pred : np.ndarray , y_acctual:np.ndarray)-> float :
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss =  (-y_acctual * np.log(y_pred) ) -((1-y_acctual )*(np.log(1-y_pred)))
        return (1/len(y_acctual))*sum(loss)
    def categorical_cross_entropy(self , y_pred :np.ndarray , y_acctual :np.ndarray) -> float:
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.sum(y_acctual * np.log(y_pred), axis=1)
        return np.mean(loss)
        
cross_entropy = CrossEntropy()

y_actual = np.array([1, 0, 1, 0, 1])

y_pred = np.array([0.9, 0.2, 0.8, 0.1, 0.7])
print(cross_entropy.binary_cross_entorpy_cost(y_pred , y_actual))
y_actual = np.array([
[1,0,0],
[0,1,0],
[0,0,1]
])

y_pred = np.array([
[0.7,0.2,0.1],
[0.1,0.8,0.1],
[0.2,0.2,0.6]
])

print(cross_entropy.categorical_cross_entropy(y_pred,y_actual))