import numpy as np

class Transformer():
    def __init__(self):
        pass
    
    def transform(self, x):
        return x
    
    def inverse_transform(self, x):
        return x

class logTransform(Transformer):
    def __init__(self):
        super().__init__()
    
    def transform(self, x):
        return np.log(x)
    
    def inverse_transform(self, x):
        return np.exp(x)