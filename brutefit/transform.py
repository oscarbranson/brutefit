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
        self.checkvalid(x)
        return np.log(x)
    
    def inverse_transform(self, x):
        return np.exp(x)

    def checkvalid(self, x):
        if np.any(x < 0):
            raise ValueError('logTransform cannot handle negative values.')