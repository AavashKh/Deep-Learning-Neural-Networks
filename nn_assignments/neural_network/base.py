import numpy as np
import pickle

class Layer:
    def forward( self, *args, **kwargs ):
        raise NotImplementedError

    def backward( self, *args, **kwargs ):
        raise NotImplementedError
