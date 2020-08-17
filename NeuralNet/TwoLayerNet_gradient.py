import sys, os
sys.path.append('D:\\DEEP_LEARNING_FROM_SCRATCH')
from mnist_data.mnist import *
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size)