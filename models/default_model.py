import tensorflow as tf
import os,\
    sys,\
    inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from tools.tf_fuc import weight_variable, \
    bias_variable, \
    conv1d, \
    max_pooling1d, \
    fc, \
    batch_norm
from global_config.global_config import config
cfg = config()


class default():
    def __init__(self, inputs):

        self.layers = cfg.layer

        for i, j in enumerate(self.layers):

            if j[0] == 'conv1d':
                inputs = conv1d(inputs,
                                 j[1:],
                                j[0]+'_'+str(i))

            if j[0] == 'maxpooling':
                inputs = max_pooling1d(inputs)

            if j[0] == 'fc':
                inputs = fc(inputs,
                            j[1:],
                            j[0]+'_'+str(i))

            if j[0] == 'flatten':
                inputs = tf.reshape(inputs,
                                    [-1, j[1]])
        self.inputs = inputs

