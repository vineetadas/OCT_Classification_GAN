# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 09:38:18 2019

@author: vineeta
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:03:47 2018

@author: vineeta
"""


import numpy as np
import os
import time
import scipy.io
from keras import backend as K
from keras.preprocessing import image
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout, merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.layers import BatchNormalization, Conv2DTranspose
from keras.layers import LeakyReLU
from keras.models import Sequential
from PIL import Image
from sklearn import preprocessing     
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import classification_report,confusion_matrix,cohen_kappa_score
import itertools
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras import backend
from keras.layers import Lambda
from sklearn.metrics import roc_curve, auc
#loaded_model=load_model('sc_cnn_6000.h5')
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import math
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
set_session(sess) 

############### Minibatch Discrimination Layer ##########################

from keras import backend as K
from keras.engine import InputSpec, Layer
from keras import initializers, regularizers, constraints



# From a PR that is not pulled into Keras
# https://github.com/fchollet/keras/pull/3677
# I updated the code to work on Keras 2.x

class MinibatchDiscrimination(Layer):
    """Concatenates to each sample information about how different the input
    features for that sample are from features of other samples in the same
    minibatch, as described in Salimans et. al. (2016). Useful for preventing
    GANs from collapsing to a single output. When using this layer, generated
    samples and reference samples should be in separate batches.
    # Example
    ```python
        # apply a convolution 1d of length 3 to a sequence with 10 timesteps,
        # with 64 output filters
        model = Sequential()
        model.add(Convolution1D(64, 3, border_mode='same', input_shape=(10, 32)))
        # now model.output_shape == (None, 10, 64)
        # flatten the output so it can be fed into a minibatch discrimination layer
        model.add(Flatten())
        # now model.output_shape == (None, 640)
        # add the minibatch discrimination layer
        model.add(MinibatchDiscrimination(5, 3))
        # now model.output_shape = (None, 645)
    ```
    # Arguments
        nb_kernels: Number of discrimination kernels to use
            (dimensionality concatenated to output).
        kernel_dim: The dimensionality of the space where closeness of samples
            is calculated.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        weights: list of numpy arrays to set as initial weights.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        input_dim: Number of channels/dimensions in the input.
            Either this argument or the keyword argument `input_shape`must be
            provided when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(samples, input_dim + nb_kernels)`.
    # References
        - [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)
    """

    def __init__(self, nb_kernels, kernel_dim, init='glorot_uniform', weights=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None, input_dim=None, **kwargs):
        self.init = initializers.get(init)
        self.nb_kernels = nb_kernels
        self.kernel_dim = kernel_dim
        self.input_dim = input_dim

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)

        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2

        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        self.W = self.add_weight(shape=(self.nb_kernels, input_dim, self.kernel_dim),
            initializer=self.init,
            name='kernel',
            regularizer=self.W_regularizer,
            trainable=True,
            constraint=self.W_constraint)

        # Set built to true.
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, mask=None):
        activation = K.reshape(K.dot(x, self.W), (-1, self.nb_kernels, self.kernel_dim))
        diffs = K.expand_dims(activation, 3) - K.expand_dims(K.permute_dimensions(activation, [1, 2, 0]), 0)
        abs_diffs = K.sum(K.abs(diffs), axis=2)
        minibatch_features = K.sum(K.exp(-abs_diffs), axis=2)
        return K.concatenate([x, minibatch_features], 1)

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], input_shape[1]+self.nb_kernels

    def get_config(self):
        config = {'nb_kernels': self.nb_kernels,
                  'kernel_dim': self.kernel_dim,
                  'init': self.init.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'input_dim': self.input_dim}
        base_config = super(MinibatchDiscrimination, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



################# Define Normalization Function #####################

def scoreNormalization(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result
    
################# Define Discriminator Model #####################
 
def defineDiscriminator(input_dim=(128,256,1), n_classes=3):
    in_image = Input(shape=input_dim)
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(in_image)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = BatchNormalization()(fe)
    
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = Dropout(0.7)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = BatchNormalization()(fe)
    
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = Dropout(0.7)(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = BatchNormalization()(fe)
    
    fe = Flatten()(fe)
    fe = Dropout(0.7)(fe)
    fe = MinibatchDiscrimination(100, 5)(fe) 
    
    fe = Dropout(0.7)(fe)
    fe = Dense(n_classes)(fe)
    
    
    ########----------------- Supervised Model
    
    c_out_layer = Activation('softmax')(fe)
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])
    
    ########----------------- Unsupervised Model
    
    d_out_layer = Lambda(scoreNormalization)(fe)
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
    
    
    return d_model, c_model
    
    
########------------------ Define Discriminator and Load Model
    
    
d_model, loaded_model = defineDiscriminator()
loaded_model.summary()
loaded_model.load_weights('learnedModels/Classifier_fold3_labeled_sam500.h5',by_name=True);


#########------------------ load Data

def preProcess(X):
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

train_datagen = ImageDataGenerator(preprocessing_function=preProcess)

    
valid_generator_scale1 = train_datagen.flow_from_directory(
    directory="data/val",
    target_size=(128, 256),
    batch_size=32,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=False,
    seed=42
    
)   
  
   
predictions = loaded_model.predict_generator(valid_generator_scale1)

y_pred = np.argmax(predictions, axis=1)

y_test= valid_generator_scale1.classes

############--------------- Get Confusion Matrix

cnf_matrix = confusion_matrix(y_test, y_pred)
cnf_matrix_T=np.transpose(cnf_matrix)
############ ------------------- Get Performance Measures

n_class=3

TP=np.zeros(n_class)
FN=np.zeros(n_class)
FP=np.zeros(n_class)
TN=np.zeros(n_class)

for i in range(n_class):
    TP[i]=cnf_matrix[i,i]
    FN[i]=np.sum(cnf_matrix[i])-cnf_matrix[i,i]
    FP[i]=np.sum(cnf_matrix_T[i])-cnf_matrix[i,i]
    TN[i]=np.sum(cnf_matrix)-TP[i]-FP[i]-FN[i]
    
P=TP+FN
N=FP+TN

classwise_sensitivity=np.true_divide(TP,P)
classwise_specificity=np.true_divide(TN,N)
classwise_accuracy=np.true_divide((TP+TN), (P+N))
    
OS=np.mean(classwise_sensitivity)

OSp=np.mean(classwise_specificity)

OA=np.sum(np.true_divide(TP,(P+N)))

########### -------------- compute Kappa

Px=np.sum(P)
TPx=np.sum(TP)
FPx=np.sum(FP)
TNx=np.sum(TN)
FNx=np.sum(FN)
Nx=np.sum(N)

#####-------------- Kappa Computation

pox=OA
pex=((Px*(TPx+FPx))+(Nx*(FNx+TNx)))/(math.pow((TPx+TNx+FPx+FNx),2))

kappa_overall=[np.true_divide(( pox-pex ), ( 1-pex )),np.true_divide(( pex-pox ), ( 1-pox ))]

kappa=np.max(kappa_overall)

#######--------------------- Print all scores

print('classwise_sen',classwise_sensitivity*100)

print('classwise_spec',classwise_specificity*100)

print('classwise_acc',classwise_accuracy*100)

print('overall_sen',OS*100)

print('overall_spec',OSp*100)

print('overall_acc',OA*100)

print('Kappa',kappa)

