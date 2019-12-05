

from __future__ import print_function, division

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from numpy.random import randn
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose,BatchNormalization
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import Lambda
from keras.layers import Activation
from keras import backend
import numpy as np

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
        

############## Load and Preprocess Input ##########################


from keras.preprocessing.image import ImageDataGenerator


def preProcess(X):
	X = X.astype('float32')
	# scale from [0,255] to [-1,1]
	X = (X - 127.5) / 127.5
	return X

train_datagen = ImageDataGenerator(preprocessing_function=preProcess)


batch_size=32

#####-------------- Load the Labeled Training Data

train_generator_Sup = train_datagen.flow_from_directory(
    directory="data/labeled",
    target_size=(128, 256),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    seed=42
)
#####-------------- Load the Unlabeled Training Data

train_generator_real = train_datagen.flow_from_directory(
    directory="data/unlabeled",
    target_size=(128, 256),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    seed=42
)


#####-------------- Load the Validation Data 

valid_generator = train_datagen.flow_from_directory(
    directory="data/val",
    target_size=(128, 256),
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="sparse",
    shuffle=True,
    seed=42
)



################ Generate Noise Input for the Generator ##################

def generateNoiseSamples(input_dim, samples):
    n_input = randn(input_dim * samples)
    n_input = n_input.reshape(samples, input_dim)
    return n_input
    

################ Generate  Fake Images from the Generator ##############
 
def generateFakeImages(generator, input_dim, samples):
    f_input = generateNoiseSamples(input_dim, samples)
    images = generator.predict(f_input)
    return images
	


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
    
################# Define Generator Model #####################
	
def defineGenerator(input_dim):
    in_lat = Input(shape=(input_dim,))
    n_nodes = 128 * 4 * 8
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((4, 8, 128))(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(4,4), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = BatchNormalization()(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(8,8), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = BatchNormalization()(gen)
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    
    model = Model(in_lat, out_layer)
    return model
	
 
################# Define GAN Model #####################

def defineGAN(g_model, d_model):
    d_model.trainable = False
    gan_output = d_model(g_model.output)
    model = Model(g_model.input, gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model


    
################## Save Learned Model #####################
    
    
def saveModel(c_model):
    c_model.save_weights("Classifier_fold1_labeled_sam500.h5") 
    
    
#################  Function for Train Model ###############################
    
    
def train(g_model, d_model, c_model, gan_model, latent_dim, n_epochs=500, batch_size=64):
    
    half_batch = int(batch_size / 2)
    n_batch_per_ephoch=int(train_generator_real.n/half_batch)
    n_steps=n_batch_per_ephoch*n_epochs
    
    best_acc=0
    
    for i in range(n_steps):
        
        ####----------- update supervised classifier (c)
        
        Xsup_real, ysup_real = train_generator_Sup.next()
        
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        
        d_model.trainable = True
        
        # update unsupervised discriminator (d)
        
        
        X_real, _ = train_generator_real.next()
        
        y_real= np.ones((X_real.shape[0], 1)) - np.random.random_sample((X_real.shape[0], 1))*0.2
        
        y_fake= np.random.random_sample((half_batch, 1))*0.2
        
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        
        X_fake = generateFakeImages(g_model, latent_dim, half_batch)
        
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        
        d_model.trainable = False
        
        X_gan = generateNoiseSamples(latent_dim, batch_size)
        
        y_gan = np.ones((batch_size, 1))- np.random.random_sample((batch_size, 1))*0.2
        
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        
        if i % 25 == 0:
            _,vacc = c_model.evaluate_generator(valid_generator)
            if (vacc*100) > best_acc:
                best_acc=vacc*100
                saveModel(c_model)
                print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f],v[%.0f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss,vacc*100))
            
    

        
##########------------ Size of noise space dimension   
        
inputNoiseDim = 100

##########-------------- Create the discriminator model

d_model, c_model = defineDiscriminator()

##########------------- Create the generator model

g_model = defineGenerator(inputNoiseDim)

gan_model = defineGAN(g_model, d_model)
c_model.summary()


###########------------- Train model

train(g_model, d_model, c_model, gan_model, inputNoiseDim)
 

        
        
        
        
        
        
        
        
    
    	
     
     
     
     
     

    
    
 
