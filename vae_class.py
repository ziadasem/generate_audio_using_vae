# -*- coding: utf-8 -*-


from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization,LeakyReLU, ReLU, Flatten,Dense, Reshape, Conv2DTranspose, Activation, Lambda 
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os 
import pickle

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class VAE :
    #constructor
    def  __init__(self, input_shape, conv_filters, 
                  conv_kernels, conv_strides,
                  latent_space_dim):
        
        self.input_shape = input_shape #for images we have w*h* colors [28,28,1]
        self.conv_filters = conv_filters #list of number of filters for conv layers [2,4,8]
        self.conv_kernels = conv_kernels #kernal sizes for each layer [3,5,3]
        self.conv_strides = conv_strides #strides for each encoder [1,2,2]
        self.latent_space_dim = latent_space_dim #2
        
        self.encoder = None #encoder model
        self.decoder= None
        self.model = None # the total model
        self.reconstruction_loss_weight = 1000000
        
        
        #private values
        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None
        self._build()
        
    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()
        
    def _build_encoder(self) :
        #creating input
        encoder_input = self._add_encoder_input()
        #build conv layers
        conv_layers = self._add_conv_layers(encoder_input)
        #build bottleneck
        bottleneck = self.add_bottleneck(conv_layers)
        self._model_input = encoder_input  
        
        self.encoder = Model(encoder_input, bottleneck, name = "encoder")
    
    def _add_encoder_input(self) :
        #build input layer with specific shape
        return Input(shape = self.input_shape, name = "encoder_input")
    
    def _add_conv_layers(self, encoder_input) :
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x= self._add_conv_layer(layer_index,x)
        return x
    
    def _add_conv_layer(self, layer_index,x ) :
        layer_number = layer_index + 1 
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"enocder_conve_layer_{layer_index}"
        )
        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu{layer_number}")(x)
        x= BatchNormalization(name = f"encoder_bn{layer_number}")(x)
        return x
    
    def add_bottleneck(self, x) :
        """ flatten the conv2d and then pass to dense layer"""
        self._shape_before_bottleneck = K.int_shape(x)[1:] #to be mirrored in decoder
        x = Flatten()(x)

        #the change here is instead of latent space fixed values, we will output two
        # neural network one for MUs and one for Variance
        #  NOT SEQUN
        self.mu = Dense(self.latent_space_dim ,name = "mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name = "log_variance")(x)

        #the output should be one network to be sampled, and we need to sampe from the previous
        # two networks
        # this can be done using lambda layer

        # Lambda is used to transform the input data using an expression or function.
        #   For example, if Lambda with expression lambda x: x ** 2 is applied to a layer,
        #   then its input data will be squared before processing.
        
        #it will sample point from ND and output it
        def sample_point_from_normal_distribution(args):
          mu, log_variance = args
          #sampling random point from normal dist with mu =0 and dev = 1.0 i.e. standard ND
          epsillon = K.random_normal(shape = K.shape(self.mu), mean= 0.0, stddev = 1.0)

          sampled_point  = mu + K.exp(log_variance/2)* epsillon 
          return sampled_point



        encoder_output = Lambda(sample_point_from_normal_distribution,
                   name = "encoder_output")([self.mu, self.log_variance])
        return encoder_output
    
    
    
    def _build_decoder(self):
        #creating input
        decoder_input = self._add_decoder_input() #latent dim 2*2
        dense_layer= self._add_dense_layer(decoder_input) #connect
        reshape_layer = self._add_reshape_layer(dense_layer) #reshape to dense
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)#de conv
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder  = Model(decoder_input ,decoder_output , name = "decoder")
    
    def _add_decoder_input(self):
        return Input(shape= self.latent_space_dim, name= "decoder_input")
    
    def _add_dense_layer(self, decoder_input):
        num_neurons =np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name = "decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        #convert the flat to original shape
        reshape_layer = Reshape(self._shape_before_bottleneck, name = "reshape_layer")(dense_layer)
        return reshape_layer
    
    def _add_conv_transpose_layers(self, x ) :
        #decode
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x
    
    def _add_conv_transpose_layer(self,layer_index, x ) :
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"decodeer_deconv_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x= ReLU(name = f"decodeer_ReLu_layer_{layer_num}")(x)
        x = BatchNormalization(name = f"decodeer_BN_layer_{layer_num}")(x)
        return x
    
    
    def _add_decoder_output(self,conv_transpose_layer ) :
        last_conv_transpose_layer = Conv2DTranspose(
            filters = 1, #set to BW image
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decodeer_deconv_layer_{self._num_conv_layers}"
        )
        conv_transpose_layer = last_conv_transpose_layer(conv_transpose_layer)
        output_layer = Activation("sigmoid", name = "sigmoid")(conv_transpose_layer)
        return output_layer
    
    def _build_autoencoder(self) :
        model_input  = self._model_input 
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input,model_output, name="autoencoder" )
        
    def compile(self, learning_rate = 0.0001):
        #optimizer
        optimizer = Adam(learning_rate= learning_rate)
        self.model.compile(optimizer =optimizer , loss = self._calculate_combianed_loss,
                           metrics= [self._calculate_reconstruction_loss, self._calculate_kl_loss] 
                           ) 
        
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(x_train, x_train, batch_size=batch_size, shuffle = True, epochs =num_epochs  )
    
    def save(self , save_folder="."):
        self._create_folder_if_not_exist(save_folder)
        self._save_paramters(save_folder)
        self._save_weights(save_folder)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations
    
    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = VAE(*parameters)
        weights_path = os.path.join(save_folder, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder
    
    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _calculate_combianed_loss(self, y_target, y_predicted) :
      reconstruction_loss = self._calculate_reconstruction_loss( y_target, y_predicted)
      kl_loss = self._calculate_kl_loss(y_target, y_predicted)
      combined_loss = self.reconstruction_loss_weight * reconstruction_loss+ kl_loss
      return combined_loss

    def _calculate_reconstruction_loss(self, y_target, y_predicted):
      error = y_target - y_predicted
      reconstruction_loss = K.mean(K.square(error), axis=[1, 2, 3])
      return reconstruction_loss

    def _calculate_kl_loss(self, y_target, y_predicted):
        kl_loss = -0.5 * K.sum(1 + self.log_variance - K.square(self.mu) -
                               K.exp(self.log_variance), axis=1)
        return kl_loss

    def _create_folder_if_not_exist(self ,save_folder):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
    def _save_paramters(self, save_folder):
        paramters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "paramters.pkl")
        with open(save_path , "wb") as f:
            pickle.dump(paramters, f)
        
    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)