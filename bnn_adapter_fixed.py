# Archivo: bnn_adapter_fixed.py
import tensorflow as tf
import numpy as np

class binary_activation(tf.keras.layers.Layer):
    """Implementación de capa de activación binaria para TensorFlow 2.x"""
    
    def __init__(self, **kwargs):
        super(binary_activation, self).__init__(**kwargs)
    
    def call(self, x):
        # Función sign para binarización (-1, +1)
        return tf.sign(x)
    
    def get_config(self):
        config = super(binary_activation, self).get_config()
        return config
        
    def compute_output_shape(self, input_shape):
        return input_shape

class binary_conv(tf.keras.layers.Layer):
    """Capa convolucional binaria (1-bit)"""
    
    def __init__(self, nfilters, ch_in, k=3, padding='same', strides=(1,1), first_layer=False, **kwargs):
        super(binary_conv, self).__init__(**kwargs)
        self.nfilters = nfilters
        self.ch_in = ch_in
        self.k = k
        self.padding = padding
        self.strides = strides
        self.first_layer = first_layer
    
    def build(self, input_shape):
        # Inicialización de pesos
        stdv = 1/np.sqrt(self.k*self.k*self.ch_in)
        w_init = tf.random.normal(
            shape=(self.k, self.k, self.ch_in, self.nfilters),
            mean=0.0, stddev=stdv
        )
        self.w = self.add_weight(
            name='kernel',
            shape=(self.k, self.k, self.ch_in, self.nfilters),
            initializer=tf.keras.initializers.Constant(value=w_init),
            trainable=True
        )
        super(binary_conv, self).build(input_shape)
    
    def call(self, x):
        # Binarizar pesos para forward pass
        binary_w = tf.sign(self.w)
        
        # Usar pesos binarios para la convolución
        out = tf.nn.conv2d(
            x, binary_w, 
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding.upper()
        )
        return out
    
    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        
        if self.padding.upper() == 'SAME':
            out_height = height // self.strides[0]
            out_width = width // self.strides[1]
        else:
            out_height = (height - self.k + 1) // self.strides[0]
            out_width = (width - self.k + 1) // self.strides[1]
            
        return (batch_size, out_height, out_width, self.nfilters)
    
    def get_config(self):
        config = super(binary_conv, self).get_config()
        config.update({
            'nfilters': self.nfilters,
            'ch_in': self.ch_in,
            'k': self.k,
            'padding': self.padding,
            'strides': self.strides,
            'first_layer': self.first_layer
        })
        return config

class binary_dense(tf.keras.layers.Layer):
    """Capa densa binaria (1-bit)"""
    
    def __init__(self, n_in, n_out, first_layer=False, **kwargs):
        super(binary_dense, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out
        self.first_layer = first_layer
    
    def build(self, input_shape):
        # Inicialización de pesos
        stdv = 1/np.sqrt(self.n_in)
        w_init = tf.random.normal(
            shape=(self.n_in, self.n_out),
            mean=0.0, stddev=stdv
        )
        self.w = self.add_weight(
            name='kernel',
            shape=(self.n_in, self.n_out),
            initializer=tf.keras.initializers.Constant(value=w_init),
            trainable=True
        )
        super(binary_dense, self).build(input_shape)
    
    def call(self, x):
        # Binarizar pesos para forward pass
        binary_w = tf.sign(self.w)
        
        # Usar pesos binarios para la multiplicación de matrices
        out = tf.matmul(x, binary_w)
        return out
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_out)
    
    def get_config(self):
        config = super(binary_dense, self).get_config()
        config.update({
            'n_in': self.n_in,
            'n_out': self.n_out,
            'first_layer': self.first_layer
        })
        return config

class my_flat(tf.keras.layers.Layer):
    """Capa de aplanamiento"""
    
    def __init__(self, **kwargs):
        super(my_flat, self).__init__(**kwargs)
    
    def call(self, x):
        return tf.reshape(x, [tf.shape(x)[0], -1])
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], np.prod(input_shape[1:]))
    
    def get_config(self):
        config = super(my_flat, self).get_config()
        return config

class l1_batch_norm_mod_conv(tf.keras.layers.Layer):
    """Normalización por lotes L1 modificada para capas convolucionales"""
    
    def __init__(self, batch_size, width_in, ch_in, momentum=0.9, **kwargs):
        super(l1_batch_norm_mod_conv, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.width_in = width_in
        self.ch_in = ch_in
        self.momentum = momentum
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.ch_in,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.ch_in,),
            initializer='zeros',
            trainable=True
        )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.ch_in,),
            initializer='zeros',
            trainable=False
        )
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=(self.ch_in,),
            initializer='ones',
            trainable=False
        )
        super(l1_batch_norm_mod_conv, self).build(input_shape)
        
    def call(self, x, training=None):
        # En versiones recientes de TF, ya no se usa learning_phase()
        # En su lugar, usamos el parámetro training
        if training is None:
            training = False  # Valor predeterminado seguro
        
        if training:
            # Usar estadísticas del batch actual (L1 norm)
            mu = tf.reduce_mean(x, axis=[0, 1, 2])
            var = tf.reduce_mean(tf.abs(x - mu), axis=[0, 1, 2])
            
            # Actualizar promedios móviles
            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1 - self.momentum) * mu
            )
            self.moving_var.assign(
                self.momentum * self.moving_var + (1 - self.momentum) * var
            )
        else:
            # Usar estadísticas almacenadas
            mu = self.moving_mean
            var = self.moving_var
        
        # Normalizar y aplicar gamma y beta
        x_norm = (x - mu) / (var + 1e-4)
        output = x_norm * self.gamma + self.beta
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(l1_batch_norm_mod_conv, self).get_config()
        config.update({
            'batch_size': self.batch_size,
            'width_in': self.width_in,
            'ch_in': self.ch_in,
            'momentum': self.momentum
        })
        return config

class l1_batch_norm_mod_dense(tf.keras.layers.Layer):
    """Normalización por lotes L1 modificada para capas densas"""
    
    def __init__(self, batch_size, ch_in, momentum=0.9, **kwargs):
        super(l1_batch_norm_mod_dense, self).__init__(**kwargs)
        self.batch_size = batch_size
        self.ch_in = ch_in
        self.momentum = momentum
        
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(self.ch_in,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(self.ch_in,),
            initializer='zeros',
            trainable=True
        )
        self.moving_mean = self.add_weight(
            name='moving_mean',
            shape=(self.ch_in,),
            initializer='zeros',
            trainable=False
        )
        self.moving_var = self.add_weight(
            name='moving_var',
            shape=(self.ch_in,),
            initializer='ones',
            trainable=False
        )
        super(l1_batch_norm_mod_dense, self).build(input_shape)
        
    def call(self, x, training=None):
        # En versiones recientes de TF, usamos el parámetro training directamente
        if training is None:
            training = False  # Valor predeterminado seguro
        
        if training:
            # Usar estadísticas del batch actual (L1 norm)
            mu = tf.reduce_mean(x, axis=0)
            var = tf.reduce_mean(tf.abs(x - mu), axis=0)
            
            # Actualizar promedios móviles
            self.moving_mean.assign(
                self.momentum * self.moving_mean + (1 - self.momentum) * mu
            )
            self.moving_var.assign(
                self.momentum * self.moving_var + (1 - self.momentum) * var
            )
        else:
            # Usar estadísticas almacenadas
            mu = self.moving_mean
            var = self.moving_var
        
        # Normalizar y aplicar gamma y beta
        x_norm = (x - mu) / (var + 1e-4)
        output = x_norm * self.gamma + self.beta
        
        return output
    
    def compute_output_shape(self, input_shape):
        return input_shape
    
    def get_config(self):
        config = super(l1_batch_norm_mod_dense, self).get_config()
        config.update({
            'batch_size': self.batch_size,
            'ch_in': self.ch_in,
            'momentum': self.momentum
        })
        return config