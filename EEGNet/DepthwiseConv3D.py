import sys

import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.backend import _preprocess_conv3d_input, _preprocess_padding
from tensorflow import keras

class DepthwiseConv3D(keras.layers.Conv3D):
	def __init__(self, kernel_size, strides=(1, 1, 1), padding='valid', depth_multiplier=1, data_format=None, dilation_rate=(1, 1, 1), activation=None, use_bias=True, depthwise_initializer='glorot_uniform', bias_initializer='zeros', depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None, bias_constraint=None, **kwargs):
		super(DepthwiseConv3D, self).__init__(filters=None, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format, dilation_rate=dilation_rate, activation=activation, use_bias=use_bias, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, bias_constraint=bias_constraint, **kwargs)
		self.depth_multiplier = depth_multiplier
		self.depthwise_initializer = keras.initializers.get(depthwise_initializer)
		self.depthwise_regularizer = keras.regularizers.get(depthwise_regularizer)
		self.depthwise_constraint = keras.constraints.get(depthwise_constraint)
		self.bias_initializer = keras.initializers.get(bias_initializer)
		self.padding = _preprocess_padding(self.padding)
		self.strides = (1,) + self.strides + (1,)

	def build(self, input_shape):
		if len(input_shape) < 5:
			raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. Received input shape:', str(input_shape))
		input_shape = tensor_shape.TensorShape(input_shape)
		channel_axis = self._get_channel_axis()
		if input_shape.dims[channel_axis].value is None:
			raise ValueError('The channel dimension of the inputs to `DepthwiseConv3D` should be defined. Found `None`.')
		self.input_dim = int(input_shape[channel_axis])
		depthwise_kernel_shape = (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2], self.input_dim, self.depth_multiplier)
		self.depthwise_kernel = self.add_weight(shape=depthwise_kernel_shape, initializer=self.depthwise_initializer, name='depthwise_kernel', regularizer=self.depthwise_regularizer, constraint=self.depthwise_constraint)
	
		if self.use_bias:
			self.bias = self.add_weight(shape=(self.input_dim * self.depth_multiplier), initializer=self.bias_initializer, name='bias', regularizer=self.bias_regularizer, constraint=self.bias_constraint)
		
		else:
			self.bias = None
	
		self.input_spec = input_spec.InputSpec(ndim=5, axes={channel_axis: self.input_dim})
		self.built = True
	
	def call(self, inputs):
		inputs, tf_data_format = _preprocess_conv3d_input(inputs, self.data_format)
	
		if self.data_format == 'channels_last':
			dilation = (1,) + self.dilation_rate + (1,)
		else:
			dilation = self.dilation_rate + (1,) + (1,)
		
		if self.data_format == 'channels_first':
			outputs = tf.concat([tf.nn.conv3d(inputs[:, i:i+self.input_dim, :, :, :], self.depthwise_kernel[:, :, :, i:i+self.input_dim, :], strides = self.strides, padding=self.padding, dilations=dilation, data_format=tf_data_format) for i in range(0, self.input_dim)], axis=1)

		else:
			outputs = tf.concat([tf.nn.conv3d(inputs[:, :, :, :, i:i+self.input_dim], self.depthwise_kernel[:, :, :, i:i+self.input_dim, :], strides=self.strides, padding=self.padding, dilations=dilation, data_format=tf_data_format) for i in range(0, self.input_dim)], axis=-1)
		
		if self.bias is not None:
			outputs = keras.backend.bias_add(outputs, self.bias, data_format=self.data_format)

		if self.activation is not None:
			return self.activation(outputs)

		return outputs


	def get_config(self):
		config = super(DepthwiseConv3D, self).get_config()
		config.pop('filters')
		config.pop('kernel_initializer')
		config.pop('kernel_regularizer')
		config.pop('kernel_constraint')
		config['depth_multiplier'] = self.depth_multiplier
		config['depthwise_initializer'] = keras.initializers.serialize(self.depthwise_initializer)
		config['depthwise_regularizer'] = keras.regularizers.serialize(self.depthwise_regularizer)
		config['depthwise_constraint'] = keras.constraints.serialize(self.depthwise_constraint)
		return config
