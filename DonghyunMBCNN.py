from tensorflow.keras import utils as np_utils
from tensorflow.keras.backend import image_data_format, set_image_data_format
from tensorflow.keras.layers import Input, Dense, Flatten, Conv3D, MaxPooling3D, Activation, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model, Model
import tensorflow as tf

print(image_data_format())
#set_image_data_format('channels_first')
print(image_data_format())


def MultiBranchCNN(Timesteps, Width, Height, Num_classes):
    Timesteps = Timesteps
    Width = Width
    Height = Height
    Num_classes = Num_classes
    initializer = tf.keras.initializers.GlorotNormal()

    first_input      = Input(shape = (Height, Width, Timesteps, 1))
    Conv1            = Conv3D(filters=16, kernel_size=(3, 3, 5), strides=(2, 2, 4), kernel_initializer=initializer, padding='same', name='Conv1')(first_input)
    Conv1_BN         = BatchNormalization(name='Conv1_BN')(Conv1)
    Conv1_activation = Activation('elu', name='Conv1_activation')(Conv1_BN)
    ##################################################################

    ##### SRF Network
    SRF_input        = Input(shape = (Conv1.shape))

    SRF_Conv1        = Conv3D(32, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer=initializer, name='SRF_Conv1')(Conv1_activation)
    SRF_BN_1         = BatchNormalization(name='SRF_BN_1')(SRF_Conv1)
    SRF_activation_1 = Activation('elu', name='SRF_activation_1')(SRF_BN_1)

    SRF_Conv2        = Conv3D(64, (2, 2, 1), strides=(2, 2, 1), padding='same', kernel_initializer=initializer, name='SRF_Conv2')(SRF_activation_1)
    SRF_BN_2         = BatchNormalization(name='SRF_BN_2')(SRF_Conv2)
    SRF_activation_2 = Activation('elu', name='SRF_activation_2')(SRF_BN_2)

    SRF_flatten      = Flatten(name='SRF_flatten')(SRF_activation_1)
    
    SRF_dense_1      = Dense(32, name='SRF_dense_1', kernel_initializer=initializer)(SRF_flatten)
    SRF_BN_3         = BatchNormalization(name='SRF_BN_3')(SRF_dense_1)
    SRF_relu_1       = Activation('relu', name='SRF_relu_1')(SRF_BN_3)   

    SRF_dense_2      = Dense(32, name='SRF_dense_2', kernel_initializer=initializer)(SRF_relu_1)
    SRF_BN_4         = BatchNormalization(name='SRF_BN_4')(SRF_dense_2)
    SRF_relu_2       = Activation('relu', name='SRF_relu_2')(SRF_BN_4)   

    SRF_dense_3      = Dense(Num_classes, name='SRF_dense_3', kernel_initializer=initializer)(SRF_relu_2)
    SRF_softmax      = Activation('softmax', name='SRF_softmax')(SRF_dense_3)

    ##### MRF Network
    MRF_Conv1        = Conv3D(32, (2, 2, 3), strides=(2, 2, 2), padding='same', kernel_initializer=initializer, name='MRF_Conv1')(Conv1_activation)
    MRF_BN_1         = BatchNormalization(name='MRF_BN_1')(MRF_Conv1)
    MRF_activation_1 = Activation('elu', name='MRF_activation_1')(MRF_BN_1)

    MRF_Conv2        = Conv3D(64, (2, 2, 3), strides=(2, 2, 2), padding='same', kernel_initializer=initializer, name='MRF_Conv2')(MRF_activation_1)
    MRF_BN_2         = BatchNormalization(name='MRF_BN_2')(MRF_Conv2)
    MRF_activation_2 = Activation('elu', name='MRF_activation_2')(MRF_BN_2)

    MRF_flatten      = Flatten(name='MRF_flatten')(MRF_activation_2)
    
    MRF_dense_1      = Dense(32, name='MRF_dense_1', kernel_initializer=initializer)(MRF_flatten)
    MRF_BN_3         = BatchNormalization(name='MRF_BN_3')(MRF_dense_1)
    MRF_relu_1       = Activation('relu', name='MRF_relu_1')(MRF_BN_3)   

    MRF_dense_2      = Dense(32, name='MRF_dense_2', kernel_initializer=initializer)(MRF_relu_1)
    MRF_BN_4         = BatchNormalization(name='MRF_BN_4')(MRF_dense_2)
    MRF_relu_2       = Activation('relu', name='MRF_relu_2')(MRF_BN_4)

    MRF_dense_3      = Dense(Num_classes, name='MRF_dense_3', kernel_initializer=initializer)(MRF_relu_2)
    MRF_softmax      = Activation('softmax', name='MRF_softmax')(MRF_dense_3)

    ##### LRF Network
    LRF_Conv1        = Conv3D(32, (2, 2, 5), strides=(2, 2, 4), padding='same', kernel_initializer=initializer, name='LRF_Conv1')(Conv1_activation)
    LRF_BN_1         = BatchNormalization(name='LRF_BN_1')(LRF_Conv1)
    LRF_activation_1 = Activation('elu', name='LRF_activation_1')(LRF_BN_1)

    LRF_Conv2        = Conv3D(64, (2, 2, 5), strides=(2, 2, 4), padding='same', kernel_initializer=initializer, name='LRF_Conv2')(LRF_activation_1)
    LRF_BN_2         = BatchNormalization(name='LRF_BN_2')(LRF_Conv2)
    LRF_activation_2 = Activation('elu', name='LRF_activation_2')(LRF_BN_2)

    LRF_flatten      = Flatten(name='LRF_flatten')(LRF_activation_2)
    
    LRF_dense_1      = Dense(32, name='LRF_dense_1', kernel_initializer=initializer)(LRF_flatten)
    LRF_BN_3         = BatchNormalization(name='LRF_BN_3')(LRF_dense_1)
    LRF_relu_1       = Activation('relu', name='LRF_relu_1')(LRF_BN_3)

    LRF_dense_2      = Dense(32, name='LRF_dense_2', kernel_initializer=initializer)(LRF_relu_1)
    LRF_BN_4         = BatchNormalization(name='LRF_BN_4')(LRF_dense_2)
    LRF_relu_2       = Activation('relu', name='LRF_relu_2')(LRF_BN_4)

    LRF_dense_3      = Dense(Num_classes, name='LRF_dense_3', kernel_initializer=initializer)(LRF_relu_2)
    LRF_softmax      = Activation('softmax', name='LRF_softmax')(LRF_dense_3)

    ##### Combining Network
    final_softmax    = Activation('softmax', name='final_softmax')(SRF_softmax + MRF_softmax + LRF_softmax)

    model = Model(inputs=first_input, outputs=final_softmax)
    return model
