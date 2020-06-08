# build model
from tensorflow.python import keras
from tensorflow.keras.layers import Conv2D as Conv2D
from tensorflow.keras.layers import BatchNormalization as BatchNormalization
from tensorflow.keras.layers import Activation as Activation
from tensorflow.keras.layers import MaxPooling2D as MaxPooling2D
from tensorflow.keras.layers import Dropout as Dropout
from tensorflow.keras.layers import Dense as Dense
from tensorflow.keras.layers import Flatten as Flatten

def VGG19(input_shape = (224,224,3),classes=1000,weight_decay=1e-5,dropout=0.5):
    model = keras.models.Sequential()
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', input_shape=input_shape,
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block1_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), 
            name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block2_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), 
            name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block3_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block3_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), 
            strides=(2, 2), 
            name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block4_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block4_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block5_conv3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same', 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='block5_conv4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), 
            name='block5_pool'))

    # model modification for cifar-10
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, use_bias = True, 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='fc_1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4096, 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', 
            name='fc_2'))  
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(dropout))      
    model.add(Dense(10, 
            kernel_regularizer=keras.regularizers.l2(weight_decay), 
            kernel_initializer='he_normal', name='predictions'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))
    return model

if __name__ == "__main__":
    model = VGG19()
    model.summary()
    pass