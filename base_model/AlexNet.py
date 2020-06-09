import tensorflow as tf
from tensorflow import keras

def AlexNet(input_shape = [224, 224, 1] ,classes_num = 3,dropout = 0.5 ):
    # Define the input layer
    inputs = keras.Input(shape = input_shape)

    # Define the converlutional layer 1
    conv1 = keras.layers.Conv2D(filters= 96, kernel_size= [11, 11], strides= [4, 4], activation= keras.activations.relu, use_bias= True, padding= 'valid')(inputs)

    # Define the standardization layer 1
    stand1 = keras.layers.BatchNormalization(axis= 1)(conv1)

    # Define the pooling layer 1
    pooling1 = keras.layers.MaxPooling2D(pool_size= [3, 3], strides= [2, 2], padding= 'valid')(stand1)

    # Define the converlutional layer 2
    conv2 = keras.layers.Conv2D(filters= 256, kernel_size= [5, 5], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'valid')(pooling1)

    # Define the standardization layer 2
    stand2 = keras.layers.BatchNormalization(axis= 1)(conv2)

    # Defien the pooling layer 2
    pooling2 = keras.layers.MaxPooling2D(pool_size= [3, 3], strides= [2, 2], padding= 'valid')(stand2)

    # Define the converlutional layer 3
    conv3 = keras.layers.Conv2D(filters= 384, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'valid')(pooling2)

    # Define the converlutional layer 4
    conv4 = keras.layers.Conv2D(filters= 384, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'valid')(conv3)

    # Define the converlutional layer 5
    conv5 = keras.layers.Conv2D(filters= 256, kernel_size= [3, 3], strides= [1, 1], activation= keras.activations.relu, use_bias= True, padding= 'valid')(conv4)

    # Defien the pooling layer 3
    pooling3 = keras.layers.MaxPooling2D(pool_size= [3, 3], strides= [2, 2], padding= 'valid')(conv5)

    # Define the fully connected layer
    flatten = keras.layers.Flatten()(pooling3)

    fc1 = keras.layers.Dense(4096, activation= keras.activations.relu, use_bias= True)(flatten)
    drop1 = keras.layers.Dropout(dropout)(fc1)

    fc2 = keras.layers.Dense(4096, activation= keras.activations.relu, use_bias= True)(drop1)
    drop2 = keras.layers.Dropout(dropout)(fc2)

    fc3 = keras.layers.Dense(classes_num, activation= keras.activations.softmax, use_bias= True)(drop2)

    # 基于Model方法构建模型
    model = keras.Model(inputs= inputs, outputs = fc3)

    return model

if __name__ == "__main__":
    model = AlexNet()
    model.summary()
    pass