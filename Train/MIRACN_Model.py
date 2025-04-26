import tensorflow as tf
from keras import regularizers
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv1D, MaxPooling1D, Add, ZeroPadding1D

def create_model(dropout_rate, neurons, kernel_size, optimizer):
    input_shape = (2192, 1)
    inputs = Input(input_shape)

    conv_1 = Conv1D(neurons, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(inputs)
    maxp_1 = MaxPooling1D(pool_size=2)(conv_1)
    
    conv_2 = Conv1D(neurons * 2, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(maxp_1)
    maxp_2 = MaxPooling1D(pool_size=2)(conv_2)

    conv_3 = Conv1D(neurons * 4, kernel_size=kernel_size, activation='relu', strides=1, padding='same')(maxp_2)
    maxp_3 = MaxPooling1D(pool_size=2)(conv_3)

    conv_4 = Conv1D(neurons * 8, kernel_size=kernel_size, activation='relu', padding='same')(maxp_3)
    maxp_4 = MaxPooling1D(pool_size=2)(conv_4)

    residual_channels = tf.keras.backend.int_shape(inputs)[-1]
    residual_1 = Conv1D(residual_channels * neurons * 8 , kernel_size=1, activation='relu', padding='same')(inputs)
    residual_maxp_1 = MaxPooling1D(pool_size=16)(residual_1)

    skip_connection_1 = Add()([maxp_4, residual_maxp_1])

    flatten = Flatten()(skip_connection_1)

    dense_1 = Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flatten)
    dense_2 = Dense(neurons, activation='relu', kernel_regularizer=regularizers.l2(0.01))(flatten)

    dropout_1 = Dropout(dropout_rate)(dense_1)
    dropout_2 = Dropout(dropout_rate)(dense_2)

    output_1 = Dense(1, activation='sigmoid', name='functionality')(dropout_1)
    output_2 = Dense(7, activation='softmax', name='cell_type')(dropout_2)



    model = Model(inputs=[inputs], outputs=[output_1, output_2])
    losses = {"cell_type": "categorical_crossentropy",
              "functionality": "binary_crossentropy", }
    lossWeights = {"cell_type": 1.0, "functionality": 1.0}

    model.compile(loss=losses, loss_weights=lossWeights, optimizer=optimizer, metrics=['accuracy'])
    return model
