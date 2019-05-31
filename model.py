from keras.models import Sequential
from keras.models import *
from keras.layers import *
from keras.layers.embeddings import Embedding


def make_model():
    """Builds the LSTM, with 2 layers and reasonable drop out, 
    and binary cross entropy as the loss function
    """
    model = Sequential()
    model.add(Embedding(50000, 100, input_length=11))
    #layer 1
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    #layer 2
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))

    #Defines loss function and prediction metric
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model
