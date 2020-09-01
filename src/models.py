import keras 
import tensorflow as tf
from keras.optimizers import  SGD, Adam
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Dropout, GaussianNoise, BatchNormalization, Input, concatenate, Activation
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.regularizers import l1, l2, l1_l2
keras.backend.clear_session()

def build_lstm_model(input_shape,output_shape):
    model = Sequential()
    model.add(LSTM(256, return_sequences=True,input_shape=input_shape))
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.1))
    model.add(LSTM(256, return_sequences=True))
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.1))
    model.add(LSTM(256, kernel_regularizer=l1_l2(0.001,0.001)))
    #model.add(BatchNormalization())
    #model.add(Activation('tanh'))
    #model.add(Dropout(0.1))
    model.add(Dense(output_shape,activation='linear'))
    return model

def build_complex_model(injector_shape,booster_shape):
    # Define the tensors for the two input images
    injector_input = Input(injector_shape)
    booster_input  = Input(booster_shape)
    
    ## 
    injector_model = build_lstm_model(input_shape=injector_shape)
    
    ##
    booster_model = build_lstm_model(input_shape=booster_shape)

    ##
    injector_output = injector_model(injector_input)
    booster_output = booster_model(booster_input)
    
    ##
    complex_output = concatenate([booster_output,injector_output])
    
    ##
    complext_net = Model(inputs=[injector_input,booster_input],outputs=complex_output)

    return complext_net

def train_lstm_model(in_shape,out_shape,x,y,epochs=5,batch_size=256):
    ## Make sure that the session is cleared
    keras.backend.clear_session()
    ## Start training
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    with strategy.scope():
        model = build_lstm_model(input_shape=in_shape,output_shape=out_shape)
        opt = Adam(lr=1e-2)
        #opt = SGD(lr=1e-2,nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        model.summary()
        from keras.callbacks import ReduceLROnPlateau, EarlyStopping
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                      factor=0.85, patience=5, min_lr=1e-6,verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss',
                                       min_delta=0, patience=10, verbose=1, mode='auto',
                                       baseline=None, restore_best_weights=False)

        history = model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2,
                             callbacks=[reduce_lr,early_stopping], verbose=2, shuffle=True)
        return history, model
