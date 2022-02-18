import keras
from keras.models import Model
from keras.layers import Dense , LSTM , Dropout , Input , concatenate
from keras.callbacks import ReduceLROnPlateau , EarlyStopping
from keras.optimizer_v2.adam import Adam
from keras.regularizers import l1_l2

keras.backend.clear_session ()


def build_lstm_model ( input_shape , output_shape = None ) :
    inputs = keras.Input ( shape = input_shape )
    x = LSTM ( 256 , return_sequences = True , input_shape = input_shape ) ( inputs , training = True )
    x = Dropout ( 0.1 , seed = 0 ) ( x , training = True )
    x = LSTM ( 256 , return_sequences = True ) ( x , training = True )
    x = LSTM ( 256 , kernel_regularizer = l1_l2 ( 0.001 , 0.001 ) ) ( x , training = True )
    outputs = Dense ( output_shape , activation = 'linear' ) ( x )
    model = keras.Model ( inputs = inputs , outputs = outputs )
    return model


def build_complex_model ( injector_shape , booster_shape ) :
    # Define the tensors for the two input images
    injector_input = Input ( injector_shape )
    booster_input = Input ( booster_shape )
    ##
    injector_model = build_lstm_model ( input_shape = injector_shape )
    booster_model = build_lstm_model ( input_shape = booster_shape )
    ##
    injector_output = injector_model ( injector_input )
    booster_output = booster_model ( booster_input )
    complex_output = concatenate ( [ booster_output , injector_output ] )

    complext_net = Model ( inputs = [ injector_input , booster_input ] , outputs = complex_output )

    return complext_net


def train_lstm_model ( in_shape ,
                       out_shape ,
                       x ,
                       y ,
                       epochs:int = 5 ,
                       lr: float = 1e-2,
                       batch_size:int = 256,
                       clipnorm: float = 1.0,
                       clipvalue: float = 0.5,
                       training_loss: str = 'mean_squared_error') :
    ## Make sure that the session is cleared
    keras.backend.clear_session ()
    ## Start training
    model = build_lstm_model ( input_shape = in_shape ,
                               output_shape = out_shape )
    opt = Adam ( lr = lr,
                 clipnorm = clipnorm ,
                 clipvalue = clipvalue )
    # opt = SGD(lr=1e-2,nesterov=True)
    model.compile ( loss = training_loss ,
                    optimizer = opt )
    model.summary ()
    reduce_lr = ReduceLROnPlateau ( monitor = 'val_loss' ,
                                    factor = 0.85 , patience = 5 , min_lr = 1e-6 , verbose = 1 )
    early_stopping = EarlyStopping ( monitor = 'val_loss' ,
                                     min_delta = 0 , patience = 10 , verbose = 1 , mode = 'auto' ,
                                     baseline = None , restore_best_weights = False )

    history = model.fit ( x , y , epochs = epochs , batch_size = batch_size , validation_split = 0.2 ,
                          callbacks = [ reduce_lr , early_stopping ] , verbose = 2 , shuffle = True )
    return history , model