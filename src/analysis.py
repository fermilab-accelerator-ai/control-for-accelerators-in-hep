import matplotlib.pyplot as plt

def plot_loss(history,name='loss'):
    plt.figure(figsize=(12,10))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.yscale('log')
    plt.savefig('{}.png'.format(name))
    plt.show()

def plot_test(model,x_test,y_test,nvar=2,name='test',start=0,end=500):
    Y_predict = model.predict(x_test[start:end,:,:])
    print(Y_predict.shape)
    fig, axs = plt.subplots(nvar,figsize=(14,12))
    for v in range (nvar):
        Y_test_var1 = y_test[start:end,v].reshape(-1,1)
        Y_predict_var1 = Y_predict[:,v].reshape(-1,1)
        axs[v].plot(Y_test_var1,label='Data')
        axs[v].plot(Y_predict_var1, label='Prediction')
    plt.savefig('{}.png'.format(name))