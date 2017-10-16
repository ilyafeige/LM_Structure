import numpy as np
import time
import random


def build_sparsity(sparseness, omega, L):
    if sparseness == 1:
        return None
    else:
        L_run = np.int((2*np.pi)/omega)
        chosen_n = np.random.choice(L-L_run, 
                                    size=int(sparseness*L/L_run), 
                                    replace=False)
        inds = chosen_n[:, None] + range(L_run)
        sparsity = np.zeros(L)
        sparsity[inds] = 1
        return sparsity


def data_function(amp, omega, L, normalisation, 
                  noise_amp=None, sparsity=None, fun=np.sin):
    
    underlying = amp*fun(omega*np.arange(L))/normalisation

    if noise_amp is None:
        data = underlying
    else:
        noise = amp*noise_amp*np.random.randn(L)/normalisation
        data = underlying + noise

    if sparsity is not None:
        data = sparsity*(data)

    return data

def sparse_sine(x, thresh=0.9):
    return np.sign(np.sin(x))*(np.abs(np.sin(x)) > thresh)
    

def build_data(L_data=10000000, T_long=90.0, T_med=20.0, T_short=5.0, 
               sparseness_long=1, sparseness_med=1, sparseness_short=1, 
               noise_amp=None, fun=np.sin, verbose=True):
    
    start = time.time()
    
    omega_short = 2*np.pi/T_short
    omega_med = 2*np.pi/T_med
    omega_long = 2*np.pi/T_long
    
    amp_short = 1.0
    amp_med = 1.0
    amp_long = 1.0
    normalisation = 3.0
        
    sparsity_short = build_sparsity(sparseness_short, omega=omega_short, L=L_data)
    sparsity_med = build_sparsity(sparseness_med, omega=omega_med, L=L_data)
    sparsity_long = build_sparsity(sparseness_long, omega=omega_long, L=L_data)
        
    x_short = data_function(amp_short, omega_short, L_data, normalisation,
                            noise_amp=noise_amp, sparsity=sparsity_short, fun=fun)
    x_med = data_function(amp_med, omega_med, L_data, normalisation,
                          noise_amp=noise_amp, sparsity=sparsity_med, fun=fun)
    x_long = data_function(amp_long, omega_long, L_data, normalisation,
                           noise_amp=noise_amp, sparsity=sparsity_long, fun=fun)
    
    data_used = x_short + x_med + x_long

    if verbose:
        print("Built dataset in {0:.2f} seconds".format(time.time()-start))

    return data_used

    
def prep_data_batched(data, batch_size, sent_len, n_batches, Train_Test_split=0.8):

    l_data = batch_size*sent_len*n_batches

    if len(data) > l_data:
        data = data[:l_data]

    sent_array = data.reshape(np.int(l_data/sent_len), sent_len)

    X_batches = []
    for n in range(n_batches):
        mask = np.random.randint(0, n_batches, batch_size)
        X_batches.append(sent_array[mask])

    mask = np.random.rand(len(X_batches)) < Train_Test_split
    batch_list_train = np.arange(len(X_batches))[mask]
    batch_list_test = np.arange(len(X_batches))[~mask]

    X_train=[]; X_test=[]; Y_train=[]; Y_test=[]
    for batch_n in batch_list_train:
    #    array = np.vstack(data_array[batch])
        array = X_batches[batch_n]
        X_train.append(np.delete(array, (-1), axis=1))
        Y_train.append(np.hstack(np.delete(array, (0), axis=1)))

    for batch_n in batch_list_test:
    #    array = np.vstack(data_array[batch])
        array = X_batches[batch_n]    
        X_test.append(np.delete(array, (-1), axis=1))
        Y_test.append(np.hstack(np.delete(array, (0), axis=1)))

    return np.asarray(X_train), np.asarray(X_test), np.asarray(Y_train), np.asarray(Y_test)    
    
    
def prep_data_unbatched(data, sent_len, Train_Test_split=0.8):

    threshold = np.int(len(data)*Train_Test_split)
    train_array = data[:threshold].reshape(np.int(threshold/sent_len), sent_len)
    test_array = data[threshold:].reshape(np.int((len(data)-threshold)/sent_len), sent_len)

    X_train = np.delete(train_array, (-1), axis=1)
    Y_train = np.delete(train_array, (0), axis=1)
    X_test = np.delete(test_array, (-1), axis=1)
    Y_test = np.delete(test_array, (0), axis=1)

    return X_train, X_test, Y_train, Y_test    