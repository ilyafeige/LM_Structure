import numpy as np

from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import keras
from keras.layers.core import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.models import Sequential
from keras.models import load_model

import time


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class RNNTS_keras:
    
    def __init__(self, model_type="LSTM", layer_size=128, save_file_append=None, verbose=True):
    
        if save_file_append is not None:
            self.model = load_model("results/Keras_"+model_type+"_"+save_file_append)
            if verbose:
                print("Model loaded from file")
                
        else:
            self.model = Sequential()
            self.model_type = model_type
            
            if model_type == "LSTM":
                self.model.add(LSTM(layer_size, input_shape=(99, 1), return_sequences=True))
            elif model_type == "simple":
                self.model.add(SimpleRNN(layer_size, input_shape=(99, 1), return_sequences=True))
            elif model_type == "complex":
                self.model.add(LSTM(layer_size, input_shape=(99, 1), return_sequences=True))
                self.model.add(LSTM(np.int(layer_size/2), return_sequences=True))
                self.model.add(LSTM(np.int(layer_size/2), return_sequences=True))
            else:
                raise ValueError("Only LSTM, simple, and complex are accepted as model_type's.")
            
        #    model.add(Dropout(0.2))
            self.model.add(Dense(1))
            self.model.add(Activation("tanh"))
            
            self.model.compile(loss="mse", optimizer="adam")
            if verbose:
                print("Model created from scratch")
    
    def run_network(self, X_train, X_test, Y_train, Y_test, 
                    epochs=1, batch_size=100, save_file_append="test"):
        
        global_start_time = time.time()
        
        self.save_file = "results/"+"Keras_"+self.model_type+"_"+save_file_append
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        Y_train = Y_train.reshape(Y_train.shape[0], Y_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        Y_test = Y_test.reshape(Y_test.shape[0], Y_test.shape[1], 1)

        try:
            self.history = LossHistory()
            self.model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, 
                      validation_split=0.10, callbacks=[self.history])
            predicted = self.model.predict(X_test)
            predicted = np.reshape(predicted, (predicted.size,))
        except KeyboardInterrupt:
            print('\nTraining duration (s) : {} seconds'.format(time.time() - global_start_time))
        
        self.model.save(self.save_file)    
        
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,5))
            
        ax1.plot(self.history.losses)
        ax1.set_title("Learning rate over training", fontsize=16)
        ax1.set_xlabel("SGD steps", fontsize=14)
        ax1.set_ylabel("Loss value", fontsize=14)
        
        try:
            ax2.plot(Y_test.flatten()[:100], label="true")
            ax2.plot(predicted[:100], label="predicted")
            ax2.set_title("Example prediction", fontsize=16) 
            ax2.set_xlabel("Test data step", fontsize=14)
            ax2.set_ylabel("Test value", fontsize=14)    
            ax2.legend()
            plt.show()   
        except Exception as e:
            print(str(e))
            plt.show()
            
        print('Training duration (s) : {} seconds'.format(time.time() - global_start_time))
    
    def generate_data(self, seed_data, L_gen, time_len=99):
        gen_data = np.asarray(seed_data)
        for i in range(min(L_gen,time_len-1)):
            input_data = np.append(gen_data, np.zeros(time_len-i-1)).reshape(1,time_len,1)    
            gen_data = np.append(gen_data, np.asarray(self.model.predict(input_data).flatten()[i]))
        
        if L_gen >= time_len:
            for i in range(L_gen-time_len+1):
                input_data = gen_data[-time_len:].reshape(1,time_len,1) 
                gen_data = np.append(gen_data, np.asarray(self.model.predict(input_data).flatten()[-1]))

        return gen_data    
    
    def plot_from_seed(self, seed_data_point):
        seed_data = np.append(np.asarray(seed_data_point), np.zeros(98)).reshape(1,99,1)
        plt.plot(self.model.predict(seed_data).flatten())
        plt.title("Data generated from seed {}".format(seed_data_point), fontsize=16)
        plt.show()
        
    def compare_to_sequence(self, data_seq):
        
        gen_data = self.generate_data(data_seq[0], len(data_seq))
        
        plt.figure(figsize=(16,5))
        plt.plot(data_seq, label="true data")
        plt.plot(gen_data, label="generated data")
        plt.title("Sequentially generated vs true data", fontsize=16)
        plt.xlim([0,len(data_seq)])
        plt.legend()
        plt.show()
        
    def return_FFT(self, data_seq, FFT_Start=0, FFT_End=200, rm_mean=False, gen_data=None):
        
        if gen_data is None:
            gen_data = self.generate_data(data_seq[0], len(data_seq))
        
        if rm_mean:
            pred_data = gen_data[FFT_Start:FFT_End] - np.mean(gen_data[FFT_Start:FFT_End])
            compare_data = data_seq[FFT_Start:FFT_End] - np.mean(data_seq[FFT_Start:FFT_End])
        else:
            pred_data = gen_data[FFT_Start:FFT_End]
            compare_data = data_seq[FFT_Start:FFT_End]
        
        true_FFT = np.absolute(np.fft.fft(compare_data))
        predict_FFT = np.absolute(np.fft.fft(pred_data))
        return true_FFT, predict_FFT
        
    def plot_vs_FFT(self, data_seq, FFT_Start=0, FFT_End=200, FFT_xlim=None, rm_mean=False):

        true_FFT, predict_FFT = self.return_FFT(data_seq, FFT_Start, FFT_End, rm_mean)
        
        if FFT_xlim is None:
            FFT_xlim = np.int((FFT_End - FFT_Start)/4.)
        
        plt.figure(figsize=(16,5))
        plt.plot(true_FFT, label="true data")
        plt.plot(predict_FFT, label="generated data")
        plt.title("FFT of generated vs true data", fontsize=16)
        plt.xlim([0,FFT_xlim])
        plt.legend()
        plt.show()        
        
    def plot_seq_and_FFT(self, data_seq, description, rm_mean=True):
        
        gen_data = self.generate_data(data_seq[0], len(data_seq))
        true_FFT, predict_FFT = self.return_FFT(data_seq, FFT_Start=0, FFT_End=100, rm_mean=rm_mean, gen_data=gen_data)

        plt.figure(figsize=(16,5))
        
        plt.subplot(1,2,1)
        plt.plot(data_seq, label="Data Actual")
        plt.plot(gen_data, label="Data Generated")
#        plt.title(description)
        plt.xlim([0,len(data_seq)])
        plt.legend()
        
        plt.subplot(1,2,2)
        plt.plot(true_FFT, label="FFT Actual")
        plt.plot(predict_FFT, label="FFT Generated")
#        plt.title(description)
        plt.xlim([0,25])
        plt.legend()
        
        plt.suptitle(description)
        plt.show()      
        
    def plot_FFT_evolution(self, data_seq, FFT_step=100, FFT_xlim=None, rm_mean=False):
        
        n_steps = np.int(len(data_seq)/FFT_step)
        
        cNorm = colors.Normalize(vmin=0,vmax=n_steps) #normalise the colormap
        scalarMap = cm.ScalarMappable(norm=cNorm,cmap='RdYlGn') #map numbers to colors
        
        plt.figure(figsize=(16,5))
        
        for i in range(n_steps):
            true_FFT, predict_FFT = self.return_FFT(data_seq, FFT_Start=i*FFT_step, FFT_End=(i+1)*FFT_step, rm_mean=rm_mean)
            plt.plot(predict_FFT, c=scalarMap.to_rgba(n_steps-1-i), label="generated data from "+str(i*FFT_step)+" to "+str((i+1)*FFT_step))
        
        if FFT_xlim is None:
            FFT_xlim = np.int(FFT_step/4.)        
        
        plt.plot(true_FFT, c=scalarMap.to_rgba(n_steps), label="true data")
        plt.title("FFT of generated vs true data", fontsize=16)
        plt.xlim([0,FFT_xlim])
        plt.legend()
        plt.show()       