from collections import OrderedDict
import time

import numpy as np
import theano
from theano import tensor as T
#import lasagne
try:
    import cPickle as pickle
except:
    import pickle

from .utilities import matrix_init
from .utilities import offset_init
from .utilities import to_one_hot as my_one_hot
from .utilities import adam


class TS_parent:
    """
    Parent class for all time series RNN models.
    """
    
    def __init__(self, non_linearity=T.tanh, save_dir="results/"):

        self.phi = non_linearity
        self.save_dir = save_dir   

    def _build_network(self):
        
        # Define all the input to the computation graph
        self.X = T.fmatrix() # KxN
        self.labels = T.fvector() # K*N x 1

    def save_network(self, save_dir=None, save_file=None):
        if save_file is None:
            save_file = self.save_file + "_model.save"
        if save_dir is None:
            save_dir = self.save_dir
        with  open(save_dir + save_file, 'wb') as f:
            for obj in self.all_params:
                pickle.dump(obj.get_value(), f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Saved model as " + save_file)

    def load_network(self, save_dir=None, save_file=None):
        if save_file is None:
            save_file = self.save_file + "_model.save"
        if save_dir is None:
            save_dir = self.save_dir
        with open(save_dir + save_file, 'rb') as f:
            for obj in self.all_params:
                obj.set_value(pickle.load(f))            


class RNNTS_parent(TS_parent):
    """
    Parent class for the RNN time series baseline models.
    """

    def __init__(self, non_linearity=T.tanh, save_dir="results/"):
        self.model_type = "RNNTS"     
        super().__init__(non_linearity, save_dir)              
        
    def train_step(self, X_batch, y_batch):
        return self.train_func(X_batch, y_batch)

    def get_preplexity(self, X_batch, y_batch):
        loss = self.loss_func(X_batch, y_batch)
        perplexity = 2**(loss/(X_batch.size*np.log(2)))
        return loss, perplexity

    def calc_dataset_perplexity(self, n_points, zip_data, save_location, data_type='train'):
      
        batches = np.random.choice(len(zip_data), n_points, replace=False)
        zip_data = np.asarray(zip_data)            
        zip_data = zip_data[batches]        
        
#        print('Calculating losses for '+data_type+' dataset')
        start = time.time()
        full_loss, full_perplexity = 0.0, 0.0
        N_words = 0
        
        for x_batch, y_batch in zip_data:
            loss = self.loss_func(x_batch.T, y_batch)                            
            full_loss += loss
            N_words += x_batch.size            
      
        full_loss = full_loss/N_words
        full_perplexity = 2**(full_loss/np.log(2))
    
        with open(save_location, 'a') as loss_file:
            loss_file.write("%f, %f \n" % (full_loss, full_perplexity))
        print('Calculated '+data_type+' loss of %2f in %2f minutes' % (full_loss, (time.time()-start)/60.0))    
    
    def predict_word(self, sent):
        return self.predict_func(sent)[-1]
    
    def generate_data(self, starting_value=[0.], max_len=100.):
        
        L = max_len + len(starting_value)
        sent_list = starting_value
        sent_array = np.asarray(sent_list).reshape(len(sent_list),1)

        while len(sent_list) < L:
            
            next_value = self.predict_word(sent_array)
            sent_list.append(next_value)
            sent_array = np.asarray(sent_list).reshape(len(sent_list),1)

        return sent_list
    

class RNNTS_1L(RNNTS_parent):
    """
    Implements the most basic RNN time series model.
    """

    def __init__(self, time_len=100, hidden_dim=10, adam_learn_rate=0.001, non_linearity=T.tanh, 
                 save_dir="results/", save_file="RNNTS"):
        
        super().__init__(non_linearity, save_dir)
        self.save_file = save_file
        self.time_len = time_len
        self.hdn_dim = hidden_dim
        
        # init params
        self.U = offset_init(self.hdn_dim)
        self.V = matrix_init(self.hdn_dim, self.hdn_dim)
        self.M = matrix_init(self.hdn_dim, 1)
        self.b = offset_init(self.hdn_dim)
        self.c = offset_init(1)

        # build network
        self.loss, self.preds = self._build_network()
        self.all_params = [self.U, self.V, self.M, self.b, self.c]
        
        # compute gradient descent updates for each descent step
        updates = adam(self.loss, self.all_params, learning_rate=adam_learn_rate, 
                       b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8)
                       
        # Compile theano functions
        self.train_func = theano.function(inputs=[self.X, self.labels],
                                          outputs=self.loss,
                                          updates=updates,
                                          allow_input_downcast=True)

        self.predict_func = theano.function(inputs=[self.X],
                                            outputs=self.preds,
                                            allow_input_downcast=True)

        self.loss_func = theano.function(inputs=[self.X, self.labels],
                                         outputs=self.loss,
                                         allow_input_downcast=True)

    def _build_network(self):

        super()._build_network()
        
        def step(X_t, H_tm1, U, V, M, b, c):
            """ The core equations for the RNN updates"""

            H_t = self.phi(T.outer(X_t, U) + T.dot(H_tm1, V) + b.dimshuffle(['x', 0]))
            Y_t = T.tanh(T.dot(H_t, M) + c)

            return H_t, Y_t

        K, N = self.X.shape
        H_init = T.zeros((N, self.hdn_dim))

        outputs, _ = theano.scan(step,
                                 sequences=[self.X],
                                 outputs_info=[H_init, None],
                                 non_sequences=[self.U, self.V, self.M, self.b, self.c])

        # Finally we get the ouputs and create the loss
        preds = T.reshape(outputs[1], (K*N), ndim=1)
        loss = T.sum( T.pow(preds-self.labels,2) )

        return loss, preds



class RNNTS_1L_LSTM(RNNTS_parent):
    """
    Implements an RNN time series model with LSTM cells.
    """

    def __init__(self, time_len=100, hidden_dim=10, adam_learn_rate = 0.001, 
                 non_linearity=T.tanh, save_dir="results/", save_file="RNNTS"):
        
        super().__init__(non_linearity, save_dir)
        self.save_file = save_file
        self.time_len = time_len
        self.hdn_dim = hidden_dim

        # init params
        self.U_i = offset_init(self.hdn_dim)
        self.U_f = offset_init(self.hdn_dim)
        self.U_o = offset_init(self.hdn_dim)
        self.U_g = offset_init(self.hdn_dim)        
        self.V_i = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_f = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_o = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_g = matrix_init(self.hdn_dim, self.hdn_dim)
        self.b_i = offset_init(self.hdn_dim)
        self.b_f = offset_init(self.hdn_dim)
        self.b_o = offset_init(self.hdn_dim)
        self.b_g = offset_init(self.hdn_dim)
        self.M = matrix_init(self.hdn_dim, 1)
        self.b = offset_init(1)
        
        # build network
        self.loss, self.preds = self._build_network()
        self.all_params = [self.U_i, self.U_f, self.U_o, self.U_g, self.V_i, self.V_f, self.V_o, 
                           self.V_g, self.b_i, self.b_f, self.b_o, self.b_g, self.M, self.b]

        # compute gradient descent updates for each descent step
        updates = adam(self.loss, self.all_params, learning_rate=adam_learn_rate, 
                       b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8)
                       
        # Compile theano functions
        self.train_func = theano.function(inputs=[self.X, self.labels],
                                          outputs=self.loss,
                                          updates=updates,
                                          allow_input_downcast=True)

        self.predict_func = theano.function(inputs=[self.X],
                                            outputs=self.preds,
                                            allow_input_downcast=True)

        self.loss_func = theano.function(inputs=[self.X, self.labels],
                                         outputs=self.loss,
                                         allow_input_downcast=True)

    def _build_network(self):

        super()._build_network()

        def step(X_t, H_t, C_t):
            """ The core equations for the RNN updates"""
            
            i_t = T.nnet.hard_sigmoid(T.outer(X_t, self.U_i) + T.dot(H_t, self.V_i) + self.b_i.dimshuffle(['x', 0]))
            f_t = T.nnet.hard_sigmoid(T.outer(X_t, self.U_f) + T.dot(H_t, self.V_f) + self.b_f.dimshuffle(['x', 0]))
            o_t = T.nnet.hard_sigmoid(T.outer(X_t, self.U_o) + T.dot(H_t, self.V_o) + self.b_o.dimshuffle(['x', 0]))
            G_t = self.phi(T.outer(X_t, self.U_g) + T.dot(H_t, self.V_g) + self.b_g.dimshuffle(['x', 0]))
            C_t = f_t*C_t + i_t*G_t
            H_t = self.phi(C_t)*o_t
            Y_t = T.tanh(T.dot(H_t, self.M) + self.b)
            return H_t, C_t, Y_t        

        K, N = self.X.shape
        H_init = T.zeros((N, self.hdn_dim))
        C_init = T.zeros((N, self.hdn_dim))
        # Outputs[1] will be a KxNxV tensor of probability vectors
        outputs, _ = theano.scan(step,
                                 sequences=[self.X],
                                 outputs_info=[H_init, C_init, None])

        # Finally we get the ouputs and create the loss
        preds = T.reshape(outputs[2], (K*N), ndim=1)
        loss = T.sum( T.pow(preds-self.labels,2) )        
        
        return loss, preds          