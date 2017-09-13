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


class LM_parent:
    """
    Parent class for all language models.
    """
    
    def __init__(self, vocab_size=None, emb_dim=None, 
                 non_linearity=T.tanh, save_dir="results/", word_rep="one-hot"):

        self.vocab_size = vocab_size
        if emb_dim:
            self.emb_dim = emb_dim
        else:
            self.emb_dim = vocab_size           
        self.phi = non_linearity
        self.save_dir = save_dir   
        self.word_rep = word_rep
        self.log_like_thresh = None
        
    def _build_network(self):
        
        # Define all the input to the computation graph
        if self.word_rep == "pre-trained":
            self.X = T.imatrix() # KxN
            #ToDo Load up pre-trained word vectors
        elif self.word_rep == 'one-hot':
            self.X = T.imatrix() # KxN
            self.emb_mat = my_one_hot(self.X, self.vocab_size)
        elif self.word_rep == 'embeddings':
            self.X = T.imatrix()
            self.W_emb = matrix_init(self.vocab_size, self.emb_dim)
            self.emb_mat = self.W_emb[self.X]
        else:
            raise ValueError('Unknown word-representation type')

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

    def set_likelihood_threshold(self, thresh=False):
        
        if not thresh==False:
            self.log_like_thresh = thresh

        elif self.log_like_thresh==None:
            loss_filename = self.save_dir+"/"+self.save_file+"_train_loss.save"
            losses = []
            loss_file = open(loss_filename, 'r')
            for line in loss_file.readlines():
                losses.append([float(i) for i in line.split(', ')])            
            losses = np.asarray(losses)    
            loss = losses[:,0]        
            self.log_like_thresh = -1*np.mean([loss[-1], loss[-2]])
            
    def generate_sentence(self, speed='fast'):

        max_len = 0
        min_len = 0
        log_like_multiplier = 0.0
            
        if speed == 'fast':
            max_len = 30           

        elif speed == 'slow':
            max_len = 200
            min_len = 6    

        elif speed == 'likelihood':
            max_len = 200
            min_len = 6
            self.set_likelihood_threshold()
            log_like_multiplier = .95
            
        else:
            raise ValueError('In function generate_sentence, speed must be set to fast or slow')               

        return max_len, min_len, log_like_multiplier
                
        
        
class RNNlm_parent(LM_parent):
    """
    Parent class for the RNNlm baseline models.
    """

    def __init__(self, vocab_size=None, emb_dim=None, 
                 non_linearity=T.tanh, save_dir="results/", word_rep="one-hot"):
        self.model_type = "RNNlm"     
        super().__init__(vocab_size, emb_dim, non_linearity, save_dir, word_rep)              
        
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
    
    def predict_word(self, init_word):
        return self.predict_func(init_word)[-1]
    
    def generate_sentence(self, word_to_ind, speed='fast'):

        max_len, min_len, log_like_multiplier = super().generate_sentence(speed)

        new_sent = [word_to_ind['SENTENCE_START']] 

        if speed == 'fast':
            while len(new_sent) < max_len and new_sent[-1] != word_to_ind['SENTENCE_END']:
                next_word_probs = self.predict_word([new_sent])
                sample = np.random.multinomial(1, next_word_probs)
                sampled_word = np.argmax(sample)
                new_sent.append(sampled_word)

        elif speed == 'slow':     
            counter = 0
            while new_sent[-1] != word_to_ind['SENTENCE_END']:
                next_word_probs = self.predict_word([new_sent])
                sample = np.random.multinomial(1, next_word_probs)
                sampled_word = np.argmax(sample)
                new_sent.append(sampled_word)
                counter += 1
                if (counter > max_len or sampled_word == word_to_ind['UNKNOWN_TOKEN']):
                    new_sent = None
                    break           
            if counter < min_len:
                new_sent = None
            
        elif speed == 'likelihood':     
            counter = 0
            thresh = self.log_like_thresh*log_like_multiplier
            logP = [0.0]
            while new_sent[-1] != word_to_ind['SENTENCE_END'] and np.mean(logP) > thresh:
                next_word_probs = self.predict_word([new_sent])
                sample = np.random.multinomial(1, next_word_probs)
                sampled_word = np.argmax(sample)
                logP.append(np.log(next_word_probs[sampled_word]))
                new_sent.append(sampled_word)
                counter += 1
                if (counter > max_len or sampled_word == word_to_ind['UNKNOWN_TOKEN']):
                    new_sent = None
                    break           
            if counter < min_len:
                new_sent = None
                
            if new_sent:
                if np.mean(logP) < thresh:
                    new_sent = None
                else:
                    print("Per-word log likelihood was: ", np.mean(logP))

        return new_sent
    
    

class RNNlm_1L(RNNlm_parent):
    """
    Implements the most basic RNN language model.
    """

    def __init__(self, hidden_dim=10, vocab_size=None, emb_dim=None,
                 adam_learn_rate=0.001, non_linearity=T.tanh, 
                 save_dir="results/", save_file="RNNlm", word_rep="one-hot"):
        
        super().__init__(vocab_size, emb_dim, non_linearity, save_dir, word_rep)
        self.hdn_dim = hidden_dim           
        self.save_file = save_file

        # init params
        self.U = matrix_init(self.emb_dim, self.hdn_dim)
        self.V = matrix_init(self.hdn_dim, self.hdn_dim)
        self.M = matrix_init(self.hdn_dim, self.vocab_size)
        self.b = offset_init(self.hdn_dim)

        # build network
        self.loss, self.preds = self._build_network()
        self.all_params = [self.U, self.V, self.M, self.b]
        if self.word_rep == 'embeddings':
            self.all_params.append(self.W_emb)
        
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
        
        def step(X_t, H_tm1, U, V, M, b):
            """ The core equations for the RNN updates"""

            H_t = self.phi(T.dot(X_t, U) + T.dot(H_tm1, V) + b.dimshuffle(['x', 0]))
            Y_t = T.nnet.softmax(T.dot(H_t, M))

            return H_t, Y_t

        self.labels = T.ivector() # K*N x 1
        K, N, D = self.emb_mat.shape
        H_init = T.zeros((N, self.hdn_dim))
        # Outputs[1] will be a KxNxV tensor of probability vectors
        outputs, _ = theano.scan(step,
                                 sequences=[self.emb_mat],
                                 outputs_info=[H_init, None],
                                 non_sequences=[self.U, self.V, self.M, self.b])

        # Finally we get the ouputs and create the loss
        preds = T.reshape(outputs[1], (K*N, self.vocab_size), ndim=2)
        loss = T.sum(T.nnet.categorical_crossentropy(preds, self.labels))

        return loss, preds
    
    def iterate_h(self, n_steps, sentence=[]):
        
        if self.phi==T.tanh:
            def phi(x):
                return np.tanh(x)
        elif self.phi==T.nnet.relu:
            def phi(x):
                return np.maximum(0,x)
        
        if len(sentence)==0:
            h = np.zeros(self.hdn_dim)
            for n in range(n_steps):
                h = phi(h.dot(self.V.get_value()) + self.b.get_value())
                
        else:
            if self.word_rep == 'one-hot':
                emb_sent = []
                for w in sentence:
                    one_hot = np.zeros(self.vocab_size)
                    one_hot[w] = 1
                    emb_sent.append(one_hot)
            elif self.word_rep == 'embeddings':
                emb_sent = self.W_emb.get_value()[sentence]
            
            h = np.zeros(self.hdn_dim)
            for n in range(n_steps):
                x = emb_sent[n%len(sentence)]
                h = phi(x.dot(self.U.get_value()) + h.dot(self.V.get_value()) + self.b.get_value())        
            
        return h
    
    
class RNNlm_1L_LSTM(RNNlm_parent):
    """
    Implements a basic RNN language model with LSTM cells.
    """

    def __init__(self, hidden_dim=10, vocab_size=None, emb_dim=None,
                 adam_learn_rate = 0.001, non_linearity=T.tanh, 
                 save_dir="results/", save_file="RNNlm", word_rep="one-hot"):
        
        super().__init__(vocab_size, emb_dim, non_linearity, save_dir, word_rep)
        self.hdn_dim = hidden_dim           
        self.save_file = save_file

        # init params
        self.U_i = matrix_init(self.emb_dim, self.hdn_dim)
        self.U_f = matrix_init(self.emb_dim, self.hdn_dim)
        self.U_o = matrix_init(self.emb_dim, self.hdn_dim)
        self.U_g = matrix_init(self.emb_dim, self.hdn_dim)
        self.V_i = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_f = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_o = matrix_init(self.hdn_dim, self.hdn_dim)
        self.V_g = matrix_init(self.hdn_dim, self.hdn_dim)
        self.b_i = offset_init(self.hdn_dim)
        self.b_f = offset_init(self.hdn_dim)
        self.b_o = offset_init(self.hdn_dim)
        self.b_g = offset_init(self.hdn_dim)
        self.M = matrix_init(self.hdn_dim, self.vocab_size)
        self.b = offset_init(self.vocab_size)
        
        # build network
        self.loss, self.preds = self._build_network()
        self.all_params = [self.U_i, self.U_f, self.U_o, self.U_g, self.V_i, self.V_f, self.V_o, 
                           self.V_g, self.b_i, self.b_f, self.b_o, self.b_g, self.M, self.b]
        
        if self.word_rep == 'embeddings':
            self.all_params.append(self.W_emb)
        
        # compute gradient descent updates for each descent step
        updates = adam(self.loss, self.all_params, learning_rate=adam_learn_rate, 
                       b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8)

        # Compile theano functions
        self.train_func = theano.function(inputs=[self.X, self.labels],
                                          outputs=self.loss,
                                          updates=updates,
                                          allow_input_downcast=True)

        all_grads = theano.grad(self.loss, self.all_params)
#        norm_list = [(update[0]-update[1]).norm(2) for update in updates]
        norm_list = [grad.norm(2) for grad in all_grads]
        self.updates_norm = theano.function(inputs=[self.X, self.labels],
                                          outputs=norm_list,
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
            
            i_t = T.nnet.hard_sigmoid(T.dot(X_t, self.U_i) + T.dot(H_t, self.V_i) + self.b_i.dimshuffle(['x', 0]))
            f_t = T.nnet.hard_sigmoid(T.dot(X_t, self.U_f) + T.dot(H_t, self.V_f) + self.b_f.dimshuffle(['x', 0]))
            o_t = T.nnet.hard_sigmoid(T.dot(X_t, self.U_o) + T.dot(H_t, self.V_o) + self.b_o.dimshuffle(['x', 0]))
            G_t = self.phi(T.dot(X_t, self.U_g) + T.dot(H_t, self.V_g) + self.b_g.dimshuffle(['x', 0]))
            C = f_t*C_t + i_t*G_t
            H = self.phi(C_t)*o_t
            Y = T.nnet.softmax(T.dot(H, self.M) + self.b.dimshuffle(['x', 0]))
            return H, C, Y        

        self.labels = T.ivector() # K*N x 1
        K, N, D = self.emb_mat.shape
        H_init = T.zeros((N, self.hdn_dim))
        C_init = T.zeros((N, self.hdn_dim))
        # Outputs[1] will be a KxNxV tensor of probability vectors
        outputs, _ = theano.scan(step,
                                 sequences=[self.emb_mat],
                                 outputs_info=[H_init, C_init, None])
#                                 ,non_sequences=[self.U, self.V, self.M, self.b])

        # Finally we get the ouputs and create the loss
        preds = T.reshape(outputs[2], (K*N, self.vocab_size), ndim=2)
        loss = T.sum(T.nnet.categorical_crossentropy(preds, self.labels))

        return loss, preds    
        
    def iterate_h(self, n_steps, sentence=[]):
        
        if self.phi==T.tanh:
            def phi(x):
                return np.tanh(x)
        elif self.phi==T.nnet.relu:
            def phi(x):
                return np.maximum(0,x)
                
        def hard_sigmoid(x):
            x = np.asarray(x)
            min_term = 1*(np.ones(len(x)) < (0.5*x + 0.5)) + (0.5*x + 0.5)*(np.ones(len(x)) >= (0.5*x + 0.5))
            max_term = min_term*(min_term > 0)
            return max_term
        
        if len(sentence)==0:
            h = np.zeros(self.hdn_dim)
            c = np.zeros(self.hdn_dim)
            for n in range(n_steps):
                i = hard_sigmoid(h.dot(self.V_i.get_value()) + self.b_i.get_value())
                f = hard_sigmoid(h.dot(self.V_f.get_value()) + self.b_f.get_value())
                o = hard_sigmoid(h.dot(self.V_o.get_value()) + self.b_o.get_value())
                g = phi(h.dot(self.V_g.get_value()) + self.b_g.get_value())
                c = f*c + i*g
                h = phi(c)*o   
                
        else:
            if self.word_rep == 'one-hot':
                emb_sent = []
                for w in sentence:
                    one_hot = np.zeros(self.vocab_size)
                    one_hot[w] = 1
                    emb_sent.append(one_hot)
            elif self.word_rep == 'embeddings':
                emb_sent = self.W_emb.get_value()[sentence]
            
            h = np.zeros(self.hdn_dim)
            c = np.zeros(self.hdn_dim)
            for n in range(n_steps):
                x = emb_sent[n%len(sentence)]
                i = hard_sigmoid(x.dot(self.U_i.get_value()) + h.dot(self.V_i.get_value()) + self.b_i.get_value())
                f = hard_sigmoid(x.dot(self.U_f.get_value()) + h.dot(self.V_f.get_value()) + self.b_f.get_value())
                o = hard_sigmoid(x.dot(self.U_o.get_value()) + h.dot(self.V_o.get_value()) + self.b_o.get_value())
                g = phi(x.dot(self.U_g.get_value()) + h.dot(self.V_g.get_value()) + self.b_g.get_value())
                c = f*c + i*g
                h = phi(c)*o               
            
        return h
    

    

class RNNlm_2L(RNNlm_parent):
    """
    Implements the most basic RNN language model.
    """

    def __init__(self, hidden_dim_1=100, hidden_dim_2=100, vocab_size=None, emb_dim=None,
                 adam_learn_rate=0.001, non_linearity=T.tanh, 
                 save_dir="results/", save_file="RNNlm_2L", word_rep="one-hot"):

        super().__init__(vocab_size, emb_dim, non_linearity, save_dir, word_rep)
        self.hdn_dim_1 = hidden_dim_1
        self.hdn_dim_2 = hidden_dim_2        
        self.save_file = save_file

        # init params
        self.U = matrix_init(self.emb_dim, self.hdn_dim_1)
        self.V = matrix_init(self.hdn_dim_1, self.hdn_dim_1)
        self.b = offset_init(self.hdn_dim_1)
        self.U2 = matrix_init(self.hdn_dim_1, self.hdn_dim_2)
        self.V2 = matrix_init(self.hdn_dim_2, self.hdn_dim_2)
        self.b2 = offset_init(self.hdn_dim_2)
        self.M = matrix_init(self.hdn_dim_2, self.vocab_size)
        self.c = offset_init(self.vocab_size)
        
        # build network
        self.loss, self.preds = self._build_network()
        self.all_params = [self.U, self.V, self.b, self.U2, self.V2, self.b2, self.M, self.c]
        if self.word_rep == 'embeddings':
            self.all_params.append(self.W_emb)

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
                                          allow_input_downcast=True
                                            )

        self.loss_func = theano.function(inputs=[self.X, self.labels],
                                         outputs=self.loss,
                                          allow_input_downcast=True)

    def _build_network(self):

        super()._build_network()

        def step(X_t, H_tm1, H2_tm1, U, V, b, U2, V2, b2, M, c):
            """ The core equations for the RNN updates"""

            H_t = self.phi(T.dot(X_t, U) + T.dot(H_tm1, V) + b.dimshuffle(['x', 0]))
            H2_t = self.phi(T.dot(H_t, U2) + T.dot(H2_tm1, V2) + b2.dimshuffle(['x', 0]))
            Y_t = T.nnet.softmax(T.dot(H2_t, M) + c.dimshuffle(['x', 0]))

            return H_t, H2_t, Y_t

        self.labels = T.ivector() # K*N x 1
        K, N, D = self.emb_mat.shape
        H_init = T.zeros((N, self.hdn_dim_1))
        H2_init = T.zeros((N, self.hdn_dim_2))
        # Outputs[1] will be a KxNxV tensor of probability vectors
        outputs, _ = theano.scan(step,
                                 sequences=[self.emb_mat],
                                 outputs_info=[H_init, H2_init, None],
                                 non_sequences=[self.U, self.V, self.b, self.U2, self.V2, self.b2, self.M, self.c])

        # Finally we get the ouputs and create the loss
        preds = T.reshape(outputs[2], (K*N, self.vocab_size), ndim=2)
        loss = T.sum(T.nnet.categorical_crossentropy(preds, self.labels))

        return loss, preds            