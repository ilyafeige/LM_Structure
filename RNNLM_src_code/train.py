""" Code needed to train the models """
import os
import time
import random
try:
    import cPickle as pickle
except:
    import pickle
import copy

import numpy as np

from . import printing


def train_language_model(model, X_batches, X_test, Y_batches, Y_test, 
                         index_to_word, word_to_index, num_epochs=1, print_frequency=0.1,
                         save_dir=None, save_file=None, full_loss_freq=1, N_loss_eval_points=50):

    """Take in a RNNlm object and some data and train the model"""
    if save_dir is None:
        save_dir = model.save_dir
    if save_file is None:
        save_file = model.save_file
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    open(save_dir+save_file+'_train_loss.save', 'w').close()
    open(save_dir+save_file+'_test_loss.save', 'w').close()
    open(save_dir + save_file + '_batch_loss.save', 'w').close()
    open(save_dir + save_file + '_sentences.txt', 'w').close()

    print('Training Model...')
    num_data_seen = 0
    len_per_epoch = len(X_batches) + 0.0
    model_type = model.model_type 

    Data_train = list(zip(X_batches, Y_batches))
    Data_test = list(zip(X_test, Y_test))
    
    model.calc_dataset_perplexity(N_loss_eval_points, Data_train, 
                                  save_dir+save_file+'_train_loss.save', data_type='train')
    model.calc_dataset_perplexity(N_loss_eval_points, Data_test, 
                                  save_dir+save_file+'_test_loss.save', data_type='test')
    
    for i in range(num_epochs):
        counter = 0
        acum_loss = 0.0
        percent_time = time.time()
        for x_batch, y_batch in Data_train:
            batch_loss = model.train_step(x_batch.T, y_batch)/(1.0*x_batch.shape[0])
            acum_loss += batch_loss
            
#            update_norms = np.asarray(model.updates_norm(x_batch.T, y_batch))
#            update_norms = update_norms/len(update_norms)
#            print(np.max(update_norms), np.argmax(update_norms))
            
            # loss_file.write((str(batch_loss) + '\n').encode('utf8').strip())
            with open(save_dir + save_file + '_batch_loss.save', 'a') as loss_file:
                loss_file.write('%f \n' % batch_loss)
                loss_file.flush()
                num_data_seen += np.size(x_batch.shape[0])

            # Occasionally printing progress level
            counter += 1
            old_percent_complete = np.true_divide(counter-1, len_per_epoch*print_frequency)
            loop_percent_complete = np.true_divide(counter,  len_per_epoch*print_frequency)
            if np.floor(loop_percent_complete) != np.floor(old_percent_complete):
                print('\tAfter %.2f additional minutes, loop in epoch %d is %d%% complete with loss of %.2f' \
                % ((time.time()-percent_time)/60.0, i, loop_percent_complete*print_frequency*100., (acum_loss/counter)))
                percent_time = time.time()
                acum_loss = 0.
                
                model.calc_dataset_perplexity(N_loss_eval_points, Data_train, 
                                              save_dir+save_file+'_train_loss.save', data_type='train')
                model.calc_dataset_perplexity(N_loss_eval_points, Data_test, 
                                              save_dir+save_file+'_test_loss.save', data_type='test')

        model.save_network(save_dir, save_file + '_model.save')
        
#        if i%full_loss_freq == 0:            
#            model.calc_dataset_perplexity(N_loss_eval_points, Data_train, 
#                                          save_dir+save_file+'_train_loss.save', data_type='train')
#            model.calc_dataset_perplexity(N_loss_eval_points, Data_test, 
#                                          save_dir+save_file+'_test_loss.save', data_type='test')
                 
        print('\nSentence examples after epoch %d:\n' % i)
        printing.print_sentences(model, 3, index_to_word, word_to_index, max_attempts=100)

        random.shuffle(Data_train)
