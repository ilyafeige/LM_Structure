""" Code to load and pre-process data """
from collections import Counter
import time
try:
    import cPickle as pickle
except:
    import pickle

import pandas as pd
import numpy as np
import nltk


def cluster_pos(pos_list):

    pos_to_lowpos = dict()
    
    for orig_pos_str in pos_list:

        pos_str = orig_pos_str
        
        # remove modifiers
        pos_str = pos_str.replace('-HL','')
        pos_str = pos_str.replace('-TL','')
        pos_str = pos_str.replace('FW-','')
        pos_str = pos_str.replace('-NC','')    

        # strip off second POS in combined words (e.g., "wanna" is tagged VB+TO, 
        # since it is a contracted form of the two words, want/VB and to/TO.)
        pos_str = pos_str.split('+',1)[0]

        # Remove negations    
        if not (pos_str == '*'):
            pos_str = pos_str.replace('*','')

        # Merge all nouns    
        if pos_str[0:2] in ['NN', 'NR']:
            pos_str = 'NN'

        # Merge all verbs
        if pos_str[0:2] in ['VB', 'BE', 'DO', 'HV']:
            pos_str = 'VB'           

        # Merge all advectives
        if pos_str[0:2] == 'JJ':
            pos_str = 'JJ'

        # Merge all pronouns
        if pos_str in ['PN', 'PN$', 'PP$', 'PP$$', 'PPL', 'PPLS', 'PPO', 
                       'PPS', 'PPSS', 'PRP', 'PRP$', 'WP$', 'WPO', 'WPS']:
            pos_str = 'PN'

        # Merge all adverbs
        if pos_str in ['RB', 'RB$', 'RBR', 'RBT', 'RN', 'RP', 'WRB']:
            pos_str = 'RB'

        # Merge all proper nouns
        if pos_str[0:2] == 'NP':
            pos_str = 'NP'        

        # Merge all qualifiers / quantifiers / determiners
        if pos_str in ['ABL', 'ABN', 'ABX', 'AP', 'AP$', 'CD', 'CD$', 'DT', 
                       'DT$', 'DTI', 'DTS', 'DTX', 'OD', 'QL', 'QLP', 'WDT', 'WQL']:
            pos_str = 'DT'        

        # Merge all conjunctions
        if pos_str in ['CC', 'CS']:
            pos_str = 'C'            
        
        pos_to_lowpos[orig_pos_str] = pos_str
    
    return pos_to_lowpos
    

def load_data(corpus, vocab_size=10000, train_frac=1, save_file='data.save', hierarchy_level=0):
    """ Load Data from an NLTK Corpus

        inputs
        ------
        corpus - nltk.corpus.reader.tagged.CategorizedTaggedCorpusReader or
                 start if loading data from file.
        vocab_size - integer how many words to keep
        save_file - str

        outputs
        -------
        X - np integer array of word indices of shape (num_sents, sent_len)
        word_to_index - dict
        index_to_word - dict
        POS - np integer array of word indices of shape (num_sents, sent_len)
        pos_to_index - dict
        index_to_pos - dict
    """

    # Load the data
    if type(corpus) == str:
        with open(corpus, 'rb') as f:
            results = []
            for i in range(12):
                results.append(pickle.load(f))
        return tuple(results)
    else:
        pos_sents_corpusview = corpus.tagged_sents()

    # Add the sent-start and sent-end tokens & make words lower case
    pos_sents = []
    for i, pos_sent in enumerate(pos_sents_corpusview):
        pos_sent = [(pos_pair[0].lower(), pos_pair[1]) for pos_pair in pos_sent]
        pos_sent = [('SENTENCE_START', 'str')] + pos_sent + [('SENTENCE_END', 'end')]
        pos_sents.append(pos_sent)

    # Count the word frequencies to remove uncommon words
    words = [group[0] for sent in pos_sents for group in sent]
#    poss  = list(set([group[1] for sent in pos_sents for group in sent]))
    poss = [group[1] for sent in pos_sents for group in sent]
    
    pos_to_lowpos = cluster_pos(poss)
    low_poss = list(pos_to_lowpos.values())
    
    word_freq = nltk.FreqDist(words)
    pos_freq = nltk.FreqDist(poss)
    low_poss_freq = nltk.FreqDist(low_poss)
    
    vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:vocab_size-1]
    pos = sorted(pos_freq.items(), key=lambda x: x[1], reverse=True)
    low_pos = sorted(low_poss_freq.items(), key=lambda x: x[1], reverse=True)
    
    vocab = [item[0] for item in vocab]
    pos = [item[0] for item in pos]
    low_pos = [item[0] for item in low_pos]
    
    vocab += ['UNKOWN_TOKEN']
#    pos += ['UNK']

    # Get the mappings to index and back
    index_to_word = {index: word for index, word in enumerate(vocab)}
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_pos = {index: pos for index, pos in enumerate(pos)}
    pos_to_index = {po: index for index, po in enumerate(pos)}
    index_to_lowpos = {index: lpos for index, lpos in enumerate(low_pos)}
    lowpos_to_index = {lpo: index for index, lpo in enumerate(low_pos)}    
    
    '''
    # Replace words with indices
    for sent in pos_sents:
        for i, word_pos in enumerate(sent):
            if word_pos[0] not in word_to_index:
                sent[i] = (word_to_index['UNKOWN_TOKEN'], pos_to_index[word_pos[1]], 
                           lowpos_to_index[pos_to_lowpos[word_pos[1]]])
            else:
                sent[i] = (word_to_index[word_pos[0]], pos_to_index[word_pos[1]], 
                           lowpos_to_index[pos_to_lowpos[word_pos[1]]])

    # split the POS from the words and return
    X = np.asarray([[word_pos[0] for word_pos in sent] for sent in pos_sents])
    POS = np.asarray([[word_pos[1] for word_pos in sent] for sent in pos_sents])
    LowPOS = np.asarray([[word_pos[2] for word_pos in sent] for sent in pos_sents])
    '''
    
    X = []
    POS = []
    LowPOS = [] 
    for sent in pos_sents:
        X_sent = []
        POS_sent = []
        LowPOS_sent = []
        for word_pos in sent:
            
            if word_pos[0] not in word_to_index:
                X_sent.append(word_to_index['UNKOWN_TOKEN'])
            else:
                X_sent.append(word_to_index[word_pos[0]])
                
            POS_sent.append(pos_to_index[word_pos[1]])
            LowPOS_sent.append(lowpos_to_index[pos_to_lowpos[word_pos[1]]])
        
        X.append(X_sent)
        POS.append(POS_sent)
        LowPOS.append(LowPOS_sent)        
    X = np.asarray(X)
    POS = np.asarray(POS)
    LowPOS = np.asarray(LowPOS)
        
    # Get the test train slit
    train_mask = np.random.permutation(X.shape[0]) < np.floor(train_frac*X.shape[0])
    X_train = X[train_mask]
    X_test = X[train_mask == False]
    POS_train = POS[train_mask]
    POS_test = POS[train_mask == False]
    LowPOS_train = LowPOS[train_mask]
    LowPOS_test = LowPOS[train_mask == False]
    
    # Get the batches
    X_batches, Y_batches, XPOS_batches, YPOS_batches, XLowPOS_batches, YLowPOS_batches = get_batches(X_train, POS_train, LowPOS_train)
        
    X_test, Y_test, XPOS_test, YPOS_test, XLowPOS_test, YLowPOS_test = get_batches(X_test, POS_test, LowPOS_test)

    # Save the results for easier loading later
    with open(save_file, 'wb') as f:
        for item in [X_batches, X_test, Y_batches, Y_test, word_to_index,
                     index_to_word, XPOS_batches, XPOS_test, YPOS_batches,
                     YPOS_test, pos_to_index, index_to_pos, XLowPOS_batches, 
                     XLowPOS_test, YLowPOS_batches, YLowPOS_test, 
                     lowpos_to_index, index_to_lowpos]:
            pickle.dump(item, f)

    if hierarchy_level == 2:
        return (X_batches, X_test, Y_batches, Y_test, word_to_index, 
                index_to_word, XPOS_batches, XPOS_test, YPOS_batches,
                YPOS_test, pos_to_index, index_to_pos, XLowPOS_batches, 
                XLowPOS_test, YLowPOS_batches, YLowPOS_test, 
                lowpos_to_index, index_to_lowpos)        
    elif hierarchy_level == 1:            
        return (X_batches, X_test, Y_batches, Y_test, word_to_index, 
                index_to_word, XPOS_batches, XPOS_test, YPOS_batches,
                YPOS_test, pos_to_index, index_to_pos)
    elif hierarchy_level == 0:
        return (X_batches, X_test, Y_batches, Y_test, word_to_index, 
                index_to_word)


def get_batch_lists(dataset, min_batch_size=16, max_batch_size=64):

    print("Gathering sentence lengths...")
    lengths = [len(sent) for sent in dataset]

    dataset_DF = pd.DataFrame({'Length': lengths, 'Sentence': dataset})

    print("Filtering out sentences with rare lengths")
    start = time.time()
    filtered_lengths = []
    counter_lengths = Counter(lengths)
    all_lengths = np.asarray(list(counter_lengths.keys()))
    np.random.shuffle(all_lengths)

    for ell in all_lengths:
        if counter_lengths[ell] >= min_batch_size:
            filtered_lengths.append(ell)
    print("Filtered all lengths in {} minutes.".format((time.time()-start)/60.0))

    print("Building batch lists...")
    start = time.time()
    batch_lists = []
    batch_lengths = []
    for ell in filtered_lengths:
        sent_indices = np.asarray(dataset_DF[dataset_DF['Length'] == ell].index.values)
        np.random.shuffle(sent_indices)
        BS_uncut = len(sent_indices)
        if BS_uncut > max_batch_size:
            N_batches_ell = np.int_(np.floor(np.true_divide(BS_uncut, max_batch_size)))
            for i in range(N_batches_ell):
                batch_lists.append(sent_indices[i*max_batch_size:(i+1)*max_batch_size])
                batch_lengths.append(max_batch_size)
            if ((np.true_divide(BS_uncut, max_batch_size)-N_batches_ell)*max_batch_size) >= min_batch_size:
                batch_lists.append(sent_indices[N_batches_ell*max_batch_size:BS_uncut])
                batch_lengths.append(BS_uncut-N_batches_ell*max_batch_size)
        else:
            batch_lists.append(sent_indices)
            batch_lengths.append(BS_uncut)

    np.random.shuffle(batch_lists)
    print('There are {} batches of data'.format(len(batch_lists)))

    return batch_lists


def flatten(x):
    return np.array([item for sublist in x for item in sublist]).astype('int32')


def get_batches(X, POS, LowPOS):

    batch_lists = get_batch_lists(X)
    X_batches= []
    Y_batches = []
    XPOS_batches = []
    YPOS_batches = []
    XLowPOS_batches = []
    YLowPOS_batches = []
    
    for batch_inds in batch_lists:

        s_batch = X[batch_inds]
        x_batch = np.array([s_batch[i][:-1] for i in range(len(s_batch))]).astype('int32')
        y_batch = np.array([s_batch[i][1:] for i in range(len(s_batch))]).astype('int32')
        y_batch = flatten(y_batch) # Flatten because T.nnet.categoricalcrossentropy embeds for us.
        X_batches.append(x_batch)
        Y_batches.append(y_batch)

        spos_batch = POS[batch_inds]
        xpos_batch = np.array([spos_batch[i][:-1] for i in range(len(spos_batch))]).astype('int32')
        ypos_batch = np.array([spos_batch[i][1:] for i in range(len(spos_batch))]).astype('int32')
        ypos_batch = flatten(ypos_batch)
        XPOS_batches.append(xpos_batch)
        YPOS_batches.append(ypos_batch)
        
        slowpos_batch = LowPOS[batch_inds]
        xlowpos_batch = np.array([slowpos_batch[i][:-1] for i in range(len(slowpos_batch))]).astype('int32')
        ylowpos_batch = np.array([slowpos_batch[i][1:] for i in range(len(slowpos_batch))]).astype('int32')
        ylowpos_batch = flatten(ylowpos_batch)
        XLowPOS_batches.append(xlowpos_batch)
        YLowPOS_batches.append(ylowpos_batch)        

    return X_batches, Y_batches, XPOS_batches, YPOS_batches, XLowPOS_batches, YLowPOS_batches
