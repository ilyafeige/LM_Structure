""" Code to load and pre-process data """

import pandas as pd
import numpy as np
import nltk
import csv
import sys
import itertools
import operator
import time
from collections import Counter
#try:
#    import cPickle as pickle
#except:
#    import pickle
from matplotlib import pyplot as plt         


csv.field_size_limit(sys.maxsize)

def load_data(filename="books_merged.txt", vocab_size=5000, max_N_sentences=100000, start_token="SENTENCE_START", end_token="SENTENCE_END", unk_token="UNKNOWN_TOKEN"):

    word_to_index = []
    index_to_word = []
    tokenized_sentences = []
    
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading %s file..." % filename)
    data_dir = '/project/data/'
    start = time.time()
    counter = 0
    with open(data_dir+filename, 'r') as f:
        reader = csv.reader(f)
        for s in reader:
            if not (("isbn" in str(s)) or (("copyright" in str(s)) and ("all rights reserved" in str(s)))):
                s = " ".join([start_token, str(s).replace("', '", ", ")[2:-2], end_token])
                tokenized_sentences.append(nltk.word_tokenize(s))
                if (counter % 1000000 == 0) and (counter > 0):
                    print("Read %dM sentences" % (counter/1000000))
#                    print tokenized_sentences[counter]
                counter += 1
            if counter > max_N_sentences:
                break
    print("Parsed %d sentences in %.2f minutes." % (len(tokenized_sentences), (time.time()-start)/60.0))
    
    # Count the word frequencies
    print("\nCounting word frequencies...")
    start = time.time()
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens in %.2f minutes." % (len(word_freq.items()), (time.time()-start)/60.0))

    # Get the most common words and build index_to_word and word_to_index vectors
    print("\nSorting vocabulary...")
    vocab = sorted(word_freq.items(), key=lambda x: (x[1], x[0]), reverse=True)[:vocab_size-1]
    print("Using vocabulary size %d." % vocab_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))
    
    sorted_vocab = sorted(vocab, key=operator.itemgetter(1), reverse=True)
    index_to_word = [x[0] for x in sorted_vocab] + [unk_token]
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    
    # Replace all words not in our vocabulary with the unknown token
    print("\nReplacing unknown words...")
    start = time.time()
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unk_token for w in sent]
    print("Replaced unknown words the unknown token in %.2f minutes." % ((time.time()-start)/60.0))

    # Create the training data
    print("\nCreating the training data as an array...")
    start = time.time()
    data_array = np.asarray([[word_to_index[w] for w in sent] for sent in tokenized_sentences])
    print("Constructed training data array in %.2f minutes." % ((time.time()-start)/60.0))

    return data_array, word_to_index, index_to_word


def get_batch_lists(dataset, min_batch_size=10, max_batch_size=100):

    print("Gathering sentence lengths...")
    start = time.time()
    lengths = []
    for sent in dataset:
        lengths.append(len(sent))  
    print("Computed all lengths in %.2f minutes." % ((time.time()-start)/60.0))
    
    print("\nPlotting sentence-length distribution...")
    fig = plt.figure(figsize=(12,6))
    # set bin widths to 1
    BINS = range(min(lengths), 200, 1)
    plt.hist(lengths, bins=BINS)
    plt.title("Histogram of sentence lengths in dataset")
    plt.xlabel("Sentence Length")
    plt.ylabel("Frequency")
    plt.show()

    dataset_DF = pd.DataFrame({'Length': lengths, 'Sentence': dataset})
    
    print("\nFiltering out sentences with rare lengths (if there are fewer than %d in dataset)" % min_batch_size)
    start = time.time()
    filtered_lengths = []
    counter_lengths = Counter(lengths)
    all_lengths = np.asarray(list(counter_lengths.keys()))
    np.random.shuffle(all_lengths)
    for ell in all_lengths:
        if counter_lengths[ell] >= min_batch_size:
            filtered_lengths.append(ell)
    print("Filtered all lengths in %.2f minutes." % ((time.time()-start)/60.0))

    print("\nBuilding batch lists...")
    start = time.time()
    batch_lists = []
    batch_lengths = []
    for ell in filtered_lengths:
        sent_indices = np.asarray(dataset_DF[dataset_DF['Length']==ell].index.values)
        np.random.shuffle(sent_indices)
        BS_uncut = len(sent_indices)
        if BS_uncut > max_batch_size:
            N_batches_ell = np.int_(np.floor(np.true_divide(BS_uncut,max_batch_size)))
            for i in range(N_batches_ell):
                batch_lists.append(sent_indices[i*max_batch_size:(i+1)*max_batch_size])
                batch_lengths.append(max_batch_size)
            if ((np.true_divide(BS_uncut,max_batch_size)-N_batches_ell)*max_batch_size) >= min_batch_size:
                batch_lists.append(sent_indices[N_batches_ell*max_batch_size:BS_uncut])
                batch_lengths.append(BS_uncut-N_batches_ell*max_batch_size)
        else:
            batch_lists.append(sent_indices)
            batch_lengths.append(BS_uncut)

    np.random.shuffle(batch_lists)
            
    print("Found %d batches with batch sizes between %d and %d in %.2f minutes." % (len(batch_lists), np.min(batch_lengths), np.max(batch_lengths), (time.time()-start)/60.0))
    print("The shortest sentences have %d words, and the longest have %d" % (np.min(filtered_lengths), np.max(filtered_lengths)))

    print("\nPlotting batch-size distribution...")
    fig = plt.figure(figsize=(12,6))
    # set bin widths to 1
    BINS = range(0, max_batch_size+1, 1)
    plt.hist(batch_lengths, bins=BINS)
    plt.title("Histogram of batch sizes processed from dataset")
    plt.xlabel("Batch size")
    plt.ylabel("Frequency")
    plt.yscale('log', nonposy='clip')
    plt.show()
    
    return batch_lists

    
def get_test_train_XY_split(filename="books_merged.txt", vocab_size=5000, max_N_sentences=100000, start_token="SENTENCE_START", end_token="SENTENCE_END", unk_token="UNKNOWN_TOKEN", train_split=0.8, min_batch_size=10, max_batch_size=100):
    
    data_array, w_to_i, i_to_w = load_data(filename, vocab_size, max_N_sentences, start_token, end_token, unk_token)
    
    batch_list = get_batch_lists(data_array, min_batch_size, max_batch_size)
    
    mask = np.random.rand(len(batch_list)) < 0.8
    batch_list_train = np.asarray(batch_list)[mask]
    batch_list_test = np.asarray(batch_list)[~mask]
    
    X_train=[]; X_test=[]; Y_train=[]; Y_test=[]
    
    for batch in batch_list_train:
        array = np.vstack(data_array[batch])
        X_train.append(np.delete(array, (-1), axis=1))
        Y_train.append(np.hstack(np.delete(array, (0), axis=1)))
        
    for batch in batch_list_test:
        array = np.vstack(data_array[batch])
        X_test.append(np.delete(array, (-1), axis=1))
        Y_test.append(np.hstack(np.delete(array, (0), axis=1)))
        
    return X_train, X_test, Y_train, Y_test, w_to_i, i_to_w
