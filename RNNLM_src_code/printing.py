""" Functions for printing sentences from modesl """
try:
    import cPickle as pickle
except:
    import pickle

import numpy as np


def print_sentence(s, index_to_word):
    sentence_str = [index_to_word[x] for x in s[1:]]
    return " ".join(sentence_str)


def print_sentences(model, num_sent, index_to_word, word_to_index,
                    max_attempts=10, save_dir=None, save_file=None, speed='fast'):

    if save_dir is None:
        save_dir = model.save_dir
    if save_file is None:
        save_file = model.save_file + "_sentences.txt"

    f = open(save_dir + save_file, 'a')
    for i in range(num_sent):
        sent = None
        counter = 0
        while not sent:
            sent = model.generate_sentence(word_to_index, speed=speed)
            counter += 1
            if counter > max_attempts:
                print("No valid sentence generated after %d attempts" % max_attempts)
                print("")
                break
        if counter > max_attempts:
            continue

        print(print_sentence(sent, index_to_word) + ' \n')
        f.write(print_sentence(sent, index_to_word) + ' \n')
        f.flush()
    f.close()
