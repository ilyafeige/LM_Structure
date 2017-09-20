import numpy as np
from matplotlib import pyplot as plt         
import pandas as pd


# For large n_steps, this will take a while (quadratic time in n_steps)
def h_iterator(model, n_steps, sentence=[]):
    start_time = time.time()
    hs = []
    for n in range(1,n_steps+1):
        hs.append(model.iterate_h(n, sentence=sentence))
    print("Completed h iteration over "+str(n_steps)+
          " steps in %.2f minutes" % ((time.time() - start_time)/60.))
    return hs

def show_h_norm(hs):
    norms = [np.linalg.norm(hs[n]) for n in range(len(hs))]
    plt.plot(np.arange(len(hs)), norms)    
    plt.show()
    
def compute_xyarea(hs, n_start=0, n_end=None, rand_h1=None, rand_h2=None):
    
    if n_end is None:
        n_end = len(hs)
    xs = []
    ys = []
    area = []
    if rand_h1 is None:
        rand_h1 = np.random.uniform(-1,1,len(hs[0]))
    if rand_h2 is None:        
        rand_h2 = np.random.uniform(-1,1,len(hs[0]))
    for n in range(n_start, n_end):
        xs.append(np.linalg.norm(hs[n]-rand_h1))
        ys.append(np.linalg.norm(hs[n]-rand_h2))
        area.append((n_end-n_start+0.0)/(n+1.))

    return np.asarray(xs), np.asarray(ys), np.asarray(area)
    
def show_single_2D(hs, title="h iteration", n_start=0, n_end=None, 
                   rand_h1=None, rand_h2=None, sentence_len=None):    

    xs, ys, area = compute_xyarea(hs, n_start=n_start, n_end=n_end, 
                                  rand_h1=rand_h1, rand_h2=rand_h2)

    plt.figure(figsize=(8,6))
    
    if sentence_len is None:
        plt.scatter(xs,ys,s=area)
    else:
        for n in range(sentence_len):
            mask = np.arange(n_start + n, len(xs), sentence_len)
            word_number = (n_start + n) % sentence_len
            plt.scatter(xs[mask],ys[mask],s=area[mask], label="word "+str(word_number))
        plt.legend()
        
    plt.title(title, fontsize=14)
    plt.show()    
    
def show_multiple_2D(hs_list, title="h iteration", n_start=0, 
                      n_end=None, rand_h1=None, rand_h2=None):    

    if n_end is None:
        n_end = len(hs_list[0])

    if rand_h1 is None:
        rand_h1 = np.random.uniform(-1,1,len(hs_list[0][0]))
    if rand_h2 is None:        
        rand_h2 = np.random.uniform(-1,1,len(hs_list[0][0]))        
    
    plt.figure(figsize=(8,6))
    
    for sent_n, hs in enumerate(hs_list):
        xs, ys, area = compute_xyarea(hs, n_start=n_start, n_end=n_end, 
                                      rand_h1=rand_h1, rand_h2=rand_h2)
        plt.scatter(xs, ys, s=area, label="sentence "+str(sent_n)) 

    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()   
    
def show_one_iteration_2D(hs_list, sentence_list,
                          title="One iteration through each sentence", 
                          n_start=0, rand_h1=None, rand_h2=None):    

    if rand_h1 is None:
        rand_h1 = np.random.uniform(-1,1,len(hs_list[0][0]))
    if rand_h2 is None:        
        rand_h2 = np.random.uniform(-1,1,len(hs_list[0][0]))        
    
    plt.figure(figsize=(8,6))
    
    for sent_n, hs in enumerate(hs_list):
        n_end = len(sentence_list[sent_n])
        xs, ys, area = compute_xyarea(hs, n_start=n_start, n_end=n_end, 
                                      rand_h1=rand_h1, rand_h2=rand_h2)
        area = np.multiply(area[::-1], 20.)
        area[-1] = 3.*area[-1]
        plt.scatter(xs, ys, s=area, label="sentence "+str(sent_n)) 

    plt.title(title, fontsize=14)
    plt.legend()
    plt.show()       
    
def save_hs(hs, filename="hs.pkl"):    
    # the rows are the h's
    save_dir = "hs/"
    df = pd.DataFrame(hs)
    df.to_pickle(save_dir+filename)  # where to save it, usually as a .pkl
    
def load_hs(filename):    
    # the rows are the h's
    save_dir = "hs/"
    df = pd.read_pickle(save_dir+filename)
    return df

def compute_ten_hs(model, sentences, n_h_iterations=10, filename_pre="hs_test"):
    
    hs_nosent = h_iterator(model, n_h_iterations, sentence=[])
    save_hs(hs_nosent, filename=filename_pre+"_nosent.pkl")
    hs_sent = []
    for n, sentence in enumerate(sentences):
        h = h_iterator(model, n_h_iterations, sentence=sentence)
        hs_sent.append(h)
        save_hs(h, filename=filename_pre+"_sent"+str(n)+".pkl")
        
    return hs_nosent, hs_sent

def load_ten_hs(filename_pre="hs_test", n_sent=10):
    
    df = load_hs(filename_pre+"_nosent.pkl")
    hs_nosent = df.values
    hs_sent = []
    for n in range(n_sent):
        df = load_hs(filename_pre+"_sent"+str(n)+".pkl")
        hs_sent.append(df.values)
        
    return hs_nosent, hs_sent