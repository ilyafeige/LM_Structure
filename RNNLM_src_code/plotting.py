import numpy as np
from matplotlib import pyplot as plt

def plot_loss(loss_file):

    loss = []
    loss_file = open(loss_file, 'rb')
    for y in loss_file.readlines():
        loss.append(float(y))

    plt.figure(figsize=(16,4))
    plt.plot(loss)
    #plt.title('Loss after '+str(N_epochs)+' epochs for '+model.model_type, fontsize=16)
    plt.xlabel('Number of SGD steps', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

def get_dataset_loss(loss_file):

    losses = []
    loss_file = open(loss_file, 'r')
    for line in loss_file.readlines():
        losses.append([float(i) for i in line.split(', ')])
            
    losses = np.asarray(losses)    

    if losses.shape[1] == 2:
        loss = losses[:,0]
        perp = losses[:,1]        
        return loss, perp
        
    elif losses.shape[1] == 4:
        loss = losses[:,0]
        loss_pos = losses[:,1]
        perp = losses[:,2]
        perp_pos = losses[:,3]
        return loss, perp, loss_pos, perp_pos
    
    else:
        print("Loss file contains "+str(losses.shape[1])+" columns, not 2 or 4 as expected")
        return None
    
def plot_perplexity(loss_file1, loss_file2=None):

    losses1 = get_dataset_loss(loss_file1)
    loss1 = losses1[0]
    perp1 = losses1[1]
    
    if loss_file1[-14:-10] == "rain":
        label1="training data"
    elif loss_file1[-14:-10] == "test":
        label1="testing data"
    else:
        label1="first file"
        
    if loss_file2!=None:
        losses2 = get_dataset_loss(loss_file2)
        loss2 = losses2[0]
        perp2 = losses2[1]
        
        if loss_file2[-14:-10] == "rain":
            label2="training data"
        elif loss_file2[-14:-10] == "test":
            label2="testing data"
        else:
            label2="first file"    
        
    plt.figure(figsize=(16,4))
    plt.subplot(1,2,1)
    if loss_file2!=None:
        plt.plot(loss1,'-', label=label1)
        plt.plot(loss2,'-', label=label2)
    else:
        plt.plot(loss1,'-', label=label1)
    #plt.title('Loss after '+str(N_epochs)+' epochs for '+model.model_type, fontsize=16)
    plt.xlabel('Number of saves', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.subplot(1,2,2) 
    if loss_file2!=None:
        plt.plot(perp1,'-', label=label1)
        plt.plot(perp2,'-', label=label2)
    else:
        plt.plot(perp1,'-', label=label1)
    #plt.title('Loss after '+str(N_epochs)+' epochs for '+model.model_type, fontsize=16)
    plt.xlabel('Number of saves', fontsize=14)
    plt.ylabel('Perplexity', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend()
    plt.show()    
    
def plot_multiple(filenames, figsize=(18, 12)):
    
    plt.figure(figsize=figsize)
    datasets = ["train", "test"]
    metrics = ["loss", "perplexity"]
    for i, metric in enumerate(metrics): 
        plt.subplot(2,2,i+1)
        plt.title(metric, fontsize=14)
        plt.xlabel('number of saves', fontsize=14)
#        plt.ylabel(metric, fontsize=14)        
#        plt.yscale('log')
        for j, dataset in enumerate(datasets):
            for k, file in enumerate(filenames):
                filename = 'results/'+file+'_'+dataset+'_loss.save'
                losses = get_dataset_loss(filename)
                plot_metric = losses[i]
                plt.plot(np.arange(1,len(plot_metric)+1), plot_metric, '-', label=filename[8:-5])
        plt.legend()
    plt.show()    