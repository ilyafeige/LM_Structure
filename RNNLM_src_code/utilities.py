from collections import Counter
import os
import time
import random

import numpy as np
import theano
import theano.tensor as T
try:
    import cPickle as pickle
except:
    import pickle
#import pandas as pd

import copy



def matrix_init(fan_in, fan_out, constant=1):
    """ Xavier initialization of network weights"""
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return theano.shared(np.random.uniform(size=(fan_in, fan_out),
                                           low=low, high=high,
                                           ).astype(theano.config.floatX))


def offset_init(out_size, constant=1):
    """ Xavier initialization of network weights"""
    normalise = constant*np.sqrt(6.0/out_size)
    return theano.shared(normalise*np.random.randn(out_size).astype(theano.config.floatX))


def to_one_hot(y, nb_class, dtype=None):
    """
    Return a matrix where each row correspond to the one hot
    encoding of each element in y.
    Parameters
    ----------
    y
        A tensor of integer value between 0 and nb_class - 1.
    nb_class : int
        The number of class in y.
    dtype : data-type
        The dtype of the returned matrix. Default floatX.
    Returns
    -------
    object
        A matrix of shape (y.shape[0], y.shape[1], nb_class),
        where the i,j element is the one hot encoding of y[i,j]
    """
    ret = theano.tensor.zeros((y.shape[0], y.shape[1], nb_class),
                              dtype=dtype)

    # Get the slicing indices
    inds_0 = theano.tensor.arange(y.shape[0]).dimshuffle([0, 'x'])
    inds_1 = theano.tensor.arange(y.shape[1]).dimshuffle(['x', 0])
    ret = theano.tensor.set_subtensor(ret[inds_0,
                                          inds_1,
                                          y],
                                      1)
    return ret
    

def adam(loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-8, gamma=1-1e-8):
    """
    ADAM update rules
    Default values are taken from [Kingma2014]
    References:
    [Kingma2014] Kingma, Diederik, and Jimmy Ba.
    "Adam: A Method for Stochastic Optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    http://arxiv.org/pdf/1412.6980v4.pdf
    """
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate
    t = theano.shared(np.float32(1))
    b1_t = b1*gamma**(t-1)   #(Decay the first moment running average coefficient)

    for theta_previous, g in zip(all_params, all_grads):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape,
                                            dtype=theano.config.floatX))

        m = b1_t*m_previous + (1 - b1_t)*g         # (Update biased first moment estimate)
        v = b2*v_previous + (1 - b2)*g**2          # (Update biased second raw moment estimate)
        m_hat = m / (1-b1**t)                      # (Compute bias-corrected first moment estimate)
        v_hat = v / (1-b2**t)                      # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)

        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
        
    updates.append((t, t + 1.))
    return updates    

def adam2(cost, params, learning_rate=0.001, b1=0.1, b2=0.001, e=1e-8, gamma=1-1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = learning_rate * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates