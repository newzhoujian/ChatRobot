import numpy as np
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


# weight initializer, normal by default
def norm_weight(nin, nout=None, scale=0.01, ortho=False):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * np.random.randn(nin, nout)
    return W.astype('float32')
def uniform_weight(size,scale=0.1):
    return np.random.uniform(size=size,low=-scale, high=scale).astype(theano.config.floatX)


def glorot_uniform(size):
    fan_in, fan_out = size
    s = np.sqrt(6. / (fan_in + fan_out))
    return np.random.uniform(size=size,low=-s, high=s).astype(theano.config.floatX)


class GRU(object):
    def __init__(self, n_in, n_hidden, n_out, activation=T.tanh,inner_activation=T.nnet.sigmoid,
                 output_type='real',batch_size=200):

        self.activation = activation
        self.inner_activation = inner_activation
        self.output_type = output_type

        self.batch_size = batch_size
        self.n_hidden = n_hidden

        # recurrent weights as a shared variable
        self.U_z = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_z = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_z = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_r = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_r = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_r = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)

        self.U_h = theano.shared(ortho_weight(n_hidden),borrow=True)
        self.W_h = theano.shared(glorot_uniform((n_in,n_hidden)),borrow=True)
        self.b_h = theano.shared(value=np.zeros((n_hidden,),dtype=theano.config.floatX),borrow=True)


        self.params = [self.W_z,self.W_h,self.W_r,
                       self.U_h,self.U_r,self.U_z,
                       self.b_h,self.b_r,self.b_z]

    def __call__(self, input,input_lm=None, return_list = False, Init_input =None,check_gate = False):
         # activation function
        if Init_input == None:
            init = theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)
        else:
            init = Init_input

        if check_gate:
            self.h_l, _ = theano.scan(self.step3,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=[init, theano.shared(value=np.zeros((self.batch_size,self.n_hidden),
                                                                  dtype=theano.config.floatX),borrow=True)])
            return [self.h_l[0][:,-1,:], self.h_l[1]]



        if input_lm == None:
            self.h_l, _ = theano.scan(self.step2,
                        sequences=input.dimshuffle(1,0,2),
                        outputs_info=init)
        else:
            self.h_l, _ = theano.scan(self.step,
                        sequences=[input.dimshuffle(1,0,2),T.addbroadcast(input_lm.dimshuffle(1,0,'x'), -1)],
                        outputs_info=init)
        self.h_l = self.h_l.dimshuffle(1,0,2)
        if return_list == True:
            return self.h_l
        return self.h_l[:,-1,:]

    def step2(self,x_t, h_tm1):
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        return h
    def step3(self,x_t,mask, h_tm1, gate_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return [h,r]

    def step(self,x_t,mask, h_tm1):
        #h_tm1 =  mask * h_tm1
        x_z = T.dot(x_t, self.W_z) + self.b_z
        x_r = T.dot(x_t, self.W_r) + self.b_r
        x_h = T.dot(x_t, self.W_h) + self.b_h
        z = self.inner_activation(x_z + T.dot(h_tm1, self.U_z))
        r = self.inner_activation(x_r + T.dot(h_tm1, self.U_r))

        hh = self.activation(x_h + T.dot(r * h_tm1, self.U_h))
        h = z * h_tm1 + (1 - z) * hh
        h = mask * h + (1-mask) * h_tm1

        return h
