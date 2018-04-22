from CNN import LeNetConvPoolLayer2
import theano
import theano.tensor as T
import numpy as np
def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')


class HiddenLayer2(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.activation = activation

        self.params = [self.W, self.b]

    def __call__(self, input):
        lin_output = T.dot(input, self.W) + self.b
        return self.activation(lin_output)

class ConvSim(object):
    def __init__(self, rng,  n_in, n_out, W=None, b=None,
                 activation=T.tanh,hidden_size=100):
        self.W = theano.shared(value=ortho_weight(hidden_size), borrow=True)
        self.activation = activation

        self.conv_layer = LeNetConvPoolLayer2(rng,filter_shape=(8,2,3,3),
                                    image_shape=(200,2,50,50)
                       ,poolsize=(3,3),non_linear='relu')

        self.hidden_layer = HiddenLayer2(rng,2048,n_out)
        self.params = [self.W,] + self.conv_layer.params + self.hidden_layer.params
    def Get_M2(self, input_l, input_r):
        return T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))

    def __call__(self, origin_l,origin_r,input_l,input_r):
        channel_1 = T.batched_dot(origin_l,origin_r.dimshuffle(0,2,1))
        channel_2 = T.batched_dot(T.dot(input_l,self.W),input_r.dimshuffle(0,2,1))
        input = T.stack([channel_1,channel_2],axis=1)
        mlp_in = T.flatten(self.conv_layer(input),2)

        return self.hidden_layer(mlp_in)

