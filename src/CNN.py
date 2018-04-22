import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv

def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)

class LeNetConvPoolLayer2(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """
        print 'image shape', image_shape
        print 'filter shape', filter_shape
        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /np.prod(poolsize))
        # initialize weights with random weights
        if self.non_linear=="none" or self.non_linear=="relu":
            self.W = theano.shared(np.asarray(rng.uniform(low=-0.01,high=0.01,size=filter_shape),
                                                dtype=theano.config.floatX),borrow=True,name="W_conv")
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX),borrow=True,name="W_conv")
        b_values =np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        self.params = [self.W, self.b]
        # convolve input feature maps with filters


    def __call__(self, input):
        conv_out = conv.conv2d(input=input, filters=self.W,filter_shape=self.filter_shape, image_shape=self.image_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        elif self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            self.output =theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True,mode="max")
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True,mode="max")
            self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return self.output


    def predict(self, new_data, batch_size):
        """
        predict for new data
        """
        img_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv.conv2d(input=new_data, filters=self.W, filter_shape=self.filter_shape, image_shape=img_shape)
        if self.non_linear=="tanh":
            conv_out_tanh = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        if self.non_linear=="relu":
            conv_out_tanh = ReLU(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
            output = theano.tensor.signal.pool.pool_2d(input=conv_out_tanh, ds=self.poolsize, ignore_border=True)
        else:
            pooled_out = theano.tensor.signal.pool.pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)
            output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        return output