import cPickle, gzip, numpy
import theano
import theano.tensor as T

class LogisticRegression(object):
    def __init__(self,input,n_in,n_out,rng):
        self.W = theano.shared( numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            ))
        self.b = theano.shared(value=numpy.zeros(n_out,dtype=theano.config.floatX),borrow=True,name='b')
        self.predict_prob = T.nnet.softmax(T.dot(input,self.W)+self.b)
        self.predict_y = T.argmax(self.predict_prob,axis=1)
        self.params=[self.W,self.b]

    def negative_log_likelihood(self, y):
        #return - T.mean(y * T.log(self.predict_prob) + (1 - y) * T.log(1 - self.predict_prob))
        return -T.mean(T.log(self.predict_prob)[T.arange(y.shape[0]), y])

    def errors(self,y):
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.predict_y,y))
        else:
            raise NotImplementedError

def load_data(dataset):
    def shared_data(data_xy):
        data_x,data_y = data_xy
        shared_x = theano.shared(data_x)
        shared_y = theano.shared(data_y)
        return shared_x, T.cast(shared_y,'int32')

    f = gzip.open(dataset)
    train_set, dev_set, test_set = cPickle.load(f)
    f.close()

    train_set_x, train_set_y = shared_data(train_set)
    dev_set_x, dev_set_y = shared_data(dev_set)
    test_set_x, test_set_y = shared_data(test_set)

    rval = [(train_set_x,train_set_y),(dev_set_x, dev_set_y ),
    (test_set_x,test_set_y)]
    return rval
