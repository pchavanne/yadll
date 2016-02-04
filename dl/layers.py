# -*- coding: UTF-8 -*-
import timeit

from .init import *
from .objectives import *
from .updates import *
from .utils import *

# from theano.tensor.shared_randomstreams import RandomStreams
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

T_rng = RandomStreams(np_rng.randint(2 ** 30))


class Layer(object):
    def __init__(self, incoming, name=None):
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape
            self.input_layer = incoming

        self.name = name
        self.params = []
        self.reguls = 0

    def get_reguls(self):
        return self.reguls

    @property
    def output_shape(self):
        return self.input_shape

    def get_output(self, **kwargs):
        raise NotImplementedError


class InputLayer(Layer):
    def __init__(self, shape, input_var=None, **kwargs):
        super(InputLayer, self).__init__(shape, **kwargs)
        self.input = input_var

    def get_output(self, **kwargs):
        return self.input


class ReshapeLayer(Layer):
    def __init__(self, incoming, output_shape=None, **kwargs):
        super(ReshapeLayer, self).__init__(incoming, **kwargs)
        self.reshape_shape = output_shape

    @property
    def output_shape(self):
        return self.reshape_shape

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.reshape(self.reshape_shape)


class FlattenLayer(Layer):
    def __init__(self, incoming, ndim=1, **kwargs):
        super(FlattenLayer, self).__init__(incoming, **kwargs)
        self.ndim = ndim

    @property
    def output_shape(self):
        return self.input_shape[0], np.prod(self.input_shape[1:])

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return X.flatten(self.ndim)


class DenseLayer(Layer):
    def __init__(self, incoming, nb_units, W=glorot_uniform, b=constant,
                 activation=tanh, l1=None, l2=None, **kwargs):
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.shape = (self.input_shape[1], nb_units)
        if isinstance(W, theano.compile.SharedVariable):
            self.W = W
        else:
            self.W = initializer(W, shape=self.shape, name='W')
        self.params.append(self.W)
        if isinstance(b, theano.compile.SharedVariable):
            self.b = b
        else:
            self.b = initializer(b, shape=(self.shape[1],), name='b')
        self.params.append(self.b)
        self.activation = activation
        if l1:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return self.input_shape[0], self.shape[1]

    def get_output(self, **kwargs):
        X = self.input_layer.get_output(**kwargs)
        return self.activation(T.dot(X, self.W) + self.b)


class UnsupervisedLayer(DenseLayer):
    def __init__(self, incoming, nb_units, hyperparameters, **kwargs):
        super(UnsupervisedLayer, self).__init__(incoming, nb_units, **kwargs)
        self.hp = hyperparameters
        self.unsupervised_params = list(self.params)

    def get_encoded_input(self, stochastic=False, **kwargs):
        raise NotImplementedError

    def get_unsupervised_cost(self, stochastic=False, **kwargs):
        raise NotImplementedError

    @timer(' Pretraining the layer')
    def unsupervised_training(self, x, train_set_x):
        print '... Pretraining the layer: %s' % self.name
        index = T.iscalar('index')
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] / self.hp.batch_size
        cost = self.get_unsupervised_cost(stochastic=True)
        updates = sgd_updates(cost, self.unsupervised_params, self.hp.learning_rate)
        pretrain = theano.function(inputs=[index], outputs=cost, updates=updates,
                                   givens={x: train_set_x[index * self.hp.batch_size: (index + 1) * self.hp.batch_size]})
        for epoch in xrange(self.hp.n_epochs):
            c = []
            for minibatch_index in xrange(n_train_batches):
                c.append(pretrain(minibatch_index))
            print 'Layer: %s, pretraining epoch %d, cost %d' % (self.name, epoch, np.mean(c))


class LogisticRegression(DenseLayer):
    def __init__(self, incoming, nb_class, W=constant, activation=softmax, **kwargs):
        super(LogisticRegression, self).__init__(incoming, nb_class, W=W,
                                                 activation=activation, **kwargs)


class Dropout(Layer):
    def __init__(self, incoming, corruption_level=0.5, **kwargs):
        super(Dropout, self).__init__(incoming, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        return X


class Dropconnect(DenseLayer):
    def __init__(self, incoming, nb_units, corruption_level=0.5, **kwargs):
        super(Dropconnect, self).__init__(incoming, nb_units, **kwargs)
        self.p = 1 - corruption_level

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            self.W = self.W * T_rng.binomial(self.shape, n=1, p=self.p, dtype=floatX)
        return self.activation(T.dot(X, self.W) + self.b)


class PoolLayer(Layer):
    def __init__(self, incoming, poolsize, stride=None, ignore_border=True,
                 padding=(0, 0), mode='max', **kwargs):
        super(PoolLayer, self).__init__(incoming, **kwargs)
        self.poolsize = poolsize
        self.stride = stride    # If st is None, it is considered equal to ds
        self.ignore_border = ignore_border
        self.padding = padding
        self.mode = mode    # {'max', 'sum', 'average_inc_pad', 'average_exc_pad'}

    def pool(self, input, ds):
        return downsample.max_pool_2d(input=input, ds=ds, st=self.stride, ignore_border=self.ignore_border,
                                      padding=self.padding, mode=self.mode)

    @property
    def output_shape(self):
        return

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        return self.pool(input=X, ds=self.poolsize)


class ConvLayer(Layer):
    def __init__(self, incoming, image_shape=None, filter_shape=None, W=glorot_uniform, b=constant,
                 border_mode='valid', subsample=(1, 1), l1=None, l2=None, **kwargs):
        super(ConvLayer, self).__init__(incoming, **kwargs)
        assert image_shape[1] == filter_shape[1]
        self.image_shape = image_shape      # (batch size, num input feature maps, image height, image width)
        self.filter_shape = filter_shape    # (number of filters, num input feature maps, filter height, filter width)
        self.border_mode = border_mode      # {'valid', 'full'}
        self.subsample = subsample
        self.fan = (np.prod(filter_shape[1:]), filter_shape[0] * np.prod(filter_shape[2:]))
        self.W = initializer(W, shape=self.filter_shape, fan=self.fan, name='W')
        self.params.append(self.W)
        self.b = initializer(b, shape=(self.filter_shape[0],), name='b')
        self.params.append(self.b)
        if l1:
            self.reguls += l1 * T.mean(T.abs_(self.W))
        if l2:
            self.reguls += l2 * T.mean(T.sqr(self.W))

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.input_shape[1],
                self.image_shape[2] - self.filter_shape[2] + 1,
                self.image_shape[3] - self.filter_shape[3] + 1)

    def conv(self, input, filters, image_shape, filter_shape):
        return conv.conv2d(input=input, filters=filters, image_shape=image_shape,
                           filter_shape=filter_shape, border_mode=self.border_mode, subsample=self.subsample)

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        return self.conv(input=X, filters=self.W, image_shape=self.image_shape,
                         filter_shape=self.filter_shape)


class ConvPoolLayer(ConvLayer, PoolLayer):
    def __init__(self, incoming, poolsize, image_shape=None, filter_shape=None,
                 activation=tanh, **kwargs):
        super(ConvPoolLayer, self).__init__(incoming, poolsize=poolsize, image_shape=image_shape,
                                            filter_shape=filter_shape, **kwargs)
        self.activation = activation

    @property
    def output_shape(self):
        return (self.input_shape[0],
                self.input_shape[1],
                self.image_shape[2] - self.filter_shape[2] + 1,
                self.image_shape[3] - self.filter_shape[3] + 1)

    def get_output(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        conv_X = self.conv(input=X, filters=self.W, image_shape=self.image_shape,
                           filter_shape=self.filter_shape)
        pool_X = self.pool(input=conv_X, ds=self.poolsize)
        return self.activation(pool_X + self.b.dimshuffle('x', 0, 'x', 'x'))


class AutoEncoder(UnsupervisedLayer):
    def __init__(self, incoming, nb_units, hyperparameters, corruption_level=0.5,
                 W=(glorot_uniform, {'gain': sigmoid}), b_prime=constant, **kwargs):
        super(AutoEncoder, self).__init__(incoming, nb_units, hyperparameters, W=W, **kwargs)
        self.W_prime = self.W.T
        if isinstance(b_prime, theano.compile.SharedVariable):
            self.b_prime = b_prime
        else:
            self.b_prime = initializer(b_prime, shape=(self.shape[0],), name='b_prime')
        self.unsupervised_params.append(self.b_prime)
        self.p = 1 - corruption_level

    def get_encoded_input(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        if self.p > 0 and stochastic:
            X = X * T_rng.binomial(self.input_shape, n=1, p=self.p, dtype=floatX)
        Y = sigmoid(T.dot(X, self.W) + self.b)
        Z = sigmoid(T.dot(Y, self.W_prime) + self.b_prime)
        return Z

    def get_unsupervised_cost(self, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        Z = self.get_encoded_input(stochastic=stochastic, **kwargs)
        cost = T.mean(categorical_crossentropy(Z, X))
        return cost


class RBM(UnsupervisedLayer):
    def __init__(self, incoming, nb_units, hyperparameters, W=glorot_uniform,
                 b_hidden=constant, activation=sigmoid, **kwargs):
        super(RBM, self).__init__(incoming, nb_units, hyperparameters, W=W,
                                  activation=activation, **kwargs)
        if isinstance(b_hidden, theano.compile.SharedVariable):
            self.b_hidden = b_hidden
        else:
            self.b_hidden = initializer(b_hidden, shape=(self.shape[0],), name='b_hidden')
        self.unsupervised_params.append(self.b_prime)

    def free_energy(self, v_sample):
        wx_b = T.dot(v_sample, self.W) + self.b_hidden
        vbias_term = T.dot(v_sample, self.b)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def prop_up(self, vis):
        pre_sigmoid_activation = T.dot(vis, self.W) + self.b_hidden
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_h_given_v(self, v0_sample):
        pre_sigmoid_h1, h1_min = self.prop_up(v0_sample)
        h1_sample = T_rng.binomial(size=h1_min.shape, n=1, p=h1_min, dtype=floatX)
        return [pre_sigmoid_h1, h1_min, h1_sample]

    def prop_down(self, hid):
        pre_sigmoid_activation = T.dot(hid, self.W) + self.b
        return [pre_sigmoid_activation, sigmoid(pre_sigmoid_activation)]

    def sample_v_given_h(self, h0_sample):
        pre_sigmoid_v1, v1_min = self.prop_down(h0_sample)
        v1_sample = T_rng.binomial(size=v1_min.shape, n=1, p=v1_min, dtype=floatX)
        return [pre_sigmoid_v1, v1_min, v1_sample]

    def gibbs_hvh(self, h0_sample):
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h0_sample)
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v1_sample)
        return [pre_sigmoid_v1, v1_mean, v1_sample, pre_sigmoid_h1, h1_mean, h1_sample]

    def gibbs_vhv(self, v0_sample):
        pre_sigmoid_h1, h1_mean, h1_sample = self.sample_h_given_v(v0_sample)
        pre_sigmoid_v1, v1_mean, v1_sample = self.sample_v_given_h(h1_sample)
        return [pre_sigmoid_h1, h1_mean, h1_sample, pre_sigmoid_v1, v1_mean, v1_sample]

    def get_reconstruction_cost(self, updates, pre_sigmoid_nv, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        cross_entropy = T.mean(T.sum(X * T.log(sigmoid(pre_sigmoid_nv)) +
                (1 - X) * T.log(1 - sigmoid(pre_sigmoid_nv)), axis=1))
        # c = binary_crossentropy(X, sigmoid(pre_sigmoid_nv))
        # TODO compare the two cross entropies and check without updates
        return cross_entropy

    def get_pseudo_likelihood_cost(self, updates, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        bit_i_idx = theano.shared(value=0, name='bit_i_idx')
        xi = T.round(X)
        fe_xi = self.free_energy(xi)
        xi_flip = T.set_subtensor(xi[:, bit_i_idx], 1 - xi[:, bit_i_idx])
        fe_xi_flip = self.free_energy(xi_flip)
        cost = T.mean(self.shape[0] * T.log(T.nnet.sigmoid(fe_xi_flip - fe_xi)))
        updates[bit_i_idx] = (bit_i_idx + 1) % self.shape[0]
        return cost

    def get_encoded_input(self, stochastic=False, **kwargs):
        raise NotImplementedError

    def get_unsupervised_cost(self, persistent=None, k=1, stochastic=False, **kwargs):
        X = self.input_layer.get_output(stochastic=stochastic, **kwargs)
        pre_sigmoid_ph, ph_mean, ph_sample = self.sample_h_given_v(X)
        if persistent is None:
            chain_start = ph_sample
        else:
            chain_start = persistent
        ([pre_sigmoid_nvs, nv_means, nv_samples, pre_sigmoid_nhs, nh_means, nh_samples], updates) = \
            theano.scan(fn=self.gibbs_hvh,
                        outputs_info=[None, None, None, None, None, chain_start],
                        n_steps=k)
        chain_end = nv_samples[-1]
        cost = T.mean(self.free_energy(X)) - T.mean(self.free_energy(chain_end))
        gparams = T.grad(cost, self.params, consider_constant=[chain_end])
        for gparam, param in zip(gparams, self.unsupervised_params):
            updates[param] = param - gparam * T.cast(self.hp.learning_rate, dtype=floatX)
        if persistent:
            updates[persistent] = nh_samples[-1]
            monitoring_cost = self.get_pseudo_likelihood_cost(updates)
        else:
            monitoring_cost = self.get_reconstruction_cost(updates, pre_sigmoid_nvs[-1])
        return monitoring_cost, updates


class LSTM(Layer):
    def __init__(self, incoming, **kwargs):
        super(LSTM, self).__init__(self, incoming, **kwargs)
        # TODO implement LSTM class

    def get_output(self, **kwargs):
        raise NotImplementedError

