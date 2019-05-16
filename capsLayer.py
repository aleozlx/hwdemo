"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import numpy as np
import tensorflow as tf

epsilon = 1e-9
cfg_stddev = 0.01

class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.

    Returns:
        A 4-D tensor.
    '''
    # def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
    #     self.num_outputs = num_outputs
    #     self.vec_len = vec_len
    #     self.with_routing = with_routing
    #     self.layer_type = layer_type
    def __init__(self, num_outputs, vec_len, iter_routing=3, batch_size=16, input_shape=None, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.iter_routing = iter_routing
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.iter_routing > 0:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                # assert input.get_shape() == [cfg.batch_size, 20, 20, 256]
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                capsules = tf.reshape(capsules, (self.batch_size, -1, self.vec_len, 1))

                # return tensor with shape [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                return(capsules)

        if self.layer_type == 'FC':
            if self.iter_routing > 0:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(self.batch_size, self.input_shape[1], 1, self.input_shape[2], 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([self.batch_size, self.input_shape[1], self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ, self.iter_routing, num_outputs=self.num_outputs, num_dims=self.vec_len)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ, iter_routing, num_outputs=10, num_dims=16):
    ''' The routing algorithm.

    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
        num_outputs: the number of output capsules.
        num_dims: the number of dimensions for output capsule.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    input_shape = get_shape(input)
    W = tf.get_variable('Weight', shape=[1, input_shape[1], num_dims * num_outputs] + input_shape[-2:],
                        dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=cfg_stddev))
    biases = tf.get_variable('bias', shape=(1, 1, num_outputs, num_dims, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, num_dims * num_outputs, 1, 1])
    # assert input.get_shape() == [cfg.batch_size, 1152, 160, 8, 1]

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], num_outputs, num_dims, 1])
    # assert u_hat.get_shape() == [cfg.batch_size, 1152, 10, 16, 1]

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                # assert s_J.get_shape() == [cfg.batch_size, 1, num_outputs, num_dims, 1]

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
                # assert v_J.get_shape() == [cfg.batch_size, 1, 10, 16, 1]
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)
                # assert u_produce_v.get_shape() == [cfg.batch_size, 1152, 10, 1, 1]

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return v_J

# import tensorflow.contrib.slim as slim
# # def routing(input, b_IJ, iter_routing, num_outputs=10, num_dims=16):
# cfg_ac_lambda0 = 0.01
# def em_routing(votes, activation, iter_routing, num_outputs, num_dims):
#     weights_regularizer = tf.contrib.layers.l2_regularizer(5e-04)
#     test = []

#     batch_size = int(votes.get_shape()[0])
#     caps_num_i = int(activation.get_shape()[1])
#     n_channels = int(votes.get_shape()[-1])

#     sigma_square = []
#     miu = []
#     activation_out = []
#     beta_v = slim.variable('beta_v', shape=[num_outputs, n_channels], dtype=tf.float32,
#                            initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
#                            regularizer=weights_regularizer)
#     beta_a = slim.variable('beta_a', shape=[num_outputs], dtype=tf.float32,
#                            initializer=tf.constant_initializer(0.0),#tf.truncated_normal_initializer(mean=0.0, stddev=0.01),
#                            regularizer=weights_regularizer)

#     # votes_in = tf.stop_gradient(votes, name='stop_gradient_votes')
#     # activation_in = tf.stop_gradient(activation, name='stop_gradient_activation')
#     votes_in = votes
#     activation_in = activation

#     for iters in range(iter_routing):
#         # if iters == cfg.iter_routing-1:

#         # e-step
#         if iters == 0:
#             r = tf.constant(np.ones([batch_size, caps_num_i, num_outputs], dtype=np.float32) / num_outputs)
#         else:
#             # Contributor: Yunzhi Shi
#             # log and exp here provide higher numerical stability especially for bigger number of iterations
#             log_p_c_h = -tf.log(tf.sqrt(sigma_square)) - \
#                         (tf.square(votes_in - miu) / (2 * sigma_square))
#             log_p_c_h = log_p_c_h - \
#                         (tf.reduce_max(log_p_c_h, axis=[2, 3], keep_dims=True) - tf.log(10.0))
#             p_c = tf.exp(tf.reduce_sum(log_p_c_h, axis=3))

#             ap = p_c * tf.reshape(activation_out, shape=[batch_size, 1, num_outputs])

#             # ap = tf.reshape(activation_out, shape=[batch_size, 1, num_outputs])

#             r = ap / (tf.reduce_sum(ap, axis=2, keep_dims=True) + epsilon)

#         # m-step
#         r = r * tf.reshape(activation_in, tf.shape(r))
#         r = r / (tf.reduce_sum(r, axis=2, keep_dims=True)+epsilon)

#         r_sum = tf.reduce_sum(r, axis=1, keep_dims=True)
#         r1 = tf.reshape(r / (r_sum + epsilon),
#                         shape=[batch_size, caps_num_i, num_outputs, 1])

#         miu = tf.reduce_sum(votes_in * r1, axis=1, keep_dims=True)
#         sigma_square = tf.reduce_sum(tf.square(votes_in - miu) * r1,
#                                      axis=1, keep_dims=True) + epsilon

#         if iters == iter_routing-1:
#             r_sum = tf.reshape(r_sum, [batch_size, num_outputs, 1])
#             cost_h = (beta_v + tf.log(tf.sqrt(tf.reshape(sigma_square,
#                                                          shape=[batch_size, num_outputs, n_channels])))) * r_sum

#             activation_out = tf.nn.softmax(cfg_ac_lambda0 * (beta_a - tf.reduce_sum(cost_h, axis=2)))
#         else:
#             activation_out = tf.nn.softmax(r_sum)
#         # if iters <= cfg.iter_routing-1:
#         #     activation_out = tf.stop_gradient(activation_out, name='stop_gradient_activation')
#     return activation_out
#     # return miu, activation_out, test

def squash(vector):
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)

def get_shape(inputs, name=None):
    name = "shape" if name is None else name
    with tf.name_scope(name):
        static_shape = inputs.get_shape().as_list()
        dynamic_shape = tf.shape(inputs)
        shape = []
        for i, dim in enumerate(static_shape):
            dim = dim if dim is not None else dynamic_shape[i]
            shape.append(dim)
        return(shape)
