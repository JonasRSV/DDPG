import tensorflow as tf
import numpy as np

class OrnsteinNoiseTensorflow(object):

    def __init__(self, delta, sigma, ou_a, ou_mu
                , noise_default=0, exploration=1
                , decay=2e-5):
        self.delta = tf.constant(delta, dtype=tf.float32)
        self.sigma = tf.constant(sigma, dtype=tf.float32)
        self.ou_a  = tf.constant(ou_a,  dtype=tf.float32)
        self.ou_mu = tf.constant(ou_mu, dtype=tf.float32)

        self.noise_default    = tf.constant(noise_default, dtype=tf.float32)
        self.noise_assign_ops = []

        self.exploration = tf.Variable(exploration, 
                                       dtype=tf.float32, 
                                       trainable=False)

        self.decay       = tf.constant(1 - decay, dtype=tf.float32)

        self.exploration_update_op =\
                self.exploration.assign(self.exploration * self.decay)

    def brownian_motion(self, noise):
        sqrt_delta_sigma = tf.sqrt(self.delta) * self.sigma
        randomness = tf.random_normal(noise.shape, 
                                      mean=0, 
                                      stddev=sqrt_delta_sigma)

        return randomness

    def __call__(self, inputs):

        noise = tf.ones_like(inputs, dtype=tf.float32)  * self.noise_default
        noise = tf.Variable(noise,
                            dtype=tf.float32,
                            trainable=False)


        drift           = self.ou_a * (self.ou_mu - noise) * self.delta
        brownian_motion = self.brownian_motion(noise)
        new_noise       = (noise + drift + brownian_motion) * self.exploration

        self.noise_assign_ops.append(noise.assign(new_noise))

        outputs         = new_noise + inputs

        return outputs

    def noise_update_tensors(self):
        return [*self.noise_assign_ops, self.exploration_update_op]

