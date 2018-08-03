import tensorflow as tf
import numpy as np
import random
from collections import deque


def OrnsteinNoise(theta, sigma, state):
    while True:
        yield state
        state += -theta * state + sigma * (np.random.rand() - 0.5)


class ExperienceReplay(object):

    def __init__(self, capacity):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def add(self, frame):
        self.buffer.append(frame)

    def get(self, batchsz):

        if len(self.buffer) < batchsz:
            batchsz = len(self.buffer)

        choices = random.sample(self.buffer, batchsz)

        sb_1 = []
        ab_1 = []
        rb_1 = []
        db_1 = []
        sb_2 = []

        while batchsz:
            sb1, ab1, rb1, db1, sb2 = choices.pop()

            sb_1.append(sb1)
            ab_1.append(ab1)
            rb_1.append(rb1)
            db_1.append(db1)
            sb_2.append(sb2)

            batchsz -= 1

        """ numpyfy """
        sb_1 = np.array(sb_1)
        ab_1 = np.array(ab_1)
        rb_1 = np.array(rb_1)
        db_1 = np.array(db_1)
        sb_2 = np.array(sb_2)

        return sb_1, ab_1, rb_1, db_1, sb_2


class DDPG(object):

    def __init__(self, state_dim, action_dim, memory=0.99, actor_lr=0.001,
                 critic_lr=0.001, tau=0.1, critic_hidden_layers=3,
                 actor_hidden_layers=3, critic_hidden_neurons=32,
                 actor_hidden_neurons=32, dropout=0.0, regularization=0.01,
                 scope="ddpg", add_layer_norm=False, training=True, 
                 action_noise=None, max_exp_replay=100000, exp_batch=1024):

        self.sess  = tf.get_default_session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.memory = memory
        self.action_noise = action_noise

        if training and action_noise is None:
            self.action_noise = OrnsteinNoise(0.15, 0.2, 0)

        self.exp_replay = ExperienceReplay(max_exp_replay)
        self.exp_batch  = exp_batch

        self.add_layer_norm  = add_layer_norm
        self.training        = training

        ########################################################
        # Define Actor Critic Architecture and target networks #
        ########################################################

        with tf.variable_scope(scope):

            with tf.variable_scope("pi"):
                with tf.variable_scope("actor"):
                    self.actor_state, self.actor_out =\
                            self.create_actor(actor_hidden_layers, 
                                              actor_hidden_neurons,
                                              dropout,
                                              regularization)
                with tf.variable_scope("critic"):
                    self.critic_state, self.critic_action, self.critic_out =\
                            self.create_critic(critic_hidden_layers,
                                               critic_hidden_neurons,
                                               dropout,
                                               regularization)

            with tf.variable_scope("target_pi"):
                with tf.variable_scope("actor"):
                    self.target_actor_state, self.target_actor_out =\
                            self.create_actor(actor_hidden_layers,
                                              actor_hidden_neurons,
                                              dropout,
                                              regularization)
                with tf.variable_scope("critic"):
                    self.target_critic_state, self.target_critic_action, self.target_critic_out =\
                            self.create_critic(critic_hidden_layers,
                                               critic_hidden_neurons,
                                               dropout,
                                               regularization)

            ###################################
            # Define Target Network Update Op #
            ###################################

            pi_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/pi'.format(scope))

            target_pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                           '{}/target_pi'.format(scope))

            self.update_op = [tpv.assign(tf.multiply(tpv, 1 - tau) +\
                                                   tf.multiply(pv, tau))
                                for tpv, pv in zip(target_pi_vars, pi_vars)]

            self.equal_op = [tpv.assign(pv)
                                for tpv, pv in zip(target_pi_vars, pi_vars)]

            ########################################################
            #             Define Learning OP for actor             #
            # The Idea behind PG methods is to let the actor try   #
            # to maximize the utility (critic output) while the    #
            # critic tries to minimize the error of its utility    #
            # compared to the environment.                         #
            #                                                      #
            # The actor trains my hill-climbing the utility        #
            # function with respect to its action                  #
            #                                                      #
            # Actor gradients will be provided by the critic       #
            ########################################################
            actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                            '{}/pi/actor'.format(scope))

            self.actor_gradients = tf.placeholder(tf.float32, [None, self.a_dim])

            actor_train_gradients = tf.gradients(self.actor_out, actor_vars, self.actor_gradients)
            # actor_train_gradients = [tf.clip_by_norm(grad, 2) for grad in actor_train_gradients]

            self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr)\
                        .apply_gradients(zip(actor_train_gradients, actor_vars))



            ################################################
            #          Define Learning OP for critic       #
            # The critic tries to minimize the error       #
            # between its utility function and the         #
            # utility given by the environment it acts on  #
            ################################################
            critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            '{}/pi/critic'.format(scope))

            self.environment_utility = tf.placeholder(tf.float32, [None, 1])
            self.loss                = tf.losses.mean_squared_error(self.environment_utility, self.critic_out) 

            critic_gradients = tf.gradients(self.loss, critic_vars)
            # critic_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in critic_gradients]

            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)\
                        .apply_gradients(zip(critic_gradients, critic_vars))

            ##########################################
            # Define OP for getting Actor Gradients. #
            # The Gradients used to train the actor. #
            ##########################################

            self.actor_gradients_op =\
                    tf.gradients(-tf.reduce_mean(self.critic_out),
                                 self.critic_action)

    def create_critic(self, layers, neurons, dropout, regularization):

        action = tf.placeholder(tf.float32, [None, self.a_dim])
        state  = tf.placeholder(tf.float32, [None, self.s_dim])

        regularizer = tf.contrib.layers.l2_regularizer(regularization)
        initializer = tf.contrib.layers.variance_scaling_initializer()
        # initializer = tf.truncated_normal_initializer(mean=0, stddev=1.)
        
        x = tf.layers.dense(tf.concat([action, state], axis=1), 
                            neurons,
                            activation=tf.nn.elu,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        x = tf.layers.dropout(x, rate=dropout, training=self.training)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.elu,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizer)

            if self.add_layer_norm:
                x = tf.contrib.layers.layer_norm(x, trainable=False)

            x = tf.layers.dropout(x, rate=dropout, training=self.training)

        out = tf.layers.dense(x, 
                             1, 
                             activation=None, 
                             kernel_regularizer=regularizer)

        return state, action, out

    def create_actor(self, layers, neurons, dropout, regularization):

        state       = tf.placeholder(tf.float32, [None, self.s_dim])
        regularizer = tf.contrib.layers.l2_regularizer(regularization)
        # initializer = tf.contrib.layers.xavier_initializer(uniform=False)
        initializer = tf.contrib.layers.variance_scaling_initializer()

        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.elu,
                            kernel_initializer=initializer,
                            kernel_regularizer=regularizer)

        if self.add_layer_norm:
            x = tf.contrib.layers.layer_norm(x, trainable=False)

        x = tf.layers.dropout(x, rate=dropout, training=self.training)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.elu,
                                kernel_initializer=initializer,
                                kernel_regularizer=regularizer)

            if self.add_layer_norm:
                x = tf.contrib.layers.layer_norm(x, trainable=False)

            x = tf.layers.dropout(x, rate=dropout, training=self.training)

        out = tf.layers.dense(x, self.a_dim,
                              activation=tf.nn.tanh, 
                              # kernel_initializer=initializer,
                              kernel_regularizer=regularizer)
        return state, out

    def predict(self, state):
        actions = self.sess.run(self.actor_out, feed_dict={self.actor_state: state})
        if self.action_noise:
            actions += next(self.action_noise)

        return actions

    def critique(self, state, action):
        return self.sess.run(self.critic_out,
                             feed_dict={self.critic_state: state,
                                        self.critic_action: action})

    def target_predict(self, state):
        return self.sess.run(self.target_actor_out,
                             feed_dict={self.target_actor_state: state})


    def target_critique(self, state, action):
        return self.sess.run(self.target_critic_out,
                             feed_dict={self.target_critic_state: state,
                                        self.target_critic_action: action})

    def train(self):
        #######################################
        # Get predicted utility of next state #
        #######################################
        s1b, a1b, r1b, tb, s2b = self.exp_replay.get(self.exp_batch)

        a2b               = self.target_predict(s2b)
        next_utility      = self.target_critique(s2b, a2b).reshape(-1)
        predicted_utility = r1b + self.memory * (1 - tb) * next_utility
        predicted_utility = predicted_utility.reshape(-1, 1)

        ################################################
        # STEP 1: Get actor gradients and train critic #
        ################################################
        loss, actor_gradients, _ = self.sess.run((self.loss, self.actor_gradients_op, self.critic_optimizer)
                                                    , feed_dict={ self.critic_state: s1b
                                                                , self.critic_action: a1b
                                                                , self.environment_utility: predicted_utility})

        ########################################################
        # STEP 2: Train actor on the gradients from the critic #
        ########################################################

        self.sess.run(self.actor_optimizer, feed_dict={ self.actor_state: s1b
                                                      , self.actor_gradients: actor_gradients[0]})

        self.update_target_network()
        return loss

    def set_networks_equal(self):
        self.sess.run(self.equal_op)

    def update_target_network(self):
        self.sess.run(self.update_op)

    def add_experience(self, experience):
        self.exp_replay.add(experience)

