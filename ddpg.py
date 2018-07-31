import tensorflow as tf
import numpy as np

class DDPG(object):

    def __init__(self, state_dim, action_dim, actor_lr=0.01, critic_lr=0.01, 
                 tau=0.001, critic_hidden_layers=3, actor_hidden_layers=3,
                 critic_hidden_neurons=32, actor_hidden_neurons=32, 
                 scope="ddpg", add_param_noise=True, training=True):

        self.sess  = tf.get_default_session()
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.add_param_noise = add_param_noise
        self.training        = training

        ########################################################
        # Define Actor Critic Architecture and target networks #
        ########################################################

        with tf.variable_scope(scope):

            with tf.variable_scope("pi"):
                with tf.variable_scope("actor"):
                    self.actor_state, self.actor_out =\
                            self.create_actor(actor_hidden_layers, 
                                              actor_hidden_neurons)
                with tf.variable_scope("critic"):
                    self.critic_state, self.critic_action, self.critic_out =\
                            self.create_critic(critic_hidden_layers,
                                               critic_hidden_neurons)


            with tf.variable_scope("target_pi"):
                with tf.variable_scope("actor"):
                    self.target_actor_state, self.target_actor_out =\
                            self.create_actor(actor_hidden_layers,
                                              actor_hidden_neurons)
                with tf.variable_scope("critic"):
                    self.target_critic_state, self.target_critic_action, self.target_critic_out =\
                            self.create_critic(critic_hidden_layers,
                                               critic_hidden_neurons)


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

            target_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            '{}/target_pi/actor'.format(scope))


            self.actor_gradients = tf.placeholder(tf.float32, [None, self.a_dim])

            """ MINUS IS SUPER IMPORTANT! Remember! Hill Climb """
            actor_train_gradients = tf.gradients(self.target_actor_out, target_actor_vars, self.actor_gradients)
            actor_train_gradients = [tf.clip_by_value(grad, -0.1, 0.1) for grad in actor_train_gradients]

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

            target_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                            '{}/target_pi/critic'.format(scope))


            self.environment_utility = tf.placeholder(tf.float32, [None, 1])
            self.loss                = tf.losses.mean_squared_error(self.environment_utility, self.target_critic_out) 

            critic_gradients = tf.gradients(self.loss, target_critic_vars)
            critic_gradients = [tf.clip_by_value(grad, -0.1, 0.1) for grad in critic_gradients]

            self.critic_optimizer = tf.train.AdamOptimizer(learning_rate=critic_lr)\
                        .apply_gradients(zip(critic_gradients, critic_vars))

            ##########################################
            # Define OP for getting Actor Gradients. #
            # The Gradients used to train the actor. #
            ##########################################

            l1_regularizer          = tf.contrib.layers.l1_regularizer(scale=0.001, scope=None)
            regularization_penalty  = tf.contrib.layers.apply_regularization(l1_regularizer, actor_vars)

            self.actor_gradients_op = tf.gradients(-tf.reduce_mean(self.target_critic_out) + regularization_penalty, self.target_critic_action)

    def create_critic(self, layers, neurons):

        action = tf.placeholder(tf.float32, [None, self.a_dim])
        state  = tf.placeholder(tf.float32, [None, self.s_dim])

        initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        
        x = tf.layers.dense(tf.concat([action, state], axis=1), 
                            neurons,
                            activation=tf.nn.tanh,
                            kernel_initializer=initializer)

        x = tf.contrib.layers.layer_norm(x)
        x = tf.layers.dropout(x, rate=0.01, training=self.training)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.tanh,
                                kernel_initializer=initializer)

            x = tf.contrib.layers.layer_norm(x)
            x = tf.layers.dropout(x, rate=0.01, training=self.training)

        out = tf.layers.dense(x, 1, activation=None, kernel_initializer=initializer)

        return state, action, out

    def create_actor(self, layers, neurons):

        state       = tf.placeholder(tf.float32, [None, self.s_dim])
        initializer = tf.random_normal_initializer(mean=0, stddev=0.1)
        param_noise = tf.keras.layers.GaussianNoise(stddev=2.0)

        x = tf.layers.dense(state, 
                            neurons,
                            activation=tf.nn.tanh,
                            kernel_initializer=initializer)

        x = tf.contrib.layers.layer_norm(x)
        x = tf.layers.dropout(x, rate=0.01, training=self.training)

        for _ in range(layers):
            x = tf.layers.dense(x,
                                neurons,
                                activation=tf.nn.tanh,
                                kernel_initializer=initializer)

            x = tf.contrib.layers.layer_norm(x)
            x = tf.layers.dropout(x, rate=0.01, training=self.training)

            if self.add_param_noise:
                x = param_noise(x)


        out = tf.layers.dense(x, self.a_dim, activation=tf.nn.tanh, kernel_initializer=initializer)

        if self.add_param_noise:
            out = param_noise(out)

        return state, out

    def predict(self, state):
        return self.sess.run(self.actor_out, feed_dict={self.actor_state: state})

    def critique(self, state, action):
        return self.sess.run(self.critic_out, feed_dict={self.critic_state: state, self.critic_action: action})

    def target_predict(self, state):
        return self.sess.run(self.target_actor_out, feed_dict={self.target_actor_state: state})

    def target_critique(self, state, action):
        return self.sess.run(self.target_critic_out, feed_dict={self.target_critic_state: state, self.target_critic_action: action})

    def train(self, state, action, environment_utility):

        ################################################
        # STEP 1: Get actor gradients and train critic #
        ################################################
        loss, actor_gradients, _ = self.sess.run((self.loss, self.actor_gradients_op, self.critic_optimizer)
                                                    , feed_dict={ self.critic_state: state
                                                                , self.critic_action: action
                                                                , self.target_critic_state: state
                                                                , self.target_critic_action: action
                                                                , self.environment_utility: environment_utility})

        ########################################################
        # STEP 2: Train actor on the gradients from the critic #
        ########################################################

        self.sess.run(self.actor_optimizer, feed_dict={ self.actor_state: state
                                                      , self.target_actor_state: state
                                                      , self.actor_gradients: actor_gradients[0]})


        return loss

    def set_networks_equal(self):
        self.sess.run(self.equal_op)

    def update_target_network(self):
        self.sess.run(self.update_op)

