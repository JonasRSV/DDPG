
import tensorflow as tf
import numpy as np

def weigth_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, dtype=tf.float32)

def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial, dtype=tf.float32)

ACTOR_CONNECTIONS  = 64
CRITIC_CONNECTIONS = 200

class PG(object):

    def __init__(self, sess, state_dim, action_dim, action_modifier=lambda x: x, learning_rate=0.01, tau=0.001, var_index=0):
        self.sess  = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.a_mod = action_modifier

        ########################################################
        # Define Actor Critic Architecture and target networks #
        ########################################################

        actor_state, actor_out, actor_scaled_out = self.create_actor("vanilla_actor")
        actor_variables  = tf.trainable_variables()[var_index:]

        critic_state, critic_action, critic_out = self.create_critic("vanilla_critic")
        critic_variables = tf.trainable_variables()[var_index + len(actor_variables):]

        vanilla_variables = tf.trainable_variables()[var_index:]

        target_actor_state, target_actor_out, target_actor_scaled_out = self.create_actor("target_actor")
        target_critic_state, target_critic_action, target_critic_out = self.create_critic("target_critic")

        target_variables = tf.trainable_variables()[var_index + len(vanilla_variables):]

        ###################################
        # Define Target Network Update Op #
        ###################################

        update_op = [target_var.assign(tf.multiply(target_var, 1 - tau) +\
                                       tf.multiply(vanilla_var, tau))
                        for target_var, vanilla_var in zip(target_variables, vanilla_variables)]



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
        actor_gradients = tf.placeholder(tf.float32, [None, self.a_dim])

        """ MINUS IS SUPER IMPORTANT! Remember! Hill Climb """
        actor_train_gradients = tf.gradients(actor_out, actor_variables, -actor_gradients)

        actor_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                .apply_gradients(zip(actor_train_gradients, actor_variables))



        ################################################
        #          Define Learning OP for critic       #
        # The critic tries to minimize the error       #
        # between its utility function and the         #
        # utility given by the environment it acts on  #
        ################################################

        environment_utility = tf.placeholder(tf.float32, [None, 1])

        loss = tf.losses.mean_squared_error(environment_utility, critic_out)
        critic_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                .minimize(loss)

        ##########################################
        # Define OP for getting Actor Gradients. #
        # The Gradients used to train the actor. #
        ##########################################

        actor_gradients_op = tf.gradients(critic_out, critic_action)

        ############################################################
        # It's important to notice that critic_action != actor_out #
        # critic_action = actor_out + EXPLORATION_NOISE            #
        # PG methods is trained through off-policy exploration,    #
        # therefore the critic_action if it was the good one is    #
        # what the training will try to make the actor_action into #
        # given the same state.                                    #
        ############################################################


        ######################################
        # Tensors needed for the PG network  #
        # Vanilla PG does not need all these #
        # tensors, theres alot of them       #
        # because of the target network      #
        #                                    #
        # The purpose of the target network  #
        # is to stabilize training through   #
        # soft-updates.                      #
        ######################################

        self.actor_state      = actor_state
        self.actor_out        = actor_out
        self.actor_scaled_out = actor_scaled_out

        self.critic_state     = critic_state
        self.critic_out       = critic_out
        self.critic_action    = critic_action

        self.target_actor_state      = target_actor_state
        self.target_actor_out        = target_actor_out
        self.target_actor_scaled_out = target_actor_scaled_out

        self.target_critic_state     = target_critic_state
        self.target_critic_out       = target_critic_out
        self.target_critic_action    = target_critic_action


        #############################
        # Training specific tensors #
        #############################
        self.actor_gradients = actor_gradients
        self.actor_optimizer = actor_optimizer

        self.environment_utility = environment_utility
        self.environment_loss    = loss
        self.critic_optimizer    = critic_optimizer

        self.actor_gradients_op  = actor_gradients_op

        ######################################
        # For Soft updates to target network #
        ######################################
        self.update_op           = update_op


    def create_critic(self, name):

        ######################################################
        # name_scope used to avoid variable name             #
        # collisions as consequence of target architechture. #
        ######################################################
        with tf.name_scope(name):

            ####################
            # Define Variables #
            ####################
            action = tf.placeholder(tf.float32, [None, self.a_dim])
            state  = tf.placeholder(tf.float32, [None, self.s_dim])

            h_w1 = weigth_variable([self.s_dim, CRITIC_CONNECTIONS])
            h_b1 = bias_variable([CRITIC_CONNECTIONS])

            h_w2 = weigth_variable([CRITIC_CONNECTIONS, CRITIC_CONNECTIONS])
            h_b2 = bias_variable([CRITIC_CONNECTIONS])

            h_w3 = weigth_variable([CRITIC_CONNECTIONS, CRITIC_CONNECTIONS])
            h_b3 = bias_variable([CRITIC_CONNECTIONS])

            con_aw = weigth_variable([CRITIC_CONNECTIONS, self.a_dim])
            con_ab = bias_variable([self.a_dim])

            out_w  = weigth_variable([self.a_dim, 1])
            out_b  = bias_variable([1])


            ###############
            # Build Graph #
            ###############
            h1 = tf.matmul(state, h_w1)
            h1 = tf.nn.tanh(tf.add(h1, h_b1))

            h2 = tf.matmul(h1, h_w2)
            h2 = tf.nn.tanh(tf.add(h2, h_b2))

            h3 = tf.matmul(h2, h_w3)
            h3 = tf.nn.tanh(tf.add(h3, h_b3))

            con_a = tf.matmul(h3, con_aw)
            con_a = tf.nn.tanh(tf.add(con_a, con_ab))
            con_a = tf.add(con_a, action)

            out = tf.matmul(con_a, out_w)
            out = tf.add(out, out_b)

        return state, action, out

    def create_actor(self, name):

        ######################################################
        # name_scope used to avoid variable name             #
        # collisions as consequence of target architechture. #
        ######################################################
        with tf.name_scope(name):

            ####################
            # Define Variables #
            ####################
            state = tf.placeholder(tf.float32, [None, self.s_dim])

            h_w1 = weigth_variable([self.s_dim, ACTOR_CONNECTIONS])
            h_b1 = bias_variable([ACTOR_CONNECTIONS])

            h_w2 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS])
            h_b2 = bias_variable([ACTOR_CONNECTIONS])

            h_w3 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS])
            h_b3 = bias_variable([ACTOR_CONNECTIONS])

            out_w = weigth_variable([ACTOR_CONNECTIONS, self.a_dim])
            out_b = bias_variable([self.a_dim])

            ###############
            # Build Graph #
            ###############
            h1 = tf.matmul(state, h_w1)
            h1 = tf.nn.tanh(tf.add(h1, h_b1))

            h2 = tf.matmul(h1, h_w2)
            h2 = tf.nn.tanh(tf.add(h2, h_b2))

            h3 = tf.matmul(h2, h_w3)
            h3 = tf.nn.tanh(tf.add(h3, h_b3))

            out = tf.matmul(h3, out_w)
            out = tf.nn.tanh(tf.add(out, out_b))

            scaled_out = self.a_mod(out)

        return state, out, scaled_out

    def predict(self, state):
        return self.sess.run(self.actor_scaled_out, feed_dict={self.actor_state: state})

    def critique(self, state, action):
        return self.sess.run(self.critic_out, feed_dict={self.critic_state: state, self.critic_action: action})

    def target_predict(self, state):
        return self.sess.run(self.target_actor_scaled_out, feed_dict={self.target_actor_state: state})

    def target_critique(self, state, action):
        return self.sess.run(self.target_critic_out, feed_dict={self.target_critic_state: state, self.target_critic_action: action})

    def train(self, state, action, environment_utility):

        ################################################
        # STEP 1: Get actor gradients and train critic #
        ################################################

        loss, actor_gradients, _ = self.sess.run((self.environment_loss, self.actor_gradients_op, self.critic_optimizer)
                                                    , feed_dict={ self.critic_state: state
                                                                , self.critic_action: action
                                                                , self.environment_utility: environment_utility})
        ########################################################
        # STEP 2: Train actor on the gradients from the critic #
        ########################################################

        self.sess.run(self.actor_optimizer, feed_dict={ self.actor_state: state
                                                      , self.actor_gradients: actor_gradients[0]})

        ################################################################
        # STEP 3: Return loss and Qmax if one like for some nice stats #
        ################################################################

        return loss

    def update_target_network(self):
        self.sess.run(self.update_op)

