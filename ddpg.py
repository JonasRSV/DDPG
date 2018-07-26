import noise
import tensorflow as tf
import numpy as np

def weigth_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01, mean=0.0)
    return tf.Variable(initial, dtype=tf.float32)


def normalize_deviation(variable):
    mean, variance = tf.nn.moments(variable, axes=0) 
    return (variable - mean) / tf.sqrt(variance + 1)


def normalize_vector(tensor, ord):
    norm = tf.norm(tensor, ord=ord, axis=0)
    return tensor / (norm + 1)


def make_normalize_devation_op(variable):
    normalize_op = variable.assign(normalize_deviation(variable))
    return normalize_op


ACTOR_CONNECTIONS  = 20
CRITIC_CONNECTIONS = 20


class DDPG(object):

    def __init__(self, sess, state_dim, action_dim, learning_rate=0.01, 
                 tau=0.001, delta=1.0, sigma=0.4, ou_a=0.4, ou_mu=0.0, var_index=0,
                 decay=5e-4, parameter_noise=True):
        self.sess  = sess
        self.s_dim = state_dim
        self.a_dim = action_dim

        self.parameter_noise = parameter_noise
        self.noise_process   = noise.OrnsteinNoiseTensorflow(delta,
                                                             sigma,
                                                             ou_a,
                                                             ou_mu,
                                                             decay=decay)

        ####################################
        # Define Normalizing OP's          #
        # https://arxiv.org/abs/1607.06450 #
        ####################################
        self.normalize_deviation_ops = []

        ########################################################
        # Define Actor Critic Architecture and target networks #
        ########################################################

        actor_state, actor_out = self.create_actor("vanilla_actor")
        actor_variables  = tf.trainable_variables()[var_index:]

        critic_state, critic_action, critic_out = self.create_critic("vanilla_critic")
        critic_variables = tf.trainable_variables()[var_index + len(actor_variables):]

        vanilla_variables = tf.trainable_variables()[var_index:]

        target_actor_state, target_actor_out = self.create_actor("target_actor")
        target_critic_state, target_critic_action, target_critic_out = self.create_critic("target_critic")

        target_variables = tf.trainable_variables()[var_index + len(vanilla_variables):]

        ###################################
        # Define Target Network Update Op #
        ###################################

        update_op = [target_var.assign(tf.multiply(target_var, 1 - tau) +\
                                       tf.multiply(vanilla_var, tau))
                        for target_var, vanilla_var in zip(target_variables, vanilla_variables)]


        equal_op = [target_var.assign(vanilla_var)
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
        actor_gradients = tf.placeholder(tf.float32, [self.a_dim, None])
        batch_size      = tf.to_float(tf.shape(actor_gradients)[1])

        """ MINUS IS SUPER IMPORTANT! Remember! Hill Climb """
        actor_train_gradients = tf.gradients(actor_out, actor_variables, -actor_gradients)
        actor_train_gradients = [normalize_vector(grad, "euclidean") / batch_size for grad in actor_train_gradients]

        actor_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate * 0.1)\
                .apply_gradients(zip(actor_train_gradients, actor_variables))



        ################################################
        #          Define Learning OP for critic       #
        # The critic tries to minimize the error       #
        # between its utility function and the         #
        # utility given by the environment it acts on  #
        ################################################

        environment_utility = tf.placeholder(tf.float32, [1, None])
        batch_size          = tf.to_float(tf.shape(environment_utility)[1])

        loss             = tf.losses.mean_squared_error(environment_utility, critic_out) 
        critic_gradients = tf.gradients(loss, critic_variables)
        critic_gradients = [normalize_vector(grad, "euclidean") / batch_size for grad in critic_gradients]
        critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
                .apply_gradients(zip(critic_gradients, critic_variables))

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

        self.critic_state     = critic_state
        self.critic_out       = critic_out
        self.critic_action    = critic_action

        self.target_actor_state      = target_actor_state
        self.target_actor_out        = target_actor_out

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
        self.update_op  = update_op
        self.equal_op   = equal_op 


    def create_critic(self, name):

        ######################################################
        # name_scope used to avoid variable name             #
        # collisions as consequence of target architechture. #
        ######################################################
        with tf.name_scope(name):

            ####################
            # Define Variables #
            ####################
            action = tf.placeholder(tf.float32, [self.a_dim, None])
            state  = tf.placeholder(tf.float32, [self.s_dim, None])

            h_l1 = weigth_variable([CRITIC_CONNECTIONS, self.a_dim + self.s_dim + 1])
            h_w1 = h_l1[:, :-1]
            h_b1 = tf.expand_dims(h_l1[:, -1], [1])

            h_l2 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS + 1])
            h_w2 = h_l2[:, :-1]
            h_b2 = tf.expand_dims(h_l2[:, -1], [1])

            h_l3 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS + 1])
            h_w3 = h_l3[:, :-1]
            h_b3 = tf.expand_dims(h_l3[:, -1], [1])

            out_l = weigth_variable([1, ACTOR_CONNECTIONS + 1])
            out_w = out_l[:, :-1]
            out_b = tf.expand_dims(out_l[:, -1], [1])



            if name == "vanilla_critic":
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l1))
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l2))
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l3))


            ###############
            # Build Graph #
            ###############
            h1 = tf.nn.tanh(tf.matmul(h_w1, tf.concat([state, action], axis=0)) + h_b1)
            h2 = tf.nn.tanh(tf.matmul(h_w2, h1) + h_b2)
            h3 = tf.nn.tanh(tf.matmul(h_w3, h2) + h_b3)
            out = tf.matmul(out_w, h3) + out_b

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
            state = tf.placeholder(tf.float32, [self.s_dim, None])

            h_l1 = weigth_variable([ACTOR_CONNECTIONS, self.s_dim + 1])
            h_w1 = h_l1[:, :-1]
            h_b1 = tf.expand_dims(h_l1[:, -1], [1])

            h_l2 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS + 1])
            h_w2 = h_l2[:, :-1]
            h_b2 = tf.expand_dims(h_l2[:, -1], [1])

            h_l3 = weigth_variable([ACTOR_CONNECTIONS, ACTOR_CONNECTIONS + 1])
            h_w3 = h_l3[:, :-1]
            h_b3 = tf.expand_dims(h_l3[:, -1], [1])

            out_l = weigth_variable([self.a_dim, ACTOR_CONNECTIONS + 1])
            out_w = out_l[:, :-1]
            out_b = tf.expand_dims(out_l[:, -1], [1])

            if name == "vanilla_actor":
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l1))
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l2))
                self.normalize_deviation_ops.append(make_normalize_devation_op(h_l3))
                self.normalize_deviation_ops.append(make_normalize_devation_op(out_l))


            ###############
            # Build Graph #
            ###############

            if self.parameter_noise:
                h_w1 = self.noise_process(h_w1)

            h1 = tf.nn.tanh(tf.matmul(h_w1, state) + h_b1)

            if self.parameter_noise:
                h_w2 = self.noise_process(h_w2)

            h2 = tf.nn.tanh(tf.matmul(h_w2, h1) + h_b2)

            if self.parameter_noise:
                h_w3 = self.noise_process(h_w3)

            h3 = tf.nn.tanh(tf.matmul(h_w3, h2) + h_b3)

            if self.parameter_noise:
                out_w = self.noise_process(out_w)

            out = tf.nn.tanh(tf.matmul(out_w, h3) + out_b)

        return state, out

    def predict(self, state):
        if self.parameter_noise:
            return self.sess.run((*self.noise_process.noise_update_tensors(), self.actor_out), feed_dict={self.actor_state: state})[-1]
        else:
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
        loss, actor_gradients, _ = self.sess.run((self.environment_loss, self.actor_gradients_op, self.critic_optimizer)
                                                    , feed_dict={ self.critic_state: state
                                                                , self.critic_action: action
                                                                , self.environment_utility: environment_utility})

        ########################################################
        # STEP 2: Train actor on the gradients from the critic #
        ########################################################

        self.sess.run(self.actor_optimizer, feed_dict={ self.actor_state: state
                                                      , self.actor_gradients: actor_gradients[0]})

        #################################
        # STEP 3: Normalize the layers. #
        #################################

        self.sess.run(self.normalize_deviation_ops)

        ################################################################
        # STEP 4: Return loss and Qmax if one like for some nice stats #
        ################################################################

        return loss

    def set_networks_equal(self):
        self.sess.run(self.equal_op)

    def update_target_network(self):
        self.sess.run(self.update_op)

