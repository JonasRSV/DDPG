import gym
import ddpg
import sys
import numpy as np
import tensorflow as tf
import gym_wrapper


ENV = 'Pendulum-v0'


def action_modifier(action):
    return np.clip(action, -2, 2)


if __name__ == "__main__":

    env = gym.make(ENV)

    with tf.Session() as sess:
        training = None
        if "-n" in sys.argv:
            training = True
        else:
            training = False

        actor = ddpg.DDPG(3,
                          1,
                          -2,
                          2,
                          memory=0.99,
                          actor_lr=0.001,
                          critic_lr=0.001,
                          tau=0.001,
                          exp_batch=256,
                          training=training)

        saver = tf.train.Saver()
        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/pendelum")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                gym_wrapper.play(env, actor, a_mod=action_modifier)
            else:
                gym_wrapper.train(env, actor, 50000, a_mod=action_modifier)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/pendelum")

