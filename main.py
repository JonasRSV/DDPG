import gym
import time
import policy_gradient
import sys
import replay_buffer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from noise import Noise


ENV = 'Pendulum-v0'

GENERATIONS   = 1000
ACTION_SPACE  = 1
STATE_SPACE   = 3 
LEARNING_RATE = 0.01

FRAME_SZ      = 1000
BATCHSZ       = 20
MEMORY        = 0.98
TAU           = 0.01

NOISE_REVERSION = 0.99


def train(env, actor, rpbuffer):

    actor.update_target_network()

    plt.style.use('dark_background')
    actions = np.arange(ACTION_SPACE)

    noise_process = Noise(1, 0, NOISE_REVERSION)

    generations   = []
    rewards       = []
    for g in range(GENERATIONS):
        s1       = env.reset()
        terminal = False

        reward = 0
        noise = np.zeros(ACTION_SPACE)
        while not terminal:
            env.render()

            s = s1.reshape(1, -1)

            action = actor.predict(s)[0]
            noise  = noise_process.process(noise)

            action = action + noise

            s2, r2, terminal, _ = env.step(action)
            reward += r2

            rpbuffer.add((s1, action, r2, terminal, s2))
            s1 = s2

            if len(rpbuffer.buffer) >= BATCHSZ:

                s1b, a1b, r1b, dd, s2b = rpbuffer.get(BATCHSZ)
                environment_utility = actor.target_critique(s2b, a1b)

                maximal_utilities = []
                for reward, utility, term in zip(r1b, environment_utility, dd):
                    if term:
                        maximal_utilities.append([reward])
                    else:
                        maximal_utilities.append(reward + MEMORY * utility)

                _ = actor.train(s1b, a1b, maximal_utilities)
                actor.update_target_network()

        generations.append(g)
        rewards.append(reward)
        
        plt.plot(generations, rewards)
        plt.pause(0.001)

    env.close()


def play(env, actor, games=20):
    for i in range(games):
        terminal = False
        s0 = env.reset()


        while not terminal:
            env.render()
            s0 = s0.reshape(1, -1)
            action = actor.predict(s0)[0]

            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env = gym.make(ENV)
    print(env.action_space)

    with tf.Session() as sess:
        actor   = policy_gradient.PG(sess, STATE_SPACE, ACTION_SPACE, learning_rate=LEARNING_RATE, tau=TAU)
        rpbuffer = replay_buffer.ReplayBuffer(FRAME_SZ)

        saver = tf.train.Saver()

        if "-n" in sys.argv:
            sess.run(tf.global_variables_initializer())
        else:
            saver.restore(sess, "model/")
            print("Restored...")

        try: 
            if "-p" in sys.argv:
                print("Playing...")
                play(env, actor)
            else:
                train(env, actor, rpbuffer)
        except KeyboardInterrupt:
            pass

        saver.save(sess, "model/")


    
