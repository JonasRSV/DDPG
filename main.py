import gym
import time
import ddpg
import sys
import replay_buffer
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from noise import Noise


ENV = 'Pendulum-v0'

GENERATIONS   = 2000
ACTION_SPACE  = 1
STATE_SPACE   = 3 
LEARNING_RATE = 0.01

FRAME_SZ      = 1000000
BATCHSZ       = 1024
MEMORY        = 0.99
TAU           = 0.01


DELTA = 1.0
SIGMA = 0.3
OU_A  = 0.3
OU_MU = 0

NOISE_DECAY  = 0.95
INTIAL_NOISE = 0.2


def train(env, actor, rpbuffer):

    actor.set_networks_equal()

    summary_write = tf.summary.FileWriter("summaries/", actor.sess.graph)
    summary_index = 0

    noise_process = Noise(DELTA, SIGMA, OU_A, OU_MU)

    steps = 0
    for g in range(1, GENERATIONS):
        s1       = env.reset()
        terminal = False
        
        reward     = 0
        avg_action = 0
        loss       = 0
        noise = np.zeros(ACTION_SPACE)
        while not terminal:
            env.render()

            steps += 1

            s = s1.reshape(1, -1)

            action = actor.predict(s)[0]

            avg_action += action

            noise  = noise_process.ornstein_uhlenbeck_level(noise) * INTIAL_NOISE * NOISE_DECAY**g * 3
            action = np.clip((action + noise) * 2, -2, 2)

            s2, r2, terminal, _ = env.step(action)

            reward += r2
            rpbuffer.add((s1, action, r2, terminal, s2))
            s1 = s2


            if len(rpbuffer.buffer) > BATCHSZ:
                s1b, a1b, r1b, dd, s2b = rpbuffer.get(BATCHSZ)
                environment_utility = actor.target_critique(s2b, a1b)

                maximal_utilities = []
                for reward, utility, term in zip(r1b, environment_utility, dd):
                    if term:
                        maximal_utilities.append([reward])
                    else:
                        maximal_utilities.append(reward + MEMORY * utility)

                loss += actor.train(s1b, a1b, maximal_utilities)
                actor.update_target_network()


        summary = tf.Summary()
        summary.value.add(tag="Reward", simple_value=float(reward))
        summary.value.add(tag="Steps", simple_value=float(steps))
        summary.value.add(tag="Loss", simple_value=float(loss / 200))
        summary.value.add(tag="Action", simple_value=float(avg_action / 200))

        summary_write.add_summary(summary, summary_index)
        summary_write.flush()

        summary_index += 1

    env.close()


def play(env, actor, games=20):
    for i in range(games):
        terminal = False
        s0 = env.reset()


        while not terminal:
            env.render()
            s0 = s0.reshape(1, -1)
            action = actor.predict(s0)[0]
            print(action)

            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env = gym.make(ENV)
    print(env.action_space)

    with tf.Session() as sess:
        actor   = ddpg.DDPG(sess, STATE_SPACE, ACTION_SPACE, learning_rate=LEARNING_RATE, tau=TAU, batch_size=BATCHSZ)
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


    
