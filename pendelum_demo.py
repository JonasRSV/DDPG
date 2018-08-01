import gym
import time
import ddpg
import sys
import replay_buffer
import numpy as np
import tensorflow as tf


ENV = 'Pendulum-v0'

EPOCHS        = 2000
ACTION_SPACE  = 1
STATE_SPACE   = 3 
ACTOR_LR      = 0.01
CRITIC_LR     = 0.01

FRAME_SZ      = 100000
BATCHSZ       = 1024
MEMORY        = 0.99
TAU           = 0.01


def train(env, actor, rpbuffer):

    actor.set_networks_equal()

    summary_write = tf.summary.FileWriter("summaries/", actor.sess.graph)
    summary_index = 0

    steps = 0
    for g in range(1, EPOCHS):
        s1       = env.reset()
        terminal = False
        
        reward_     = 0
        avg_action = 0
        loss       = 0

        noise_process = np.zeros(ACTION_SPACE)
        noise_scale = (0.1 * 0.999*g) * (4)
        while not terminal:
            env.render()

            steps += 1

            s = s1.reshape(1, -1)

            action = actor.predict(s)[0]
            avg_action += action

            noise_process = 0.15 * (0 - noise_process) + 0.2 * np.random.randn(ACTION_SPACE)
            action += noise_process


            action = np.clip(action * 2, -2, 2)
            print(action)
            s2, r2, terminal, _ = env.step(action)

            reward_ += r2
            rpbuffer.add((s1, action, r2, terminal, s2))
            s1 = s2


            if len(rpbuffer.buffer) > BATCHSZ * 5:
                print("TRAINING")
                s1b, a1b, r1b, dd, s2b = rpbuffer.get(BATCHSZ)
                loss = actor.train(s1b, a1b, r1b, dd, s2b)
                actor.update_target_network()


        summary = tf.Summary()
        summary.value.add(tag="Reward", simple_value=float(reward_))
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
            action = np.clip(action * 2, -2, 2)
            print(action)

            s0, _, terminal, _ = env.step(action)

    env.close()


if __name__ == "__main__":

    env = gym.make(ENV)
    print(env.action_space)

    with tf.Session() as sess:
        actor = ddpg.DDPG(STATE_SPACE, 
                          ACTION_SPACE, 
                          memory=MEMORY,
                          actor_lr=ACTOR_LR, 
                          critic_lr=CRITIC_LR,
                          tau=TAU)

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


    
