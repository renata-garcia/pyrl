import random
import sys

"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np

import gym
import src.base.ensemble as ensemble
#import src.experiments.gym_mountain as ensemble

n_states = 40
iter_max = 10000

alpha=0.1
gamma=0.9

min_lr = 0.003
t_max = 10000
eps = 0.02

def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    # print("env_low[", env_low[0], ",",env_low[1], "], env_dx[", env_dx[0], ",", env_dx[1],"]")
    # print("obs[0]: ", obs[0], " obs[1]: ", obs[1])
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel


def sarsa():
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print ('----- using SARSA -----')
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        #eta = max(min_lr, initial_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            pos, vel = obs_to_state(env, obs)
            action = ensemble.choose_action_qlearn(env, pos, q_table, vel)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[pos][vel][action] = q_table[pos][vel][action] + alpha * (reward + gamma * np.max(q_table[a_][b_] - q_table[pos][vel][action]))
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    run_episode(env, solution_policy, True)
    env.close()

