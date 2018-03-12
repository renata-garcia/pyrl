"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np

import gym
import src.base.ensemble as ensemble
#import src.experiments.gym_mountain as ensemble

srs_n_states = 40

srs_alpha=0.1
srs_gamma=0.9

srs_initial_lr = 1.0 # Learning rate
srs_min_lr = 0.003
srs_t_max = 100
srs_eps = 0.02


def choose_action_sarsa(pos, q_table, vel):
    logits = q_table[pos][vel]
    logits_exp = np.exp(logits)

    ensemble.ensemble_lock_ens_w_a.acquire()
    ensemble_weights_actions = np.append(ensemble.ensemble_weights_actions, logits_exp)
    ensemble.ensemble_lock_ens_w_a.release()

    ensemble.ensemble_condition.acquire()
    while True:
        try:
            action = ensemble.ensemble_stack_actions.pop()
            break
        except ValueError:
            ensemble.ensemble_condition.wait()

    ensemble.ensemble_condition.release()
    return action


def srs_run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(srs_t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += srs_gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / srs_n_states
    # print("env_low[", env_low[0], ",",env_low[1], "], env_dx[", env_dx[0], ",", env_dx[1],"]")
    # print("obs[0]: ", obs[0], " obs[1]: ", obs[1])
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel


def sarsa():
    env_name = 'MountainCar-v0'
    srs_env = gym.make(env_name)
    srs_env.seed(0)
    np.random.seed(0)
    print ('----- using SARSA -----')
    q_table = np.zeros((srs_n_states, srs_n_states, 3))
    for i in range(ensemble.ensemble_inter_max):
        obs = srs_env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(srs_min_lr, srs_initial_lr * (0.85 ** (i // 100)))
        for j in range(srs_t_max):
            pos, vel = obs_to_state(srs_env, obs)
            action = choose_action_sarsa(srs_env, pos, q_table, vel)
            obs, reward, done, _ = srs_env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(srs_env, obs)
            q_table[pos][vel][action] = q_table[pos][vel][action] + eta * (reward + srs_gamma * np.max(q_table[a_][b_] - q_table[pos][vel][action]))
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [srs_run_episode(srs_env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))

    #print(q_table)
    #print(np.shape(q_table))
    #print(solution_policy)
    ##print(np.shape(solution_policy))
    #print(solution_policy_scores)
    #print(np.shape(solution_policy_scores))
    #TODO: Animate it
    #srs_run_episode(srs_env, solution_policy, True)
    srs_env.close()

