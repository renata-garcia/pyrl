"""
Q-Learning example using OpenAI gym MountainCar enviornment
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""
import numpy as np
import time
import gym
import src.base.ensemble as ensemble
from threading import Condition

#debug
qlearn_time = 0
qlearn_time_set_weights = 0.03

ql_n_states = 40

ql_initial_lr = 1.0 # Learning rate
ql_min_lr = 0.003
ql_gamma = 1.0
ql_t_max = 100
ql_eps = 0.02

def choose_action_qlearn(e, condition, lock_ens_w_a, pos, q_table, vel):
    logits = q_table[pos][vel]
    logits_exp = np.exp(logits)
    print("qlearn -- choose_action_qlearn(pos, q_table, vel): setWeightsActions - ", logits_exp)
    lock_ens_w_a.acquire()
    qlearn_set_weights_actions = False
    while (not qlearn_set_weights_actions):
        if (not qlearn_set_weights_actions):
            time.sleep(qlearn_time + qlearn_time_set_weights)
        qlearn_set_weights_actions = e.setWeightsActions(logits_exp)
    lock_ens_w_a.release()
    condition.acquire()
    print("qlearn -- condition.acquire()")
    condition.wait()
    print("qlearn -- condition.wait()")
    action = e.get_actions()
    condition.release()
    return action

def ql_run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(ql_t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += ql_gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / ql_n_states
    # print("env_low[", env_low[0], ",",env_low[1], "], env_dx[", env_dx[0], ",", env_dx[1],"]")
    # print("obs[0]: ", obs[0], " obs[1]: ", obs[1])
    pos = int((obs[0] - env_low[0])/env_dx[0])
    vel = int((obs[1] - env_low[1])/env_dx[1])
    return pos, vel


def qlearn(e, condition, lock_ens_w_a):
    time.sleep(qlearn_time)
    print("qlearn -- begin qlearn():")
    env_name = 'MountainCar-v0'
    ql_env = gym.make(env_name)
    ql_env.seed(0)
    np.random.seed(0)
    print ('----- using Q Learning -----')
    q_table = np.zeros((ql_n_states, ql_n_states, 3))
    for i in range(e.get_inter_max()):
        print("qlearn -- i:", i)
        obs = ql_env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(ql_min_lr, ql_initial_lr * (0.85 ** (i // 100)))
        for j in range(ql_t_max):
            pos, vel = obs_to_state(ql_env, obs)
            action = choose_action_qlearn(e, condition, lock_ens_w_a, pos, q_table, vel)
            obs, reward, done, _ = ql_env.step(action)
            total_reward += reward
            # update q table
            a_, b_ = obs_to_state(ql_env, obs)
            q_table[pos][vel][action] = q_table[pos][vel][action] + eta * (reward + ql_gamma * np.max(q_table[a_][b_]) - q_table[pos][vel][action])
            #print("q_table[pos(", pos, ")][vel(", vel, ")][action(", action, ")]):", q_table[pos][vel][action], " - reward ", reward)
            if done:
                break
        if i % 100 == 0:
            print('qlearn -- Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [ql_run_episode(ql_env, solution_policy, False) for _ in range(100)]
    print("qlearn -- Average score of solution = ", np.mean(solution_policy_scores))
    # Animate it
    ql_run_episode(ql_env, solution_policy, True)
    ql_env.close()


