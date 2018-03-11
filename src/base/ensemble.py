import numpy as np


def choose_action_qlearn(env, pos, q_table, vel):
    eps = 0.02
    if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
    else:
        logits = q_table[pos][vel]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        action = np.random.choice(env.action_space.n, p=probs)
    return action


def choose_action_sarsa(env, pos, q_table, vel):
    eps = 0.1
    if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
    else:
        logits = q_table[pos][vel]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        action = np.random.choice(env.action_space.n, p=probs)
    return action
