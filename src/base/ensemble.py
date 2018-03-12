import numpy as np
from threading import Event, Lock, Condition

ensemble_lock_ens_w_a = Lock()
ensemble_condition = Condition()

ensemble_inter_max = 100

ensemble_weights_actions = np.empty(0)
ensemble_stack_actions = []
ensemble_num_algs = -1
ensemble_env_action_space_n = 0
ensemble_taken_action = 0
ensemble_eps = 0.1

def choose_action_majority_voting():
    print("choose_action_majority_voting():")
    while True:
        if ensemble_num_algs == ensemble_weights_actions.__len__():
            break
    # todo nothing
    print("ok...")
    ensemble_condition.acquire()
    if np.random.uniform(0, 1) < ensemble_eps:
        action = np.random.choice(ensemble_env_action_space_n)
        print("eps: ", ensemble_eps)
    else:
        logits_exp = np.sum(ensemble_weights_actions, axis=0) #suming lines
        print("choose_action_majority_voting() -> ensemble_weights_actions:", ensemble_weights_actions)
        print("choose_action_majority_voting() -> logits_exp:", logits_exp)

        probs = logits_exp / np.sum(logits_exp)
        action = probs.argmax(axis=0)
        for i in range(ensemble_num_algs):
            ensemble_stack_actions.append(action)
        print("choose_action_majority_voting() -> probs:", probs)
        print("choose_action_majority_voting() -> action:", action)
    ensemble_condition.notify_all()
    ensemble_condition.release()

def setLengthThreads(num):
    num_algs = num

def setEnvActionSpace(n):
    ensemble_env_action_space_n = n

"""
Events #

An event is a simple synchronization object; the event represents an internal flag, and threads can wait for the flag to be set, or set or clear the flag themselves.

event = threading.Event()

# a client thread can wait for the flag to be set
event.wait()

# a server thread can set or reset it
event.set()
event.clear()
If the flag is set, the wait method doesn’t do anything. If the flag is cleared, wait will block until it becomes set again. Any number of threads may wait for the same event.
"""


'''
def choose_action_qlearn_old(env, pos, q_table, vel):
    eps = 0.02
    if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
    else:
        logits = q_table[pos][vel]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        action = np.random.choice(env.action_space.n, p=probs)
    return action

def choose_action_sarsa_old(env, pos, q_table, vel):
    eps = 0.1
    if np.random.uniform(0, 1) < eps:
        action = np.random.choice(env.action_space.n)
    else:
        logits = q_table[pos][vel]
        logits_exp = np.exp(logits)
        probs = logits_exp / np.sum(logits_exp)
        action = np.random.choice(env.action_space.n, p=probs)
    return action
'''