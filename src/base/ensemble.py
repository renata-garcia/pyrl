import numpy as np
import time
from threading import Event, Lock, Condition
from time import sleep

#debug
ensemble_time = 0.05
class Ensemble:

    def __init__(self, cond):

        self.inter_max = 100

        self.weights_actions = 0
        self.i_weights_actions = 0
        self.stack_actions = []
        self.condition = cond
        self.num_algs = -1
        self.env_action_space_n = 0
        self.taken_action = 0
        self.eps = 0.1

    def get_actions(self):
        action = -200
        action = self.stack_actions.pop()
        print("####\nget_actions(self):\n", action, "\n####")
        return action

    def setWeightsActions(self, logits_exp):
        print("@@@\n",self.num_algs," in setWeightsActions(", self.i_weights_actions, "): \nlogits_exp(",self.i_weights_actions, ")\n", logits_exp)
        ensemble_set_weights_actions = False
        if (self.i_weights_actions < self.num_algs):
            self.weights_actions[self.i_weights_actions] = logits_exp
            self.i_weights_actions = self.i_weights_actions + 1
            ensemble_set_weights_actions = True
        print("$$$\nout setWeightsActions(", self.i_weights_actions, "): weights_actions\n", self.weights_actions)
        return ensemble_set_weights_actions

    def printWeightsActions(self):
        print("in setWeightsActions(logits_exp): logits_exp(", self.weights_actions, ")")


    def choose_action_majority_voting(self):
        while True:
            print("choose_action_majority_voting(): ensemble_num_algs(", self.num_algs, ") self.i_weights_actions(", self.i_weights_actions, ")")
            if (self.num_algs > 0) & (self.num_algs == self.i_weights_actions):
                print("ok...")
                self.condition.acquire()
                if np.random.uniform(0, 1) < self.eps:
                    action = np.random.choice(self.env_action_space_n)
                    for i in range(self.num_algs):
                        self.stack_actions.append(action)
                    print("eps: ", self.eps)
                else:
                    logits_exp = np.sum(self.weights_actions, axis=0) #suming lines
                    print("choose_action_majority_voting() -> ensemble_weights_actions:", self.weights_actions)
                    print("choose_action_majority_voting() -> logits_exp:", logits_exp)

                    probs = logits_exp / np.sum(logits_exp)
                    action = probs.argmax(axis=0)
                    for i in range(self.num_algs):
                        self.stack_actions.append(action)
                    print("choose_action_majority_voting() -> probs:", probs)
                    print("choose_action_majority_voting() -> action:", action)

                self.condition.notify_all()

                self.weights_actions = np.zeros(shape=(self.num_algs, self.env_action_space_n))
                self.i_weights_actions = 0

                self.condition.release()
            else:
                print("ensemble_sleep")
                sleep(ensemble_time);

    def get_inter_max(self):
        return self.inter_max


    def set_action_space(self, n, num):
        self.num_algs = num
        self.env_action_space_n = n
        self.weights_actions = np.zeros(shape=(self.num_algs, self.env_action_space_n))

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