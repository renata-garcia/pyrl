import src.base.ensemble as ensemble
from threading import Thread, Condition, Lock
import src.rlalgorithms.qlearn as ql
import src.rlalgorithms.sarsa as srs
import gym

gym_mountain_threads = []
condition = Condition()
lock_ens_w_a = Lock()
e = ensemble.Ensemble(condition)

def configs_algorithms():
    gym_mountain_threads.append(Thread(target=ql.qlearn, args=(e, condition, lock_ens_w_a)))
    gym_mountain_threads.append(Thread(target=srs.sarsa, args=(e,condition, lock_ens_w_a)))
    gym_mountain_threads.append(Thread(target=e.choose_action_majority_voting))
    e.set_action_space((gym.make('MountainCar-v0')).action_space.n, (gym_mountain_threads.__len__() - 1))

def run():
    print("running....")

    configs_algorithms()

    for i in range(gym_mountain_threads.__len__()):
        gym_mountain_threads[i].start()

    for i in range(gym_mountain_threads.__len__()):
        gym_mountain_threads[i].join()

if __name__ == '__main__':
    run()
