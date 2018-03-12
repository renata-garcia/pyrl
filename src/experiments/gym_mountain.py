import src.base.ensemble as ensemble
from threading import Thread
import src.rlalgorithms.qlearn as ql
import src.rlalgorithms.sarsa as srs
import gym

gym_mountain_threads = []

def configs_algorithms():
    gym_mountain_threads.append(Thread(target=ql.qlearn))
    gym_mountain_threads.append(Thread(target=srs.sarsa))

def run():
    print("running....")
    ensemble.setEnvActionSpace((gym.make('MountainCar-v0')).action_space.n)
    ensemble.setLengthThreads(gym_mountain_threads.__len__())
    policy = Thread(target=ensemble.choose_action_majority_voting())

    policy.start()
    for i in range(gym_mountain_threads.__len__()):
        gym_mountain_threads[i].start()

    policy.join()
    for i in range(gym_mountain_threads.__len__()):
        gym_mountain_threads[i].join()

    print("running....")

if __name__ == '__main__':
    run()