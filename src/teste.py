
import numpy as np

ensemble_weights_actions = np.empty(0)
threads = []

if __name__ == '__main__':
    a = 3 #env.action_space.n
    probs = [0.5, 0.25, 0.25]
    for i in range(0,5):
        print("i: ", i, " - ", np.random.choice(a,p=probs))
        threads.append(i)

    ensemble_weights_actions = np.append(ensemble_weights_actions, [0, 1, 2])
    ensemble_weights_actions = np.append(ensemble_weights_actions, [3, 4, 5])
    ensemble_weights_actions = np.append(ensemble_weights_actions, [6, 7, 8])
    print(ensemble_weights_actions)

    for i in range(threads.__len__()):
        print(threads[i])

