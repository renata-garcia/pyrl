
import numpy as np
import src.base.ensemble as ens
from threading import Condition

ensemble_weights_actions = np.empty(0)
threads = []

def teste01():
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


cond = Condition
a = ens.Ensemble(cond)
b = ens.Ensemble(cond)
c = ens.Ensemble(cond)
def teste02():
    print("Teste Objects")
    a.setWeightsActions([1,2,3]);
    b.setWeightsActions([4,5,6]);
    c.setWeightsActions([7,8,9]);

    a.printWeightsActions();
    b.printWeightsActions();
    c.printWeightsActions();


teste03 = np.zeros(shape=(2,3))
def teste03_soma_do_array():
    teste03[0] = [0, 1, 2]
    teste03[1] = [3, 4, 5]
    print(teste03)
    npsum = np.sum(teste03, axis=0)
    print("\n", teste03)
    print("\n", npsum)

if __name__ == '__main__':
    teste03_soma_do_array()
