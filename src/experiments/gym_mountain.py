import src.rlalgorithms.qlearn as q
import pandas
import numpy

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    # env.monitor.start('/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    # objetivo_passos_medio = 195
    num_max_de_passos = 1000
    num_max_de_episodios = 200
    num_posicoes_para_discretizacao = 10

    ### num_caracteristicas = env.observation_space.shape[0]
    passos_episodio_completados = numpy.ndarray(0)
    i_passos_episodios_completados = 0

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    #Quantile-based discretization function.
    feature1_bins = pandas.cut([-1.2, 0.6], bins=num_posicoes_para_discretizacao, retbins=True)[1][1:-1]
    feature2_bins = pandas.cut([-0.07, 0.07], bins=num_posicoes_para_discretizacao, retbins=True)[1][1:-1]

    # The Q-learn algorithm
    qlearn = q.QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)

    for i_episode in range(num_max_de_episodios):
        observation = env.reset()

        feature1, feature2 = observation
        state = q.build_state([q.to_bin(feature1, feature1_bins),
                         q.to_bin(feature2, feature2_bins)])

        qlearn.epsilon = qlearn.epsilon * 0.999 # added epsilon decay
        cumulated_reward = 0

        for t in range(num_max_de_passos):
            #env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)
            print(action)
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            feature1, feature2 = observation
            nextState = q.build_state([q.to_bin(feature1, feature1_bins),
                             q.to_bin(feature2, feature2_bins)])

            # TODO remove
            if reward != -1:
                print(reward)

            qlearn.learn(state, action, reward, nextState)
            state = nextState
            cumulated_reward += reward

            if done:
                passos_episodio_completados = numpy.append(passos_episodio_completados, int(t+1))
                i_passos_episodios_completados = i_passos_episodios_completados + 1
                break

        print("Episode {:d} reward score: {:0.2f}".format(i_episode, cumulated_reward))
        print("Passos episode completados {:d}".format(i_passos_episodios_completados))

    print("Passos Episodios Completados!")
    print(passos_episodio_completados)
