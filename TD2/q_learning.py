import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """
    best_next = np.max(Q[sprime])
    td_target = r + gamma * best_next
    td_error = td_target - Q[s, a]
    Q[s, a] += alpha * td_error
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    if np.random.random() < epsilone:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[s])


def epsilon_exponential(e, n_epochs, k=0.001):
    """
    Function that implement an exponential decay from 1 -> 0
    eps(e) = exp(-k * e)
    Normalized so eps(0)=1 and eps(n_epochs)=0
    k is a constant that control the speed of decay, k -> 0 means linear decay
    0 < k < 0.01 is a good range to explore
    Exponential may not give the best results but is an intersting choice to explore
    """
    return np.exp(-k * e)


if __name__ == "__main__":
    env = gym.make("Taxi-v3", render_mode=None)

    env.reset()
    env.render()

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    alpha = 0.01 # choose your own

    gamma = 0.8 # choose your own

    epsilon = 0.2 # choose your own

    n_epochs = 10000 # choose your own
    max_itr_per_epoch = 200 # choose your own
    rewards = []

    # plot epsilon decay
    eps_values = [epsilon_exponential(e, n_epochs, 0.0005) for e in range(n_epochs)]
    plt.plot(eps_values)
    plt.title("Exponential epsilon decay (1 -> 0)")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.show()

    for e in range(n_epochs):
        r = 0

        epsilon = epsilon_exponential(e, n_epochs)

        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)

            Sprime, R, done, _, info = env.step(A)

            r += R

            Q = update_q_table(
                Q=Q, s=S, a=A, r=R, sprime=Sprime, alpha=alpha, gamma=gamma
            )

            # Update state and put a stoping criteria
            S = Sprime
            if done:
                break

        print("episode #", e, " : r = ", r)

        rewards.append(r)

    print("Average reward = ", np.mean(rewards))

    # plot the rewards in function of epochs

    plt.plot(rewards)
    plt.xlabel("Epochs")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Epochs")
    plt.show()


    print("Training finished.\n")

    
    """
    
    Evaluate the q-learning algorihtm
    
    """

    env.close()
