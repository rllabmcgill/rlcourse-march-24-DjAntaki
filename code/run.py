import numpy as np
from server_problem import ServerProblem
from lspi import run_lspi
from plot import plot_policy_update
import pickle

env = ServerProblem()
nactions = env.num_actions
def phi1(x, a):
    y = np.array([int(x[1 + i] > x[1 + a]) for i in range(nactions) if not i == a])
    y_prime = 1- y
    return np.concatenate([[x[0]], y ,y_prime])

def phi2(x, a):
    f = [[x[0]]]
    for aa in range(nactions):
        if aa == a:
            y = np.array([int(x[1 + i] > x[1 + a]) for i in range(nactions) if not i == a])
            y_prime = 1- y
            y = np.concatenate([y ,y_prime])
        else :
            y = np.zeros(2*(nactions-1))
        f += y
    return np.concatenate(f)

def phi3(x, a):
    y = np.array([x[1 + i] - x[1 + a] for i in range(nactions) if not i == a])
    return np.concatenate([[x[0]], y ])


from utils import onehot

def phi8(x,a):
    f = [[x[0]]]
    for aa in range(nactions):
        if aa == a:
            y = np.array([x[1 + i] for i in range(nactions)])
            f += [y]
        else:
            f += [np.zeros((nactions,))]

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append([y3])
    f.append(onehot(nactions, a))
    return np.concatenate(f)

results = run_lspi(env, nactions, phi2 , nb_episode=100, nb_policy_update=5)
file_name = 'results.pkl'
pickle.dump(results, open(file_name, 'wb'))
plot_policy_update(env, results)
