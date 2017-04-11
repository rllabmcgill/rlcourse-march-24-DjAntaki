import numpy as np
from server_problem import ServerProblem2
from lspi import run_lspi
from plot import plot_policy_update
from utils import onehot, sigmoid
import pickle

env = ServerProblem2()
nactions = env.num_actions

def phi1(x, a):
    y = np.array([int(x[1 + i] > x[1 + a]) for i in range(nactions) if not i == a])
    y_prime = 1- y
    return np.concatenate([[x[0]], y ,y_prime])

def phi2(x, a):
    f = [[x[0]]]
    for aa in range(nactions):
        if aa == a:
            if x[1 + a] > 0:
                y = [float(x[1 + i]) / (x[1 + a] + x[1 + i]) for i in range(nactions) if not i == a]
            else:
                y = np.ones((nactions - 1,))
            y2 = [x[1 + nactions + i] - x[1 + nactions + a] for i in range(nactions) if not i == a]
            f += [y, y2]
        else:
            f.append(np.zeros((2 * nactions - 2,)))

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append([y3])
    f.append(onehot(nactions, a))
    return np.concatenate(f)

def phi3(x, a):

    f = [[x[0]]]
    for aa in range(nactions):
        if aa == a:
            y = np.array([np.tanh(float(x[1 + a]) / (x[1 + i])) if x[1 + i] > 0 else 1 for i in range(nactions) if not i == a])
            if x[1 + a] > 0:
                y2 = np.array([np.tanh(float(x[1 + i]) / (x[1 + a])) for i in range(nactions) if not i == a])
            else:
                y2 = np.ones((nactions - 1,))
#            y2 = [x[1 + nactions + i] - x[1 + nactions + a] for i in range(nactions) if not i == a]
            f += [y, 1-y, y2, 1 - y2]
        else:
            f.append(np.zeros((4 * (nactions - 1),)))

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append([y3])
    f.append(onehot(nactions, a))
    return np.concatenate(f)

def phi5(x,a):
    """bad idea"""
    f = [[x[0]]]
    for aa in range(nactions):
        if aa == a:
            y = np.concatenate([[1],np.log(x[1:1+nactions])])
        else :
            y = np.zeros((nactions+2,))
        f += [y]

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append([y3])
    return np.concatenate(f)

def phi4(x,a):
    f = [[x[0]]]
    for aa in range(nactions):
        if not aa == a :
            if x[1+a] == 0 :
                f += [np.zeros((2,))]
            else :
                if x[1+aa] == 0:
                    f += [[1, 0]]
                else:
                    frac = float(x[1+a]/x[1+aa])
                    f += [[0,frac]]
        else :
            f += [np.zeros((2,))]

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append(onehot(nactions, a))
    f.append([y3])
    return np.concatenate(f)

def phi6(x,a):
    f = [[x[0]]]
    for aa in range(nactions):
        if not aa == a :
            if x[1+a] == 0 :
                f += [np.zeros((12,))]
            else :
                if x[1+aa] == 0:
                    f += [[1], np.zeros(11,)]
                else:
                    frac = float(x[1+a]/x[1+aa])
                    y = np.array([int(frac>j) for j in (0.1, 0.2, 0.25, 1.0/3, 0.5, 1, 2, 3, 4, 5, 10) ])
                    f += [[0],y]
        else :
            f += [np.zeros((12,))]
            pass

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append(onehot(nactions, a))
    f.append([y3])
    return np.concatenate(f)

def phi7(x,a):
    f = [[x[0]]]
    for aa in range(nactions):
        if not aa == a:
            y = np.array([x[1 + i] - x[1 + a] for i in range(nactions) if not i == a])
            f += [y]
        else:
            f += [np.zeros((nactions-1,))]

    y3 = int(x[1 + a] + 1 > env.max_queue_length)
    f.append([y3])
    f.append(onehot(nactions, a))
    return np.concatenate(f)

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

results = run_lspi(env, nactions, phi6, nb_episode=100, nb_policy_update=5)
save_location = 'results.pkl'
print(save_location)
pickle.dump(results, open(save_location, 'wb'))
plot_policy_update(env, results)
