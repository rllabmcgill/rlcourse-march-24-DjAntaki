
from utils import uniform_probs,make_samples
from server_problem import evaluate_server
import numpy as np

def lstdq(D, k, phi, gamma, pi):
    """
    http://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf

    :param D: Source of sample (list of (state, action))
    :param k: Number of basis functions
    :param phi: Basis function
    :param gamma: Discount factor
    :param pi: Policy whose value function is sought
    :return: the weights that describes our policy
    """
    A = np.zeros((k,k))
    b = np.zeros((k,))

    for (s, a, r, s_p) in D:
      next_best_action = np.argmax(pi(s_p))
      A = A + phi(s,a)*(phi(s,a)- gamma*phi(s_p,next_best_action)).T
      b = b + phi(s,a)*r

    pinv = np.linalg.pinv(A)*b
    w_pi = np.dot(pinv,b)
    return w_pi

def preprocess_episode_lspi(episodes):
    D = []
    for j in range(len(episodes)):
        S,A,R,I = episodes[j]
        D.append([(S[i], A[i], R[i], S[i + 1]) for i in range(len(S) - 1)])
    D = reduce(lambda x,y:x+y,D,[])
    return D

def run_lspi(env, nactions, phi, gamma=0.95, nb_policy_update=100,nb_episode=100,max_iter=500,pi_policy='egreedy',epsilon=0.2,evaluate=evaluate_server):

    pi = lambda x: uniform_probs(nactions)

    train_stats, valid_stats = [],[]


    i = -1
    while True :
        i += 1
        print("lspi iteration %i"%i)
        Ds = make_samples(env, pi, n=nb_episode, max_iter=max_iter, pi_policy=pi_policy, epsilon=epsilon)
        train_stats.append(evaluate(Ds))
        D = preprocess_episode_lspi(Ds)
        print("Finished generating samples")
        k = len(phi(D[0][0],D[0][1]))
        w = lspi(D, nactions, k, phi, gamma=gamma, w_0=None)
        print('Done')
        Ds = make_samples(env, pi, n=nb_episode, max_iter=max_iter, pi_policy='greedy')
        valid_stats.append(evaluate(Ds))
        if i > nb_policy_update:
            break
        pi = lambda x: [np.dot(phi(x, i),w) for i in range(nactions)]
    return train_stats,valid_stats


def lspi(D,nactions, k, phi, gamma, w_0=None, epsilon=0.001):
    """

    :param D:
    :param k:
    :param phi:
    :param gamma:
    :param pi_0:
    :param epsilon:
    :return:
    """
    if w_0 is None :
        w = np.zeros((k,))
    else :
        w = w_0
    pi = lambda x: [np.dot(phi(x, i), w) for i in range(nactions)]
    c = 0
    last_norm = 0
    while True :

        c += 1
        w_p = lstdq(D, k, phi, gamma, pi)
        norm = np.linalg.norm(w_p-w)
#        print('norm %f'%norm)
        if (np.linalg.norm(w_p-w) <= epsilon):
            w = w_p
  #          print('took %i iterations'%c)
            break
        elif norm == last_norm:
 #           print("exact same norm")
            break
        last_norm = norm
        pi = lambda x: [np.dot(phi(x, i),w_p) for i in range(nactions)]
        w = w_p
    return w
