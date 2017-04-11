import numpy as np
from matplotlib import pyplot as plt
from server_problem import evaluate_good_agent, evaluate_random_agent
from operator import itemgetter

def plot_policy_update(env, results, max_iter=500):
    """ results is a tuple (train_results, greedy_results)"""
    rdn_avg_iter_taken, _, rdn_avg_nb_bad_decisions, rdn_avg_r = map(np.average,evaluate_random_agent(env,max_iter))
    good_avg_iter_taken, _, good_avg_nb_bad_decisions, good_avg_r = map(np.average,evaluate_good_agent(env,max_iter))
    nb_updates = len(results[0])
    ax1 = plt.subplot(2,1,1)
    ax1.set_title("Average reward in fct. of # of policy update done")
    ax1.set_xlabel("# of policy update completed")
    ax1.set_ylabel("Reward")

    ax2 = plt.subplot(2,1,2)
    ax2.set_title("Average number of time a task is discarded in fct of # of policy update done")
    ax2.set_ylabel("Avg. nb. of time task discarded")
    ax2.set_xlabel("# of policy update completed")

    ax1.plot((0, nb_updates-1), (rdn_avg_r, rdn_avg_r ), 'r-',label='random')
    ax1.plot((0, nb_updates)-1, (good_avg_r, good_avg_r), 'g-',label='optimal policy')

    xpoints = xrange(nb_updates)
    train_results, greedy_results = results

    ax1.plot(xpoints, map(np.average,map(itemgetter(3),train_results)), 'k-',label="LSPI")
    ax1.plot(xpoints, map(np.average,map(itemgetter(3),greedy_results)) ,'k:',label="greedy eval LSPI")
    #ax1.plot(xpoints, map(np.average,map(itemgetter(3),greedy_results)) ,'k:',label="greedy eval LSPI")

    ax2.plot((0, nb_updates), (rdn_avg_nb_bad_decisions, rdn_avg_nb_bad_decisions), 'r-',label='random agent')
    ax2.plot((0, nb_updates), (good_avg_nb_bad_decisions, good_avg_nb_bad_decisions), 'g-',label='optimal policy')
    ax2.plot(xpoints, map(np.average,map(itemgetter(2),train_results)) ,'k-', label="LSPI")
    ax2.plot(xpoints, map(np.average,map(itemgetter(2),greedy_results)) ,'k:', label="greedy eval LSPI")


    legend = ax2.legend(loc='lower right', shadow=True)
    legend = ax1.legend(loc='lower right', shadow=True)

    plt.show()

if __name__ == "__main__":
    import pickle
    from server_problem import ServerProblem, ServerProblem2
    #env = ServerProblem()
    #results = pickle.load(open("results_prob1_phi3_3.txt","rb"))

    env = ServerProblem2()
    results = pickle.load(open("results_prob2_phi8_1.txt", "rb"))

    plot_policy_update(env,results)