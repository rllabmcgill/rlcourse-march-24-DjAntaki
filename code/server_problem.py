import numpy as np
from utils import sample, uniform_probs, get_gaussian_func as gaussian, get_uniform_func as uniform, get_beta_func as beta
from utils import make_samples

def evaluate_server(episodes):
    nb_episodes = len(episodes)
    iter_taken = np.zeros((nb_episodes,))
    nb_decisions = np.zeros((nb_episodes,))
    nb_bad_decisions = np.zeros((nb_episodes,))
    total_reward = np.zeros((nb_episodes,))
    for n in range(nb_episodes):
        S,A,R,I = episodes[n]
        nb_decisions[n] = len(S) -1
        iter_taken[n] = S[-1][0]
        nb_bad_decisions[n] = len(filter(lambda i: i.get("badmove",False),I))
        total_reward[n] = np.sum(R)
    print('average nb of decisions', np.average(nb_decisions) )
    print('average time lasted', np.average(iter_taken))
    print('average bad decisions', np.average(nb_bad_decisions))
    print('average reward', np.average(total_reward))
    return iter_taken, nb_decisions, nb_bad_decisions, total_reward


def evaluate_random_agent(env,max_iter):
    print("random agent :")
    Ds = make_samples(env, pi=None, n=1000, max_iter=max_iter, pi_policy='uniform')
    return evaluate_server(Ds)

def evaluate_good_agent(env,max_iter):
    print("good agent :")
    def pi(s):
        nactions = env.num_actions
        probs = np.array([0 if any([s[1+a] > s[1+i] for i in range(env.n_server)]) else 1 for a in range(nactions)])
        return probs

    Ds = make_samples(env, pi, n=1000, max_iter=max_iter, pi_policy='greedy')
    return evaluate_server(Ds)

class ServerProblem:
    def __init__(self,n_server=3):
        self.n_server = n_server
        self.max_queue_length = 10
        self.time_lam = 2 #
        self.task_lam = 5
        self.num_actions = n_server

    def reset(self):
        # queue is a list of 3-tuple (time inserted, task workload, workload left)
        self.queue = [[] for n in range(self.n_server)]
        self.current_time = 0
        self.time_spent_on_current_task = [0 for _ in range(self.n_server)]# np.zeros((self.n_server))
        return np.zeros((3*self.n_server+1,))

    def get_reward_for_task(self, time_taken, workload):
        return 5.0/time_taken

    def get_state(self):
        def time_since_queue(x):
            if len(x) == 0 :
                return 0
            else :
                return self.current_time - x[0][0]
        return np.array([self.current_time] + map(len, self.queue) + list(map(time_since_queue,self.queue) + self.time_spent_on_current_task),dtype=np.float)

    def is_terminal(self):
        return False

    def step(self, action):
        workload = np.random.poisson(self.task_lam) +1
        reward,info = 0,{}
        assert 0<= action < self.n_server

        if len(self.queue[action]) == self.max_queue_length:
            reward -= 5
            info['badmove'] = True
        else :
            self.queue[action].append([self.current_time, workload, workload])

        # Time until the next task appears
        t = np.random.poisson(self.time_lam)

        for n in range(self.n_server):
            c = t
            while c > 0 and len(self.queue[n]) > 0:
                if self.queue[n][0][2] <= c:
                    start_time, workload, workload_left = self.queue[n].pop(0)
                    c -= workload_left
                    reward += self.get_reward_for_task(self.current_time + workload_left - start_time, workload)
                else:
                    self.queue[n][0][2] -= c
                    break

        self.current_time += t
        isterminal = False
        return self.get_state(), reward, isterminal, info

class ServerProblem2:
    def __init__(self):
        self.n_server = 2
        self.servers_capacity = [gaussian(1.2,0.05),gaussian(0.8,0.05)]
#        self.servers_capacity = [gaussian(1,0.4),gaussian(0.85,0.05),uniform(0.3,0.7),uniform(0.75,1.15)]
        self.max_queue_length = 10
        self.time_lam = 2 #
        self.task_lam = 3
        self.num_actions = self.n_server

    def reset(self):
        # queue is a list of 3-tuple (time inserted, task workload, workload left)
        self.queue = [[] for n in range(self.n_server)]
        self.current_time = 0
        self.time_spent_on_current_task = np.zeros((self.n_server))
        return np.zeros((3*self.n_server+1,))

    def get_reward_for_task(self, time_taken, workload):
        return 5.0/time_taken
        #return 1.0*workload/time_taken

    def get_state(self):
        def time_since_queue(x):
            if len(x) == 0 :
                return 0
            else :
                return self.current_time - x[0][0]
        return np.array([self.current_time] + map(len, self.queue) + list(map(time_since_queue,self.queue) + self.time_spent_on_current_task),dtype=np.float)

    def is_terminal(self):
        return False

    def step(self, action):

        workload = np.random.poisson(self.task_lam) +1
        reward,info = 0,{}
        assert 0<= action < self.n_server

        if len(self.queue[action]) == self.max_queue_length:
            reward -= 5
            info['badmove'] = True
        else :
            self.queue[action].append([self.current_time, workload, workload])
        # Time until the next task appears
        t = np.random.poisson(self.time_lam)

        for n in range(self.n_server):
            for i in range(1,t):
                c = self.servers_capacity[n]()

                while c > 0 and len(self.queue[n]) > 0:
                    if self.queue[n][0][2] <= c:
                        start_time, workload, workload_left = self.queue[n].pop(0)
                        c -= workload_left
                        self.time_spent_on_current_task[n] = 0
                        reward += self.get_reward_for_task(self.current_time + i - start_time, workload)
                    else:
                        self.queue[n][0][2] -= c
                        self.time_spent_on_current_task[n] += 1
                        break


        self.current_time += t
        isterminal = False
        return self.get_state(), reward, isterminal, info

def test():
    nb_iter = 100
    env = ServerProblem()
    s_0 = env.reset()

    policy = lambda x: uniform_probs((env.num_actions))

    s = s_0
    for i in range(nb_iter):
        action = sample(policy(s))
        s, r, t, _ = env.step(action)
        print(i, s, r)
        if t :
            break

def test2():
    nb_iter = 100
    env = ServerProblem2()
    s_0 = env.reset()

    policy = lambda x: uniform_probs((env.num_actions))

    s = s_0
    for i in range(nb_iter):
        action = sample(policy(s))
        s, r, t, _ = env.step(action)
        print(i, s, r)
        if t :
            break


#test()
