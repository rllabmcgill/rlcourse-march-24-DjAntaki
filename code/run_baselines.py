from server_problem import *

env = ServerProblem()
#env = ServerProblem2()
max_iter = 500

evaluate_random_agent(env,max_iter=max_iter)

evaluate_good_agent(env,max_iter=max_iter)
