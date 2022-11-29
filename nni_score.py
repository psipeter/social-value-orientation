import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plot
import matplotlib.pyplot as plt
import seaborn as sns
import nni

from game import *
from utils import *
from agents import *
from T4T import *

def main(args):
    architecture = args['architecture']
    player = args['player']
    nAgents = args['nAgents']
    nGames = args['nGames']
    seed = args['seed']
    opponent = args['opponent']
    finalGames = nGames - args['nTest']

    agents = []
    for n in range(nAgents):
        if architecture=='DQN':
            agents.append(
                DQN(player=player,
                    ID=f"DQN{n}",
                    seed=seed if nAgents==1 else n,
                    nActions=args['nActions'],
                    nNeurons=args['nNeurons'],
                    tau=args['tau'],
                    alpha=args['alpha'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    update=args['update'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='IBL':
            agents.append(
                IBL(player=player,
                    ID=f"IBL{n}",
                    seed=seed if nAgents==1 else n,
                    nActions=args['nActions'],
                    tau=args['tau'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    decay=args['decay'],
                    sigma=args['sigma'],
                    thrA=args['thrA'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
        elif architecture=='NEF':
            agents.append(
                NEF(player=player,
                    ID=f"NEF{n}",
                    seed=seed if nAgents==1 else n,
                    nEns=args['nEns'],
                    nArr=args['nArr'],
                    nStates=args['nStates'],
                    nActions=args['nActions'],
                    gamma=args['gamma'],
                    explore=args['explore'],
                    nGames=nGames,
                    w_s=args['w_s'],
                    w_o=args['w_o'],
                    w_i=args['w_i']))
    IDs = [agent.ID for agent in agents]
    data = run(agents, nGames=nGames, opponent=opponent, train=True).query("ID in @IDs")
    loss = data.query("game <= @finalGames")['coins'].mean()
    nni.report_final_result(loss)
    # print(loss)

if __name__ == '__main__':
    params = {
        'architecture': 'IBL',
        'opponent': 'greedy',
        'player': 'investor',
        'test': 'ks',
        'wGen': 0.5,
        'wScore': 0.5,
        'nAgents': 30,
        'nGames': 15,
        'nTest': 1,
        'nActions': 11,
        'seed': 0,
        'explore': 'exponential',
        'update': 'SARSA',
        'w_s': 1.0,
        'w_o': 0.0,
        'w_i': 0.0,
    }
    params_fixed = {
        # 'nNeurons': 100
    }
    params = params | params_fixed
    optimized_params = nni.get_next_parameter()
    params.update(optimized_params)
    main(params)

# DQN
# {
#     "nNeurons": {"_type":"randint","_value":[20, 100]},
#     "explore": {"_type":"choice","_value":["linear", "exponential"]},
#     "tau": {"_type":"quniform","_value":[1.0, 20.0, 0.1]},
#     "alpha": {"_type":"qloguniform","_value":[0.001, 1.0, 0.001]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "update": {"_type":"choice","_value":["Q-learning", "SARSA"]},
#     "w_o": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_i": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }

# IBL
# {
#     "seed": {"_type":"randint","_value":[0, 1000]},
#     "thrA": {"_type":"quniform","_value":[-2.0, 1.0, 0.01]},
#     "decay": {"_type":"quniform","_value":[-1.0, 0.0, 0.01]},
#     "sigma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "tau": {"_type":"quniform","_value":[1.0, 10.0, 0.1]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_o": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
#     "w_i": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }

# NEF
# {
#     "nEns": {"_type":"randint","_value":[1000, 1000]},
#     "nArr": {"_type":"randint","_value":[300, 300]},
#     "nStates": {"_type":"randint","_value":[100, 100]},
#     "alpha": {"_type":"qloguniform","_value":[1e-8, 1e-6, 1e-8]},
#     "gamma": {"_type":"quniform","_value":[0.0, 1.0, 0.01]},
# }